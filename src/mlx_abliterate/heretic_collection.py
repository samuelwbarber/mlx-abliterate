"""Original bounded collection over pinned Heretic train400 data.

Uses this project's adapter; no upstream implementation is vendored. Prompt
formatting explicitly disables thinking rather than inferring a response prefix.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unicodedata
from collections.abc import Mapping

from .export import verify_export
from .telemetry import MemoryGuard
from .workflow import _tokenizer


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_training_pair(directory):
    directory = Path(directory)
    lock = json.loads((directory / 'dataset-lock.json').read_text())
    groups, source = {}, {}
    seen = set()
    for group in ('harmful', 'harmless'):
        metadata = lock['datasets'][group]
        subset = metadata['subsets']['train[:400]']
        path = directory / f'{group}-train400.json'
        raw = path.read_bytes()
        if len(raw) > 16 * 1024**2 or _sha(raw) != subset['sha256']:
            raise ValueError('Training file hash or size mismatch')
        prompts = json.loads(raw)['prompts']
        if len(prompts) != 400 or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError('Each standard training group must contain 400 nonempty prompts')
        hashes = [_sha(p.encode()) for p in prompts]
        if hashes != subset['ordered_prompt_sha256']:
            raise ValueError('Training prompt identity mismatch')
        for prompt in prompts:
            normalized = ' '.join(unicodedata.normalize('NFKC', prompt).split())
            fingerprint = _sha(normalized.encode())
            if fingerprint in seen:
                raise ValueError('Duplicate normalized direction prompt')
            seen.add(fingerprint)
        groups[group] = prompts
        source[group] = {'dataset': metadata['repository'], 'revision': metadata['revision'],
                         'split': 'train[:400]', 'column': 'text', 'shuffle': False,
                         'indices': list(range(400)), 'file_sha256': subset['sha256'],
                         'ordered_prompt_sha256': hashes}
    return groups, source


def chat_tokens(tokenizer, prompt):
    value = tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    if isinstance(value, Mapping):
        value = value.get('input_ids')
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 4096
            or any(type(x) is not int or x < 0 for x in value)):
        raise ValueError('Expected flat 1–4096 valid token IDs')
    return list(value)


def estimate_directions(adapter, tokenizer, groups, mx, guard, *, orthogonalize=True):
    from .edits import normalize_direction
    means, token_counts = {}, {}
    for group in ('harmful', 'harmless'):
        mean = {}
        lengths = []
        for count, prompt in enumerate(groups[group], 1):
            tokens = chat_tokens(tokenizer, prompt)
            lengths.append(len(tokens))
            vectors = adapter.collect_last_hidden(tokens)
            if set(vectors) != set(range(65)):
                raise ValueError('Incomplete residual collection')
            for layer in range(64):
                vector = vectors[layer + 1].astype(mx.float32)
                if vector.shape != (5120,) or not mx.all(mx.isfinite(vector)).item():
                    raise ValueError('Invalid collected residual')
                mean[layer] = vector if count == 1 else mean[layer] + (vector - mean[layer]) / count
            mx.eval(mean)
            del vectors, vector
            guard.check(mx)
        if not lengths:
            raise ValueError('Empty training group')
        means[group] = mean
        token_counts[group] = {'count':len(lengths), 'min':min(lengths), 'max':max(lengths),
                               'total':sum(lengths)}
    directions = {}
    for layer in range(64):
        direction = normalize_direction(means['harmful'][layer] - means['harmless'][layer])
        if orthogonalize:
            benign = normalize_direction(means['harmless'][layer])
            direction = normalize_direction(direction - mx.sum(direction * benign) * benign)
        directions[f'layer.{layer}'] = direction
    mx.eval(directions)
    return directions, token_counts


def collect_standard(export, dataset_directory, destination, *, orthogonalize=True):
    from .mlx_adapter import load_export, runtime_versions
    import mlx.core as mx
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    groups, source_lock = load_training_pair(dataset_directory)
    identity = verify_export(Path(export), require_unmodified=True)
    guard = MemoryGuard()
    guard.configure(mx)
    adapter = load_export(Path(export), memory_limit=guard.allocator_limit_bytes,
                          cache_limit=guard.cache_limit_bytes)
    started = time.monotonic()
    directions, lengths = estimate_directions(adapter, _tokenizer(Path(export)), groups, mx, guard,
                                              orthogonalize=orthogonalize)
    fingerprint = _sha(json.dumps(source_lock, sort_keys=True, separators=(',', ':')).encode())
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.heretic-directions-', dir=destination.parent))
    try:
        artifact = staging / 'directions.safetensors'
        mx.save_safetensors(str(artifact), directions)
        report = {'schema_version':1, 'source_manifest_digest':identity['source_manifest_digest'],
                  'export_content_fingerprint':identity['content_fingerprint'],
                  'dataset_fingerprint':fingerprint, 'source_dataset_lock':source_lock,
                  'positive_group':'harmful', 'negative_group':'harmless',
                  'counts':{group:len(prompts) for group,prompts in groups.items()},
                  'token_counts':lengths, 'directions_sha256':_sha(artifact.read_bytes()),
                  'runtime':runtime_versions(), 'elapsed_seconds':time.monotonic()-started,
                  'memory':guard.check(mx), 'orthogonalize_direction':orthogonalize,
                  'method':'normalized difference of last-token layer-output means; optional removal of benign-mean component',
                  'formatting':{'system_prompt':'You are a helpful assistant.', 'enable_thinking':False,
                                'response_prefix_detection':'not performed; explicit closed-thinking template'},
                  'scope':'Standard upstream train400 pair, local MLX collection. No behavioral or refusal-rate result.'}
        (staging/'directions.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        destination.mkdir()
        published=[]
        try:
            for name in ('directions.safetensors','directions.json'):
                target=destination/name
                os.link(staging/name,target)
                published.append(target)
        except BaseException:
            for target in published:
                target.unlink(missing_ok=True)
            try:destination.rmdir()
            except OSError:pass
            raise
        return report
    finally:
        shutil.rmtree(staging,ignore_errors=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export',type=Path)
    parser.add_argument('dataset_directory',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--no-orthogonalize',action='store_true')
    args=parser.parse_args()
    report=collect_standard(args.export,args.dataset_directory,args.destination,
                            orthogonalize=not args.no_orthogonalize)
    print(json.dumps({'counts':report['counts'],'dataset_fingerprint':report['dataset_fingerprint'],
                      'elapsed_seconds':report['elapsed_seconds'],'memory':report['memory']}))


if __name__=='__main__':
    main()
