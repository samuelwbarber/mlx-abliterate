"""Local collection and drift pilot; no automatic refusal or capability claims."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from .evaluate import load_dataset_splits, first_token_kl
from .telemetry import MemoryGuard, estimate_remaining
from .export import verify_export


def _tokenizer(export: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(export), local_files_only=True, trust_remote_code=False)


def _tokens(tokenizer, prompt: str) -> list[int]:
    tokens = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                          tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False)
    if isinstance(tokens, Mapping):
        tokens = tokens.get('input_ids')
    if isinstance(tokens, (list, tuple)) and len(tokens) == 1 and isinstance(tokens[0], (list, tuple)):
        tokens = tokens[0]
    if (not isinstance(tokens, (list, tuple)) or not 1 <= len(tokens) <= 4096
            or any(type(token) is not int or token < 0 for token in tokens)):
        raise ValueError('Tokenized pilot prompt must contain a flat list of 1–4096 integer token IDs')
    return list(tokens)


def collect_directions(export: Path, dataset: Path, destination: Path, *,
                       positive_group: str, negative_group: str) -> dict:
    """Estimate normalized positive-minus-negative last residual directions.

    User-provided group labels are not interpreted as evidence of refusal.
    Each layer keeps two width-sized means, never complete prompt activations.
    """
    from .mlx_adapter import load_export, runtime_versions
    from .edits import normalize_direction
    import mlx.core as mx
    export, destination = Path(export), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if positive_group == negative_group:
        raise ValueError('Contrast groups must differ')
    data = load_dataset_splits(dataset)
    rows = data.splits['direction']
    counts = {group: sum(row['group'] == group for row in rows)
              for group in (positive_group, negative_group)}
    if not all(counts.values()):
        raise ValueError('Both selected contrast groups must contain examples')
    source_identity = verify_export(export, require_unmodified=True)
    tokenizer = _tokenizer(export)
    guard = MemoryGuard()
    guard.configure(mx)
    adapter = load_export(export, memory_limit=guard.allocator_limit_bytes,
                          cache_limit=guard.cache_limit_bytes)
    sums = {group: {} for group in counts}
    started = time.monotonic()
    for row in rows:
        group = row['group']
        if group not in sums:
            continue
        vectors = adapter.collect_last_hidden(_tokens(tokenizer, row['prompt']))
        for position, vector in vectors.items():
            if position == 0:
                continue  # Position zero is the embedding, not a decoder output.
            layer = position - 1
            previous = sums[group].get(layer, mx.zeros_like(vector, dtype=mx.float32))
            sums[group][layer] = previous + vector.astype(mx.float32)
        mx.eval(sums[group])
        guard.check(mx)
    directions = {}
    for layer in range(64):
        contrast = (sums[positive_group][layer] / counts[positive_group]
                    - sums[negative_group][layer] / counts[negative_group])
        directions[f'layer.{layer}'] = normalize_direction(contrast)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.directions-', dir=destination.parent))
    try:
        mx.save_safetensors(str(staging / 'directions.safetensors'), directions)
        sha = hashlib.sha256((staging / 'directions.safetensors').read_bytes()).hexdigest()
        report = {'schema_version': 1, 'source_manifest_digest': source_identity['source_manifest_digest'],
                  'export_content_fingerprint': source_identity['content_fingerprint'],
                  'dataset_fingerprint': data.fingerprint, 'dataset_lock': data.lock(),
                  'positive_group': positive_group, 'negative_group': negative_group,
                  'counts': counts, 'directions_sha256': sha, 'runtime': runtime_versions(),
                  'elapsed_seconds': time.monotonic() - started, 'memory': guard.check(mx),
                  'method': 'normalized difference of layer-output last-token means',
                  'scope': 'Direction extraction only; effectiveness and quality are not evaluated.'}
        (staging / 'directions.json').write_text(json.dumps(report, indent=2) + '\n')
        # Exclusive reservation prevents rename from replacing a concurrently
        # created empty directory. The metadata file is the completion marker.
        destination.mkdir()
        published = []
        try:
            for name in ('directions.safetensors', 'directions.json'):
                target = destination / name
                os.link(staging / name, target)
                published.append(target)
        except BaseException:
            for target in published:
                target.unlink(missing_ok=True)
            try:
                destination.rmdir()
            except OSError:
                pass  # Never remove another writer's unexpected files.
            raise
        shutil.rmtree(staging)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def drift_pilot(export: Path, dataset: Path, directions_directory: Path, *,
                strengths: list[float] | None = None, max_prompts: int = 8) -> dict:
    """Measure three or more edit trials using first-token KL, with fresh caches.

    Baseline logits are spooled to disk one prompt at a time. No candidate is
    selected: KL is not a measure of reduced refusal or preserved coding skill.
    """
    from .mlx_adapter import load_export, runtime_versions
    import mlx.core as mx
    export, directions_directory = Path(export), Path(directions_directory)
    strengths = [0., .05, .1] if strengths is None else strengths
    if len(strengths) < 3 or strengths[0] != 0:
        raise ValueError('Pilot requires zero first and at least three strengths')
    from .edits import _strength
    strengths = [_strength(value) for value in strengths]
    if type(max_prompts) is not int or not 1 <= max_prompts <= 64:
        raise ValueError('max_prompts must be 1–64')
    data = load_dataset_splits(dataset)
    metadata = json.loads((directions_directory / 'directions.json').read_text())
    source_identity = verify_export(export, require_unmodified=True)
    if (metadata['source_manifest_digest'] != source_identity['source_manifest_digest']
            or metadata.get('export_content_fingerprint') != source_identity['content_fingerprint']
            or metadata['dataset_fingerprint'] != data.fingerprint):
        raise ValueError('Direction source, export content, or dataset fingerprint mismatch')
    if metadata.get('runtime') != runtime_versions():
        raise ValueError('Direction runtime fingerprint mismatch')
    direction_file = directions_directory / 'directions.safetensors'
    if direction_file.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Direction artifact exceeds bounded size')
    if hashlib.sha256(direction_file.read_bytes()).hexdigest() != metadata['directions_sha256']:
        raise ValueError('Direction artifact hash mismatch')
    arrays = mx.load(str(direction_file))
    expected = {f'layer.{i}' for i in range(64)}
    if set(arrays) != expected:
        raise ValueError('Direction artifact must contain exactly 64 layer vectors')
    for vector in arrays.values():
        if vector.shape != (5120,) or not mx.all(mx.isfinite(vector)).item():
            raise ValueError('Direction vectors must be finite with width 5120')
        norm = mx.sqrt(mx.sum(vector.astype(mx.float32) ** 2)).item()
        if abs(norm - 1.) > 1e-4:
            raise ValueError('Direction vectors must have unit norm')
    tokenizer = _tokenizer(export)
    tokens = [_tokens(tokenizer, row['prompt']) for row in data.splits['search'][:max_prompts]]
    guard = MemoryGuard()
    guard.configure(mx)
    adapter = load_export(export, memory_limit=guard.allocator_limit_bytes,
                          cache_limit=guard.cache_limit_bytes)
    directions = {path: arrays[f'layer.{int(path.split(".")[3])}'] for path in adapter.projections}
    records = []
    with tempfile.TemporaryDirectory(prefix='mlx-baseline-') as folder:
        for index, prompt in enumerate(tokens):
            baseline = adapter.last_logits(prompt).astype(mx.float32)
            mx.save(str(Path(folder) / f'{index}.npy'), baseline)
            guard.check(mx)
        for strength in strengths:
            started = time.monotonic()
            values = []
            try:
                adapter.set_edits(directions, strength)
                for index, prompt in enumerate(tokens):
                    candidate = adapter.last_logits(prompt)
                    baseline = mx.load(str(Path(folder) / f'{index}.npy'))
                    values.append(first_token_kl(baseline, candidate).item())
                    # A zero-edit run is a fidelity gate, not just another
                    # diagnostic. KL alone misses a constant logit offset.
                    if strength == 0 and not mx.array_equal(
                            baseline, candidate.astype(mx.float32)).item():
                        raise ValueError('Zero-strength control changed baseline logits; pilot stopped')
                    guard.check(mx)
            finally:
                adapter.reset_edits()
            records.append({'strength': strength, 'kl_per_prompt': values,
                            'mean_kl': sum(values) / len(values),
                            'wall_seconds': time.monotonic() - started,
                            'memory': guard.check(mx)})
    return {'schema_version': 1, 'source_manifest_digest': source_identity['source_manifest_digest'],
            'export_content_fingerprint': source_identity['content_fingerprint'],
            'dataset_fingerprint': data.fingerprint, 'runtime': runtime_versions(),
            'trials': records, 'trial_only_50_more': estimate_remaining([r['wall_seconds'] for r in records], 50),
            'scope': 'First-token drift pilot only; no generation, refusal scoring, capability verdict, or selected model.'}
