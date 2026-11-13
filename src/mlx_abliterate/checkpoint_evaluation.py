"""Evaluate a verified checkpoint on final and all capability rows, without text logs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .cli import write_report
from .evaluate import load_dataset_splits
from .export import verify_export
from .study import evaluate_tasks, summarize
from .study_evaluation import load_rubrics
from .telemetry import MemoryGuard
from .workflow import _tokenizer, _tokens


def evaluate_checkpoint(export, dataset, rubric_file, *, edit_spec=None,
                        logits_dir=None, max_new_tokens=96):
    """One loaded model; dynamic edits require the immutable unchanged base.

    Logit files are one float32 last-position distribution per final/capability
    prompt. Their manifest contains hashes and IDs, never prompt/response text.
    An incomplete logits directory has no manifest and must not be compared.
    """
    from .mlx_adapter import load_export, runtime_versions
    import mlx.core as mx
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 1024:
        raise ValueError('max_new_tokens must be 1–1024')
    export = Path(export)
    if logits_dir is not None:
        logits_dir = Path(logits_dir)
        if logits_dir.exists() or logits_dir.is_symlink():
            raise FileExistsError(logits_dir)
    data = load_dataset_splits(dataset)
    rubric_document = load_rubrics(Path(rubric_file), data)
    rows = list(data.splits['final']) + list(data.splits['capability'])
    identity = verify_export(export, require_unmodified=edit_spec is not None)
    spec_hash, edits = None, None
    if edit_spec is not None:
        from .merge import _source_entries, _validate_spec, read_edit_spec
        spec, spec_hash = read_edit_spec(edit_spec)
        receipt, entries = _source_entries(export)
        edits, _, force = _validate_spec(spec, receipt, entries)
        if force:
            raise ValueError('Dynamic evaluation cannot simulate forced requantization')
    tokenizer = _tokenizer(export)
    eos = tokenizer.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
    config = export / 'generation_config.json'
    if config.exists():
        configured = json.loads(config.read_text()).get('eos_token_id', [])
        eos.extend(configured if isinstance(configured, list) else [configured])
    if not eos or any(type(x) is not int or x < 0 for x in eos):
        raise ValueError('Invalid or missing EOS configuration')
    eos = sorted(set(eos))
    guard = MemoryGuard()
    guard.configure(mx)
    started = time.monotonic()
    adapter = load_export(export, memory_limit=guard.allocator_limit_bytes,
                          cache_limit=guard.cache_limit_bytes)
    load_seconds = time.monotonic() - started
    logits_records = []
    try:
        if edits:
            # Zero-strength bundles do not need a direction and are exact no-ops.
            selected = {path: edit for path, edit in edits.items() if edit['strength'] != 0}
            if selected:
                adapter.set_edits({path: mx.array(edit['direction'], dtype=mx.float32)
                                   for path, edit in selected.items()},
                                  {path: edit['strength'] for path, edit in selected.items()})
        if logits_dir is not None:
            logits_dir.mkdir(parents=True, exist_ok=False)
            for index, row in enumerate(rows):
                tokens = _tokens(tokenizer, row['prompt'])
                logits = adapter.last_logits(tokens).astype(mx.float32)
                if not mx.all(mx.isfinite(logits)).item():
                    raise ValueError('Nonfinite checkpoint logits')
                name = f'{index:04d}.npy'
                mx.save(str(logits_dir / name), logits)
                logits_records.append({'id': row['id'], 'file': name,
                    'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(),
                    'tokens_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    'shape': list(logits.shape),
                    'sha256': hashlib.sha256((logits_dir / name).read_bytes()).hexdigest()})
                del logits
                guard.check(mx)
        results = evaluate_tasks(adapter, tokenizer, rows, rubric_document['rubrics'], eos,
                                 max_new_tokens=max_new_tokens)
        summary = summarize(results)
        memory = guard.check(mx)
    finally:
        adapter.reset_edits()
    identity_record = {'source_identity': identity, 'dataset_fingerprint': data.fingerprint,
                       'rubric_sha256': rubric_document['sha256'], 'edit_spec_sha256': spec_hash,
                       'runtime': runtime_versions()}
    report = {'schema_version': 1, 'status': 'completed' if summary['failures'] == 0 else 'failed',
              **identity_record, 'summary': summary, 'results': results,
              'row_order': [row['id'] for row in rows], 'splits': ['final', 'capability'],
              'max_new_tokens': max_new_tokens, 'load_seconds': load_seconds,
              'elapsed_seconds': time.monotonic() - started, 'memory': memory,
              'scope': 'Final plus all capability diagnostics; refusal likelihood is a proxy, not a behavioral verdict.'}
    if logits_dir is not None:
        write_report({'schema_version': 1, **identity_record, 'records': logits_records},
                     logits_dir / 'manifest.json')
        report['logits'] = {'count': len(logits_records),
                           'manifest_sha256': hashlib.sha256((logits_dir / 'manifest.json').read_bytes()).hexdigest()}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('rubric_file', type=Path)
    parser.add_argument('--edit-spec', type=Path)
    parser.add_argument('--logits-dir', type=Path)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before loading; exclusive ownership also prevents concurrent runs
    # from publishing two evaluations as the same experiment.
    with args.output.open('x', encoding='utf-8') as stream:
        report = evaluate_checkpoint(args.export, args.dataset, args.rubric_file,
                                     edit_spec=args.edit_spec, logits_dir=args.logits_dir,
                                     max_new_tokens=args.max_new_tokens)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 0 if report['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
