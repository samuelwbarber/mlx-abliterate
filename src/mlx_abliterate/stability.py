"""Bounded synthetic-prefill stability probe; no behavioral quality verdict.

Run with ``python -m mlx_abliterate.stability --export outputs/stock-noop
--output runs/stability.json`` after unloading other model servers. The parent
process watches child RSS and elapsed time; only its own worker can be killed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import tempfile
import time

from .telemetry import GIB, MemoryGuard


def validate_config(lengths, cycles, tolerance_bytes):
    if (not lengths or len(set(lengths)) != len(lengths)
            or any(type(n) is not int or not 1 <= n <= 4096 for n in lengths)):
        raise ValueError('Lengths must be distinct integers in 1–4096')
    if type(cycles) is not int or not 3 <= cycles <= 100:
        raise ValueError('Use 3–100 measured cycles per length')
    if type(tolerance_bytes) is not int or tolerance_bytes < 0:
        raise ValueError('Allocation tolerance must be nonnegative integer bytes')


def run_trials(adapter, mx, prompts, *, cycles=5, tolerance_bytes=16 * 1024**2,
               guard=None, collect=False):
    """Warm each shape, then compare post-reset live allocation across cycles.

    Synthetic nonzero edits exercise all wrappers but are not learned directions.
    Every adapter call makes fresh recurrent/KV state. Cached allocator bytes are
    recorded separately; caches are not artificially cleared between trials.
    """
    validate_config([len(p) for p in prompts], cycles, tolerance_bytes)
    if type(collect) is not bool:
        raise ValueError('collect must be boolean')
    guard = guard or MemoryGuard()
    width = next(iter(adapter.projections.values())).weight.shape[0]
    direction = mx.ones((width,), dtype=mx.float32)
    directions = {path: direction for path in adapter.projections}
    mx.eval(direction)
    results = []
    try:
        for prompt in prompts:
            baseline = adapter.last_logits(prompt)
            if not mx.all(mx.isfinite(baseline)).item():
                raise ValueError('Nonfinite baseline')
            collection = None
            if collect:
                started = time.monotonic()
                before = guard.check(mx)
                vectors = adapter.collect_last_hidden(prompt)
                layers = adapter.preflight['config']['text_config']['num_hidden_layers']
                if set(vectors) != set(range(layers + 1)):
                    raise ValueError('Incomplete residual positions')
                for vector in vectors.values():
                    if vector.shape != (width,) or not mx.all(mx.isfinite(vector)).item():
                        raise ValueError('Invalid residual shape or nonfinite residual')
                observed = guard.check(mx)
                collection_seconds = time.monotonic() - started
                del vector, vectors
                restored = adapter.last_logits(prompt)
                difference = mx.max(mx.abs(restored - baseline)).item()
                if difference != 0.:
                    raise ValueError('Post-collection fidelity gate failed')
                del restored
                gc.collect()
                collection = {'seconds': collection_seconds, 'residual_count': layers + 1,
                              'residual_width': width, 'restored_max_logit_delta': difference,
                              'memory_before': before, 'memory_with_residuals': observed,
                              'memory_after_release': guard.check(mx)}
            records = []
            # First cycle warms every edit/shape combination; exclude its memory.
            for cycle in range(cycles + 1):
                started = time.monotonic()
                errors = []
                for strength in (0., .05, .1):
                    try:
                        adapter.set_edits(directions, strength)
                        candidate = adapter.last_logits(prompt)
                        if not mx.all(mx.isfinite(candidate)).item():
                            raise ValueError('Nonfinite candidate')
                        difference = mx.max(mx.abs(baseline - candidate)).item()
                        if strength == 0. and difference != 0.:
                            raise ValueError('Zero-edit fidelity gate failed')
                        errors.append(difference)
                        del candidate
                    finally:
                        adapter.reset_edits()
                    guard.check(mx)
                restored = adapter.last_logits(prompt)
                if mx.max(mx.abs(restored - baseline)).item() != 0.:
                    raise ValueError('Restored-model fidelity gate failed')
                del restored
                gc.collect()
                observed = guard.check(mx)
                records.append({'cycle': cycle, 'warmup': cycle == 0,
                                'seconds': time.monotonic() - started,
                                'max_logit_delta_by_strength': errors,
                                'memory': observed})
            active = [r['memory']['mlx_active_bytes'] for r in records[1:]]
            spread = max(active) - min(active)
            results.append({'token_count': len(prompt), 'cycles': records,
                            'collection': collection,
                            'measured_active_spread_bytes': spread,
                            'allocation_plateau_passed': spread <= tolerance_bytes})
            del baseline
    finally:
        adapter.reset_edits()
    return {'status': 'passed' if all(r['allocation_plateau_passed'] for r in results) else 'failed',
            'lengths': results, 'strengths': [0., .05, .1], 'collection_enabled': collect,
            'allocation_spread_tolerance_bytes': tolerance_bytes,
            'scope': 'Synthetic prefill and reversible-edit stability only; no generation, quality, or serving claims.'}


def _worker(export, output, lengths, cycles, tolerance, collect):
    report = {'status': 'failed'}
    try:
        from .export import verify_export
        from .mlx_adapter import load_export, runtime_versions
        import mlx.core as mx
        from transformers import AutoTokenizer
        identity = verify_export(Path(export), require_unmodified=True)
        tokenizer = AutoTokenizer.from_pretrained(export, local_files_only=True, trust_remote_code=False)
        unit = tokenizer.encode('Explain how a Python list stores values.', add_special_tokens=False)
        if not unit:
            raise ValueError('Empty synthetic token sequence')
        prompts = [(unit * ((length + len(unit) - 1) // len(unit)))[:length] for length in lengths]
        guard = MemoryGuard()
        guard.configure(mx)
        start = time.monotonic()
        adapter = load_export(Path(export), memory_limit=guard.allocator_limit_bytes,
                              cache_limit=guard.cache_limit_bytes)
        load_seconds = time.monotonic() - start
        report = run_trials(adapter, mx, prompts, cycles=cycles,
                            tolerance_bytes=tolerance, guard=guard, collect=collect)
        report.update(schema_version=1, runtime=runtime_versions(), source_identity=identity,
                      load_seconds=load_seconds, final_memory=guard.check(mx),
                      synthetic_input_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest())
    except Exception as error:
        report = {'status': 'failed', 'error_type': type(error).__name__}
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def supervise(export, *, lengths=(32, 256, 1024), cycles=5,
              tolerance_bytes=16 * 1024**2, timeout_seconds=600,
              process_limit_bytes=34 * GIB, collect=False):
    """Run one child; terminate only that child on timeout or RSS breach.

    Requires ps RSS in KiB (macOS/Linux). External supervision supplements, but
    cannot make allocation OOM-proof, the adapter's phase-boundary checks.
    """
    validate_config(lengths, cycles, tolerance_bytes)
    if type(collect) is not bool:
        raise ValueError('collect must be boolean')
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('Timeout must be finite and positive')
    if type(process_limit_bytes) is not int or not 0 < process_limit_bytes <= 34 * GIB:
        raise ValueError('Process budget must be positive and at most 34 GiB')
    with tempfile.TemporaryDirectory(prefix='mlx-stability-') as directory:
        output = str(Path(directory) / 'result.json')
        process = multiprocessing.get_context('spawn').Process(
            target=_worker, args=(str(export), output, lengths, cycles, tolerance_bytes, collect))
        process.start()
        start = time.monotonic()
        sampled_peak = 0
        failure = None
        try:
            while process.is_alive():
                process.join(.2)
                if not process.is_alive():
                    break
                if time.monotonic() - start > timeout_seconds:
                    failure = 'timeout'
                    break
                try:
                    usage = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)],
                                           capture_output=True, text=True, timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    failure = 'watchdog_unavailable'
                    break
                if usage.returncode or not usage.stdout.strip():
                    if process.is_alive():
                        failure = 'watchdog_unavailable'
                    break
                try:
                    rss = int(usage.stdout.strip()) * 1024
                    if rss < 0:
                        raise ValueError('Negative RSS')
                except ValueError:
                    failure = 'watchdog_unavailable'
                    break
                sampled_peak = max(sampled_peak, rss)
                if sampled_peak > process_limit_bytes:
                    failure = 'memory_guard_abort'
                    break
        finally:
            if process.is_alive():
                process.terminate()
                process.join(2)
                if process.is_alive():
                    process.kill()
            process.join()
        report = (json.loads(Path(output).read_text()) if Path(output).exists()
                  else {'status': 'failed', 'error_type': 'WorkerExitedWithoutReport'})
        if process.exitcode != 0:
            report['status'] = 'failed'
            report['error_type'] = 'WorkerExitedUncleanly'
        if failure:
            report = {'status': 'failed', 'error_type': failure}
        report['watchdog'] = {'sampled_peak_rss_bytes': sampled_peak,
                              'process_limit_bytes': process_limit_bytes,
                              'elapsed_seconds': time.monotonic() - start,
                              'timeout_seconds': timeout_seconds, 'exitcode': process.exitcode}
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--lengths', nargs='+', type=int, default=[32, 256, 1024])
    parser.add_argument('--collect', action='store_true', help='Validate bounded residual collection at each length')
    parser.add_argument('--cycles', type=int, default=5)
    parser.add_argument('--timeout-seconds', type=float, default=600)
    args = parser.parse_args()
    # Reserve before expensive loading; never overwrite a previous report.
    with args.output.open('x') as stream:
        report = supervise(args.export, lengths=args.lengths, cycles=args.cycles,
                           timeout_seconds=args.timeout_seconds, collect=args.collect)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
