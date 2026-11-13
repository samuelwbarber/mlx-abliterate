"""Bounded greedy text generation and teacher-forced scoring on one adapter.

Results contain local prompt-derived output and must not be committed. Calls
preserve the caller's edits but never share recurrent, KV or positional state.
No generated code is executed here. A completed response is not a quality pass.
"""
from __future__ import annotations

import time

from .telemetry import MemoryGuard


def _ids(values, name, *, nonempty=True):
    if (not isinstance(values, (list, tuple)) or (nonempty and not values)
            or any(type(x) is not int or not 0 <= x < 2**31 for x in values)):
        raise ValueError(f'{name} must contain nonnegative int32 token IDs')
    return list(values)


def _budget(prompt, output_count, total):
    if type(total) is not int or not 1 <= total <= 4096:
        raise ValueError('max_total_tokens must be 1–4096')
    if type(output_count) is not int or not 1 <= output_count <= 1024:
        raise ValueError('Output budget must be 1–1024 tokens')
    if len(prompt) + output_count > total:
        raise ValueError('Prompt plus output budget exceeds max_total_tokens')


def _clear_position(language):
    for field in ('_rope_deltas', '_position_ids'):
        if hasattr(language, field):
            setattr(language, field, None)


def _step(language, mx, tokens, cache, guard):
    output = language(mx.array([tokens], dtype=mx.int32), cache=cache, logits_to_keep=1)
    logits = output.logits[:, -1, :].astype(mx.float32)
    mx.eval(logits)
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] == 0:
        raise ValueError('Expected one nonempty vocabulary distribution')
    if not mx.all(mx.isfinite(logits)).item():
        raise ValueError('Nonfinite generation logits')
    guard.check(mx)
    return logits


def generate(adapter, token_ids, *, eos_token_ids, max_new_tokens=128,
             max_total_tokens=4096, tokenizer=None, guard=None):
    """Greedy decode with explicit termination and fresh state.

    EOS is excluded from token_ids; sampled_tokens includes the terminal EOS.
    Only EOS termination yields completed (or empty); exhausting the requested
    budget is truncated even if decoded text looks complete. Runtime failures
    return runtime_error and error_type with partial output; admission errors
    raise ValueError. The caller may supply a tokenizer for local decoded text.
    External process supervision is still required for a hard memory/time cap.
    """
    prompt = _ids(token_ids, 'Prompt')
    stops = set(_ids(eos_token_ids, 'EOS'))
    _budget(prompt, max_new_tokens, max_total_tokens)
    import mlx.core as mx
    guard = guard or MemoryGuard()
    language = adapter.model.language_model
    tokens, sampled, cache = [], 0, None
    start = time.monotonic()
    report = {'status': 'runtime_error'}
    try:
        _clear_position(language)
        cache = language.make_cache()
        inputs = prompt
        for _ in range(max_new_tokens):
            logits = _step(language, mx, inputs, cache, guard)
            next_token = int(mx.argmax(logits[0]).item())
            del logits
            sampled += 1
            if next_token in stops:
                report['status'] = 'completed' if tokens else 'empty'
                break
            tokens.append(next_token)
            inputs = [next_token]
        else:
            report['status'] = 'truncated'
        if tokenizer is not None:
            report['text'] = tokenizer.decode(tokens, skip_special_tokens=True)
            if report['status'] == 'completed' and not report['text'].strip():
                report['status'] = 'empty'
    except Exception as error:
        report.update(status='runtime_error', error_type=type(error).__name__)
    finally:
        cache = None
        _clear_position(language)
    report.update(token_ids=tokens, sampled_tokens=sampled,
                  prompt_tokens=len(prompt), elapsed_seconds=time.monotonic() - start)
    return report


def teacher_forced_nll(adapter, token_ids, target_ids, *, max_total_tokens=4096, guard=None):
    """Score a prescribed continuation one token at a time with bounded logits.

    Mean target NLL is a retention diagnostic, not a refusal/quality verdict.
    All targets, including a caller-supplied EOS, are scored. No baseline logits
    or complete sequence-by-vocabulary tensor are retained. Errors are explicit
    and cannot produce an eligible mean score. Caller edits remain installed.
    """
    prompt = _ids(token_ids, 'Prompt')
    targets = _ids(target_ids, 'Target')
    _budget(prompt, len(targets), max_total_tokens)
    import mlx.core as mx
    guard = guard or MemoryGuard()
    language = adapter.model.language_model
    values, cache = [], None
    start = time.monotonic()
    report = {'status': 'runtime_error'}
    try:
        _clear_position(language)
        cache = language.make_cache()
        inputs = prompt
        for target in targets:
            logits = _step(language, mx, inputs, cache, guard)[0]
            if target >= logits.shape[0]:
                raise ValueError('Target token outside vocabulary')
            shifted = logits - mx.max(logits)
            nll = mx.logsumexp(shifted) - shifted[target]
            if not mx.isfinite(nll).item():
                raise ValueError('Nonfinite target likelihood')
            values.append(float(nll.item()))
            del logits, shifted, nll
            inputs = [target]
        report.update(status='completed', mean_nll=sum(values) / len(values), nll_per_token=values)
    except Exception as error:
        report.update(status='runtime_error', error_type=type(error).__name__, scored_tokens=len(values))
    finally:
        cache = None
        _clear_position(language)
    report.update(prompt_tokens=len(prompt), target_tokens=len(targets),
                  elapsed_seconds=time.monotonic() - start)
    return report


def evaluate_export(export, prompt_file, *, target_file=None, max_new_tokens=128):
    """CLI bridge: verify an unchanged or edited export before local inference."""
    import hashlib
    import json
    from pathlib import Path
    from .export import verify_export
    from .mlx_adapter import load_export, runtime_versions
    from .workflow import _tokenizer, _tokens

    def read_text(path):
        with Path(path).open('rb') as stream:
            raw = stream.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise ValueError('Prompt/target file exceeds 64 KiB')
        text = raw.decode('utf-8')
        if not text.strip():
            raise ValueError('Prompt/target must be nonblank')
        return text

    prompt = read_text(prompt_file)
    target = read_text(target_file) if target_file is not None else None
    export = Path(export)
    identity = verify_export(export, require_unmodified=False)
    tokenizer = _tokenizer(export)
    prompt_ids = _tokens(tokenizer, prompt)
    if target is not None:
        target_ids = _ids(tokenizer.encode(target, add_special_tokens=False), 'Target')
        _budget(prompt_ids, len(target_ids), 4096)
    else:
        _budget(prompt_ids, max_new_tokens, 4096)
        eos = tokenizer.eos_token_id
        stops = list(eos) if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
        configuration = export / 'generation_config.json'
        if configuration.exists():
            if configuration.stat().st_size > 1024 * 1024:
                raise ValueError('Generation configuration exceeds 1 MiB')
            configured = json.loads(configuration.read_text()).get('eos_token_id')
            if configured is not None:
                stops.extend(configured if isinstance(configured, list) else [configured])
        stops = sorted(set(_ids(stops, 'EOS')))
    import mlx.core as mx
    guard = MemoryGuard()
    guard.configure(mx)
    adapter = load_export(export, memory_limit=guard.allocator_limit_bytes,
                          cache_limit=guard.cache_limit_bytes)
    if target is None:
        report = generate(adapter, prompt_ids, eos_token_ids=stops,
                          max_new_tokens=max_new_tokens, tokenizer=tokenizer, guard=guard)
    else:
        report = teacher_forced_nll(adapter, prompt_ids, target_ids, guard=guard)
        report['target_sha256'] = hashlib.sha256(target.encode()).hexdigest()
    report.update(schema_version=1, source_identity=identity, runtime=runtime_versions(),
                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                  scope='Local experimental text inference; no generated code execution or quality verdict.')
    return report
