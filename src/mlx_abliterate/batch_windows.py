"""Experimental batched windows using installed MLX-VLM; require token fidelity.

No upstream implementation is copied. One weights instance, fresh generator per
call, no prefix reuse, drafter, cache quantization, or wired-limit changes. Local
probe reports contain token IDs and must remain outside version control.

Experimental status: the 2026-11-14 four-prompt real-model probe rejected both
batch sizes 2 and 4. Stock matched 3/4 complete token windows; dynamic edits
matched 2/4. Observed speed ratios were 1.70–2.56, but those timings do not
establish an equivalent replacement. The sequential search remains authoritative.
Tiny tests validate orchestration and sampler rules, not full-model fidelity.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

from .generation import _ids, _budget, _clear_position, generate
from .telemetry import MemoryGuard


def benchmark_tokens(tokenizer, prompt, *, system_prompt='You are a helpful assistant.'):
    tokens = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(tokens, 'get'):
        tokens = tokens.get('input_ids')
    if isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], list):
        tokens = tokens[0]
    return _ids(tokens, 'Prompt')


class RawGreedyProcessor:
    """Preserve raw float32 argmax despite upstream normalized greedy rounding.

    Return unique winner scores solely to the sampling stage; model logits and
    caches are unchanged. Deferred finite flags are checked before results escape.
    """
    def __init__(self, mx):
        self.mx = mx
        self.flags = []

    def __call__(self, tokens, logits):
        mx = self.mx
        logits = logits.astype(mx.float32)
        self.flags.append(mx.all(mx.isfinite(logits)))
        winner = mx.argmax(logits, axis=-1, keepdims=True)
        return mx.where(mx.arange(logits.shape[-1]) == winner, 0., -1.)

    def check(self):
        if self.flags:
            good = self.mx.all(self.mx.stack(self.flags)).item()
            self.flags.clear()
            if not good:
                raise ValueError('Nonfinite batched logits')


def batch_generate(adapter, tokenizer, prompts, *, eos_token_ids, batch_size=2,
                   max_new_tokens=100, guard=None, timeout_seconds=600):
    """Generate unpadded token-list inputs, returning local per-row token IDs.

    Installed BatchGenerator owns left-padding and hybrid cache handling. Any
    failure invalidates the entire batch, including already finished peers.
    """
    if type(batch_size) is not int or batch_size not in (1, 2, 4):
        raise ValueError('batch_size must be 1, 2 or 4')
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be finite and positive')
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError('Expected nonempty prompt list')
    prompts = [_ids(p, 'Prompt') for p in prompts]
    for prompt in prompts:
        _budget(prompt, max_new_tokens, 4096)
    stops = set(_ids(eos_token_ids, 'EOS'))
    import mlx.core as mx
    from mlx_vlm.generate.ar import BatchGenerator
    from mlx_vlm.utils import StoppingCriteria
    guard = guard or MemoryGuard()
    language = adapter.model.language_model
    # Installed criteria otherwise unions tokenizer EOS into explicit stops.
    # Sequential generate uses only the caller list, so preserve that contract.
    processor = SimpleNamespace(stopping_criteria=StoppingCriteria(
        list(stops), tokenizer=SimpleNamespace(eos_token_id=None)))
    checker = RawGreedyProcessor(mx)
    generator = None
    started = time.monotonic()
    rows = [{'status': 'runtime_error', 'token_ids': [], 'sampled_tokens': 0,
             'prompt_tokens': len(p)} for p in prompts]
    pending = set()
    try:
        _clear_position(language)
        generator = BatchGenerator(language, processor, max_tokens=max_new_tokens,
            stop_tokens=stops, completion_batch_size=batch_size, prefill_batch_size=batch_size,
            prefill_step_size=None, compute_logprobs=False, greedy_sampling=True,
            logits_processors=[checker], kv_bits=None, apc_manager=None,
            draft_model=None, manage_wired_limit=False)
        indices = {}
        next_index = 0
        while generator.has_work or next_index < len(prompts):
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError('Batch deadline exceeded')
            # Native scheduler requires embeddings even for text-only input.
            # Queue at most one active batch so long datasets do not retain
            # all prompt embeddings. This calls the exact ordinary embedding.
            count = min(batch_size - len(pending), len(prompts) - next_index)
            if count:
                incoming = prompts[next_index:next_index + count]
                kwargs = [{'inputs_embeds': language.model.embed_tokens(mx.array([p], dtype=mx.int32))}
                          for p in incoming]
                mx.eval([item['inputs_embeds'] for item in kwargs])
                guard.check(mx)
                uids = generator.insert(incoming, max_tokens=max_new_tokens, prompt_kwargs=kwargs)
                if len(uids) != count or len(set(uids)) != count or set(uids) & set(indices):
                    raise ValueError('Batch generator returned invalid row identities')
                indices.update({uid: next_index + i for i, uid in enumerate(uids)})
                pending.update(uids)
                next_index += count
                del kwargs, incoming
            _, responses = generator.next()
            checker.check()
            guard.check(mx)
            for response in responses:
                if response.uid not in pending:
                    raise ValueError('Unexpected or repeated finished batch row')
                row = rows[indices[response.uid]]
                row['sampled_tokens'] += 1
                if response.token not in stops:
                    row['token_ids'].append(int(response.token))
                if response.finish_reason is not None:
                    if response.finish_reason not in ('stop', 'length'):
                        raise ValueError('Unknown batch finish reason')
                    row['status'] = 'completed' if response.finish_reason == 'stop' else 'truncated'
                    row['text'] = tokenizer.decode(row['token_ids'], skip_special_tokens=True)
                    if row['status'] == 'completed' and not row['text'].strip():
                        row['status'] = 'empty'
                    pending.remove(response.uid)
        checker.check()
        if pending:
            raise ValueError('Batch ended without all finish events')
    except Exception as error:
        for row in rows:
            row.update(status='runtime_error', error_type=type(error).__name__)
    finally:
        if generator is not None:
            generator.close()
        generator = None
        checker.flags.clear()
        _clear_position(language)
        gc.collect()
    return {'rows': rows, 'batch_size': batch_size, 'elapsed_seconds': time.monotonic() - started,
            'status': 'completed' if all(r['status'] in ('completed', 'truncated', 'empty') for r in rows) else 'runtime_error',
            'memory': guard.check(mx)}


def compare_token_windows(sequential, batched):
    """CPU-only exact comparison; summarize divergence without response text."""
    differences = []
    for index in range(max(len(sequential), len(batched))):
        if index >= len(sequential) or index >= len(batched):
            differences.append({'row': index, 'reason': 'missing_row'})
            continue
        base, candidate = sequential[index], batched[index]
        left, right = base['token_ids'], candidate['token_ids']
        first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
        if first is None and len(left) != len(right):
            first = min(len(left), len(right))
        valid = base['status'] in ('completed', 'truncated') and candidate['status'] in ('completed', 'truncated')
        if (not valid or first is not None or base['status'] != candidate['status']
                or base['sampled_tokens'] != candidate['sampled_tokens']):
            differences.append({'row': index, 'reason': 'window_mismatch',
                'first_differing_token_index': first,
                'sequential_tokens': len(left), 'batch_tokens': len(right),
                'sequential_status': base['status'], 'batch_status': candidate['status'],
                'sampled_tokens_match': base['sampled_tokens'] == candidate['sampled_tokens']})
    return {'exact': bool(sequential) and not differences,
            'matched_rows': max(len(sequential), len(batched)) - len(differences),
            'differences': differences}


def fidelity_probe(adapter, tokenizer, prompts, *, eos_token_ids, sizes=(2, 4), max_new_tokens=100):
    """Compare actual complete token windows to current sequential implementation."""
    start = time.monotonic()
    sequential = [generate(adapter, p, eos_token_ids=eos_token_ids,
                           max_new_tokens=max_new_tokens, tokenizer=tokenizer) for p in prompts]
    seconds = time.monotonic() - start
    report = {'sequential_seconds': seconds, 'sequential': sequential, 'batches': []}
    valid = all(r['status'] in ('completed', 'truncated') for r in sequential)
    for size in sizes:
        measured = batch_generate(adapter, tokenizer, prompts, eos_token_ids=eos_token_ids,
                                  batch_size=size, max_new_tokens=max_new_tokens)
        measured['comparison'] = compare_token_windows(sequential, measured['rows'])
        measured['exact_tokens_and_status'] = measured['comparison']['exact']
        measured['speed_ratio'] = seconds / measured['elapsed_seconds']
        report['batches'].append(measured)
    report['passed'] = valid and all(b['exact_tokens_and_status'] for b in report['batches'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    parser.add_argument('--prompts', required=True, type=Path, help='Local JSON list of benign probe prompts')
    parser.add_argument('--edit-spec', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-new-tokens', type=int, default=100)
    args = parser.parse_args(argv)
    from .export import verify_export
    from .mlx_adapter import load_export, runtime_versions
    from .workflow import _tokenizer
    from .merge import read_edit_spec, _source_entries, _validate_spec
    import mlx.core as mx
    with args.output.open('x') as stream:
        try:
            export = args.export.resolve()
            identity = verify_export(export, require_unmodified=args.edit_spec is not None)
            prompts = json.loads(args.prompts.read_text())
            if not isinstance(prompts, list) or not 2 <= len(prompts) <= 16 or any(not isinstance(p, str) or not p.strip() for p in prompts):
                raise ValueError('Provide 2–16 nonblank benign probe prompts')
            tokenizer = _tokenizer(export)
            tokens = [benchmark_tokens(tokenizer, p) for p in prompts]
            eos = json.loads((export / 'generation_config.json').read_text())['eos_token_id']
            if isinstance(eos, int):
                eos = [eos]
            guard = MemoryGuard()
            guard.configure(mx)
            adapter = load_export(export, memory_limit=guard.allocator_limit_bytes, cache_limit=guard.cache_limit_bytes)
            states = {'stock': fidelity_probe(adapter, tokenizer, tokens, eos_token_ids=eos, max_new_tokens=args.max_new_tokens)}
            spec_hash = None
            if args.edit_spec:
                spec, spec_hash = read_edit_spec(args.edit_spec)
                receipt, entries = _source_entries(export)
                edits, _, force = _validate_spec(spec, receipt, entries)
                if force:
                    raise ValueError('Cannot simulate forced requantization')
                selected = {p: e for p, e in edits.items() if e['strength']}
                try:
                    if selected:
                        adapter.set_edits({p: mx.array(e['direction']) for p,e in selected.items()},
                                          {p: e['strength'] for p,e in selected.items()})
                    states['dynamic'] = fidelity_probe(adapter, tokenizer, tokens, eos_token_ids=eos, max_new_tokens=args.max_new_tokens)
                finally:
                    adapter.reset_edits()
            report = {'passed': all(s['passed'] for s in states.values()), 'states': states,
                      'source_identity': identity, 'runtime': runtime_versions(), 'edit_spec_sha256': spec_hash,
                      'prompt_hashes': [hashlib.sha256(p.encode()).hexdigest() for p in prompts]}
            payload = json.dumps(report, indent=2, allow_nan=False)
        except BaseException as error:
            json.dump({'passed': False, 'error_type': type(error).__name__}, stream)
            raise
        stream.write(payload + '\n')
    print(json.dumps({'passed': report['passed'], 'output': str(args.output)}))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
