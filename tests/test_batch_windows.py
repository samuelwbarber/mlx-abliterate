from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlx_abliterate.batch_windows import RawGreedyProcessor, batch_generate, benchmark_tokens, compare_token_windows

try:
    import mlx.core as mx
except ImportError:
    mx = None


class Guard:
    def check(self, mx):
        return {}


class ComparisonTests(unittest.TestCase):
    def test_divergence_missing_rows_and_status_cannot_pass(self):
        base = {'token_ids': [1,2,3], 'status': 'truncated', 'sampled_tokens': 3}
        self.assertTrue(compare_token_windows([base], [dict(base)])['exact'])
        changed = {**base, 'token_ids': [1,2,4]}
        result = compare_token_windows([base], [changed])
        self.assertFalse(result['exact'])
        self.assertEqual(result['differences'][0]['first_differing_token_index'], 2)
        for other in ([], [{**base, 'status': 'runtime_error'}], [{**base, 'sampled_tokens': 4}],
                      [{**base, 'token_ids': [1,2]}]):
            self.assertFalse(compare_token_windows([base], other)['exact'])
        self.assertFalse(compare_token_windows([], [])['exact'])


@unittest.skipIf(mx is None, 'MLX unavailable')
class BatchTests(unittest.TestCase):
    def test_raw_argmax_and_finite_guard(self):
        from mlx_vlm.sample_utils import greedy_argmax
        processor = RawGreedyProcessor(mx)
        # Native normalization can introduce ties; synthetic winner must match
        # the sequential raw-float32 argmax, including an existing exact tie.
        for values in ([[1., 1.000001, -3.]], [[1., 1., -2.]], [[-1000., -999., -998.]]):
            logits = mx.array(values, dtype=mx.float32)
            sampled = greedy_argmax(processor(None, logits))
            self.assertEqual(sampled.tolist(), mx.argmax(logits, axis=-1).tolist())
            processor.check()
        processor(None, mx.array([[0., float('nan')]]))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            processor.check()

    def test_template_matches_closed_thinking_contract(self):
        calls = []
        tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: calls.append((messages, kw)) or [1,2])
        self.assertEqual(benchmark_tokens(tokenizer, 'test'), [1,2])
        self.assertEqual(calls[0][0][0], {'role':'system','content':'You are a helpful assistant.'})
        self.assertFalse(calls[0][1]['enable_thinking'])

    def test_native_scheduler_contract_and_cleanup(self):
        instances = []
        class Scheduler:
            def __init__(self, model, processor, **kwargs):
                self.kwargs = kwargs
                self.processor = processor
                self.has_work = True
                self.step = 0
                self.closed = False
                instances.append(self)
            def insert(self, prompts, max_tokens, prompt_kwargs):
                from mlx_vlm.generate.ar import _merge_prefill_prompt_kwargs
                # Exercise native admission instead of mocking its contract.
                embeds, _ = _merge_prefill_prompt_kwargs(prompt_kwargs, prompts)
                self.embed_shape = embeds.shape
                self.prompts = prompts
                return [10, 20]
            def next(self):
                self.step += 1
                if self.step == 1:
                    return [], [SimpleNamespace(uid=10, token=1, finish_reason=None),
                                SimpleNamespace(uid=20, token=1, finish_reason=None)]
                self.has_work = False
                return [], [SimpleNamespace(uid=10, token=2, finish_reason='stop'),
                            SimpleNamespace(uid=20, token=0, finish_reason='length')]
            def close(self):
                self.closed = True
        language = SimpleNamespace(_position_ids='stale', _rope_deltas='stale',
            model=SimpleNamespace(embed_tokens=lambda ids: mx.ones((*ids.shape, 4))))
        adapter = SimpleNamespace(model=SimpleNamespace(language_model=language))
        tokenizer = SimpleNamespace(eos_token_id=999, decode=lambda ids, **kw: str(ids))
        with patch('mlx_vlm.generate.ar.BatchGenerator', Scheduler):
            report = batch_generate(adapter, tokenizer, [[1], [1,0]], eos_token_ids=[2],
                                    max_new_tokens=2, guard=Guard())
        self.assertEqual([r['status'] for r in report['rows']], ['completed','truncated'])
        self.assertEqual([r['token_ids'] for r in report['rows']], [[1],[1,0]])
        self.assertEqual(instances[0].prompts, [[1],[1,0]])
        self.assertEqual(instances[0].embed_shape, (2,2,4))
        self.assertTrue(instances[0].closed)
        self.assertFalse(instances[0].kwargs['manage_wired_limit'])
        self.assertIsNone(instances[0].kwargs['apc_manager'])
        self.assertIsNone(instances[0].kwargs['kv_bits'])
        self.assertIsNone(instances[0].kwargs['draft_model'])
        self.assertFalse(instances[0].processor.stopping_criteria(999))
        self.assertIsNone(language._position_ids)

    def test_scheduler_error_invalidates_all_rows(self):
        class Broken:
            def __init__(self, *args, **kwargs):
                self.has_work = True
            def insert(self, *args, **kwargs):
                return [1]
            def next(self):
                raise RuntimeError('private error text')
            def close(self):
                pass
        language = SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: mx.ones((*ids.shape, 4))))
        adapter = SimpleNamespace(model=SimpleNamespace(language_model=language))
        with patch('mlx_vlm.generate.ar.BatchGenerator', Broken):
            result = batch_generate(adapter, SimpleNamespace(decode=lambda *a,**k:''), [[1]],
                                    eos_token_ids=[2], guard=Guard())
        self.assertEqual(result['rows'][0]['status'], 'runtime_error')
        self.assertEqual(result['rows'][0]['error_type'], 'RuntimeError')
        self.assertNotIn('private error text', str(result))
