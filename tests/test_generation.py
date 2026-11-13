import math
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlx_abliterate.generation import evaluate_export, generate, teacher_forced_nll

try:
    import mlx.core as mx
except ImportError:
    mx = None


class Guard:
    allocator_limit_bytes = 123
    cache_limit_bytes = 12
    def configure(self, mx):
        pass
    def check(self, mx):
        return {}


class Language:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []
        self.caches = []
        self._rope_deltas = 'stale'
        self._position_ids = 'stale'

    def make_cache(self):
        cache = []
        self.caches.append(cache)
        return cache

    def __call__(self, inputs, *, cache, logits_to_keep):
        self.calls.append((inputs.tolist(), cache, logits_to_keep))
        value = self.outputs[len(cache)]
        cache.append(True)
        if isinstance(value, Exception):
            raise value
        self._position_ids = 'inference position'
        return SimpleNamespace(logits=mx.array([[value]], dtype=mx.float32))


@unittest.skipIf(mx is None, 'MLX unavailable')
class GenerationTests(unittest.TestCase):
    def adapter(self, outputs):
        language = Language(outputs)
        return SimpleNamespace(model=SimpleNamespace(language_model=language))

    def test_eos_and_fresh_cache_and_position_reset(self):
        adapter = self.adapter([[0, 4, 0], [0, 0, 4]])
        for _ in range(2):
            result = generate(adapter, [0, 1], eos_token_ids=[2], max_new_tokens=3, guard=Guard())
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['token_ids'], [1])
            self.assertEqual(result['sampled_tokens'], 2)
        language = adapter.model.language_model
        self.assertIsNot(language.caches[0], language.caches[1])
        self.assertEqual([c[0] for c in language.calls], [[[0, 1]], [[1]], [[0, 1]], [[1]]])
        self.assertIsNone(language._position_ids)
        self.assertIsNone(language._rope_deltas)

    def test_empty_and_truncated_are_distinct(self):
        for outputs, limit, status in [([[0, 0, 4]], 1, 'empty'), ([[0, 4, 0]], 1, 'truncated')]:
            result = generate(self.adapter(outputs), [0], eos_token_ids=[2],
                              max_new_tokens=limit, guard=Guard())
            self.assertEqual(result['status'], status)

    def test_runtime_errors_and_nonfinite_output_are_not_successes(self):
        for failure in (RuntimeError('private exception text'), [float('nan'), 1, 0]):
            adapter = self.adapter([[0, 4, 0], failure])
            report = generate(adapter, [0], eos_token_ids=[2], max_new_tokens=3, guard=Guard())
            self.assertEqual(report['status'], 'runtime_error')
            self.assertEqual(report['token_ids'], [1])
            self.assertIn('error_type', report)
            self.assertNotIn('private exception text', str(report))
            self.assertIsNone(adapter.model.language_model._position_ids)

    def test_decoded_empty_text(self):
        tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: '  ')
        report = generate(self.adapter([[0, 4, 0], [0, 0, 4]]), [0], eos_token_ids=[2],
                          max_new_tokens=3, tokenizer=tokenizer, guard=Guard())
        self.assertEqual(report['status'], 'empty')

    def test_admission_before_inference(self):
        adapter = self.adapter([])
        for kwargs in ({'token_ids': [True]}, {'eos_token_ids': []}, {'max_new_tokens': 0},
                       {'max_total_tokens': 4097}, {'token_ids': [0] * 4096}):
            args = dict(token_ids=[0], eos_token_ids=[2], max_new_tokens=3)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                generate(adapter, **args)
        self.assertFalse(adapter.model.language_model.calls)

    def test_teacher_forcing_uses_target_not_argmax(self):
        adapter = self.adapter([[2, 0, 0], [0, 0, 0]])
        report = teacher_forced_nll(adapter, [0], [1, 2], guard=Guard())
        self.assertEqual(report['status'], 'completed')
        expected = (math.log(math.exp(2) + 2) + math.log(3)) / 2
        self.assertAlmostEqual(report['mean_nll'], expected, places=6)
        self.assertEqual(adapter.model.language_model.calls[1][0], [[1]])
        self.assertIsNone(adapter.model.language_model._position_ids)

    def test_teacher_forcing_failure_has_no_eligible_score(self):
        adapter = self.adapter([[0, 0, 0], RuntimeError('failure')])
        result = teacher_forced_nll(adapter, [0], [1, 2], guard=Guard())
        self.assertEqual(result['status'], 'runtime_error')
        self.assertEqual(result['scored_tokens'], 1)
        self.assertNotIn('mean_nll', result)

    def test_export_bridge_verifies_edited_input_and_uses_generation_eos(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prompt = root / 'prompt.txt'
            prompt.write_text('A local prompt')
            (root / 'generation_config.json').write_text(json.dumps({'eos_token_id': [1, 2]}))
            tokenizer = SimpleNamespace(eos_token_id=2,
                apply_chat_template=lambda *a, **k: [0], decode=lambda *a, **k: '')
            with patch('mlx_abliterate.export.verify_export', return_value={'content_fingerprint': 'edited'}) as verify, \
                    patch('mlx_abliterate.workflow._tokenizer', return_value=tokenizer), \
                    patch('mlx_abliterate.mlx_adapter.load_export', return_value=self.adapter([[0, 4, 0]])), \
                    patch('mlx_abliterate.generation.MemoryGuard', Guard):
                report = evaluate_export(root, prompt, max_new_tokens=2)
            verify.assert_called_once_with(root, require_unmodified=False)
            self.assertEqual(report['status'], 'empty')
            self.assertEqual(report['sampled_tokens'], 1)
            self.assertEqual(report['source_identity']['content_fingerprint'], 'edited')
            self.assertNotIn('A local prompt', str(report))
