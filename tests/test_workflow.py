import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlx_abliterate.edits import mx
from mlx_abliterate.workflow import _tokens, collect_directions, drift_pilot


class TokenizerTests(unittest.TestCase):
    def test_list_mapping_and_single_batch(self):
        class Tokenizer:
            def __init__(self, value):
                self.value = value
            def apply_chat_template(self, *args, **kwargs):
                return self.value
        for value in ([1, 2], {'input_ids': [1, 2]}, {'input_ids': [[1, 2]]}):
            self.assertEqual(_tokens(Tokenizer(value), 'text'), [1, 2])
        for value in ({}, [[1], [2]], [True], [], ['not a token']):
            with self.assertRaises(ValueError):
                _tokens(Tokenizer(value), 'text')


class FakeGuard:
    allocator_limit_bytes = 123456
    cache_limit_bytes = 654
    def configure(self, mx):
        pass
    def check(self, mx):
        return {'fixture': True}


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {'input_ids': [1 if messages[0]['content'] == 'positive' else 2]}


class FakeAdapter:
    def __init__(self):
        self.projections = {f'model.language_model.layers.{i}.mlp.down_proj': None for i in range(64)}
        self.strength = 0
        self.resets = 0
        self.seen_directions = None
    def collect_last_hidden(self, tokens):
        if tokens == [2]:
            return {i: mx.zeros((5120,)) for i in range(65)}
        return {i: mx.concatenate([mx.array([float(i), 1.]), mx.zeros((5118,))]) for i in range(65)}
    def last_logits(self, tokens):
        return mx.array([[self.strength, 0.]])
    def set_edits(self, directions, strength):
        self.seen_directions = directions
        self.strength = strength
    def reset_edits(self):
        self.strength = 0
        self.resets += 1


@unittest.skipIf(mx is None, 'optional MLX unavailable')
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.export = self.root / 'export'
        receipt = self.export / '.mlx-abliterate'
        receipt.mkdir(parents=True)
        (receipt / 'export-report.json').write_text(json.dumps({'source_manifest_digest': 'fixture-source'}))
        self.dataset = self.root / 'examples.jsonl'
        self.rows = [
            {'id': 'a', 'prompt': 'positive', 'split': 'direction', 'group': 'positive'},
            {'id': 'b', 'prompt': 'negative', 'split': 'direction', 'group': 'negative'},
            {'id': 'c', 'prompt': 'search', 'split': 'search'},
            {'id': 'd', 'prompt': 'held out', 'split': 'final'},
            {'id': 'e', 'prompt': 'capability', 'split': 'capability'},
        ]
        self.dataset.write_text('\n'.join(json.dumps(row) for row in self.rows))
        self.destination = self.root / 'directions'
        self.adapter = FakeAdapter()
        self.loader = patch('mlx_abliterate.mlx_adapter.load_export', return_value=self.adapter).start()
        patch('mlx_abliterate.mlx_adapter.runtime_versions', return_value={'mlx': 'fixture'}).start()
        patch('mlx_abliterate.workflow._tokenizer', return_value=FakeTokenizer()).start()
        patch('mlx_abliterate.workflow.MemoryGuard', FakeGuard).start()
        self.verifier = patch('mlx_abliterate.workflow.verify_export', return_value={
            'source_manifest_digest': 'fixture-source', 'content_fingerprint': 'fixture-content'}).start()
        self.addCleanup(patch.stopall)

    def collect(self):
        return collect_directions(self.export, self.dataset, self.destination,
                                  positive_group='positive', negative_group='negative')

    def test_residual_mapping_and_pilot_zero(self):
        self.collect()
        self.loader.assert_called_with(self.export, memory_limit=123456, cache_limit=654)
        arrays = mx.load(str(self.destination / 'directions.safetensors'))
        self.assertEqual(len(arrays), 64)
        # Position 1 must map to layer 0; the embedding at position 0 is excluded.
        self.assertAlmostEqual(arrays['layer.0'][0].item(), 2 ** -.5, places=6)
        self.assertAlmostEqual(arrays['layer.63'][0].item(), 64 / (64 ** 2 + 1) ** .5, places=6)
        report = drift_pilot(self.export, self.dataset, self.destination)
        self.assertEqual(report['trials'][0]['mean_kl'], 0)
        self.assertGreater(report['trials'][2]['mean_kl'], 0)
        self.assertEqual(self.adapter.resets, 3)
        self.assertEqual(self.adapter.strength, 0)
        first = self.adapter.seen_directions['model.language_model.layers.0.mlp.down_proj']
        self.assertTrue(mx.array_equal(first, arrays['layer.0']).item())

    def test_existing_artifact_never_overwritten(self):
        self.collect()
        before = (self.destination / 'directions.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.collect()
        self.assertEqual((self.destination / 'directions.json').read_bytes(), before)

    def test_source_data_runtime_and_direction_mismatch_before_model_load(self):
        self.collect()
        metadata_path = self.destination / 'directions.json'
        original = json.loads(metadata_path.read_text())
        for key, value in [('source_manifest_digest', 'wrong'), ('dataset_fingerprint', 'wrong'),
                           ('runtime', {'mlx': 'changed'}), ('directions_sha256', 'wrong'),
                           ('export_content_fingerprint', 'changed')]:
            with self.subTest(key=key):
                metadata = dict(original)
                metadata[key] = value
                metadata_path.write_text(json.dumps(metadata))
                self.loader.reset_mock()
                with self.assertRaises(ValueError):
                    drift_pilot(self.export, self.dataset, self.destination)
                self.loader.assert_not_called()
        metadata_path.write_text(json.dumps(original))
        self.rows[2]['prompt'] += ' changed'
        self.dataset.write_text('\n'.join(json.dumps(row) for row in self.rows))
        with self.assertRaises(ValueError):
            drift_pilot(self.export, self.dataset, self.destination)

    def test_changed_or_modified_export_rejected_before_model_load(self):
        self.collect()
        self.loader.reset_mock()
        self.verifier.return_value = {'source_manifest_digest': 'fixture-source', 'content_fingerprint': 'changed'}
        with self.assertRaisesRegex(ValueError, 'export content'):
            drift_pilot(self.export, self.dataset, self.destination)
        self.loader.assert_not_called()
        self.verifier.side_effect = ValueError('unmodified no-op export required')
        with self.assertRaisesRegex(ValueError, 'unmodified'):
            drift_pilot(self.export, self.dataset, self.destination)
        with self.assertRaisesRegex(ValueError, 'unmodified'):
            collect_directions(self.export, self.dataset, self.root / 'other',
                               positive_group='positive', negative_group='negative')
        self.loader.assert_not_called()
        self.verifier.assert_called_with(self.export, require_unmodified=True)

    def test_strength_order_and_malformed_vectors(self):
        self.collect()
        with self.assertRaises(ValueError):
            drift_pilot(self.export, self.dataset, self.destination, strengths=[.1, 0, .2])
        direction_file = self.destination / 'directions.safetensors'
        mx.save_safetensors(str(direction_file), {'layer.0': mx.array([1., 0.])})
        metadata_file = self.destination / 'directions.json'
        metadata = json.loads(metadata_file.read_text())
        metadata['directions_sha256'] = hashlib.sha256(direction_file.read_bytes()).hexdigest()
        metadata_file.write_text(json.dumps(metadata))
        self.loader.reset_mock()
        with self.assertRaises(ValueError):
            drift_pilot(self.export, self.dataset, self.destination)
        self.loader.assert_not_called()
