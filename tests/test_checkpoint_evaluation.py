import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlx_abliterate.checkpoint_evaluation import evaluate_checkpoint, main

try:
    import mlx.core as mx
except ImportError:
    mx = None


class Guard:
    allocator_limit_bytes = 100
    cache_limit_bytes = 10
    def configure(self, mx):
        pass
    def check(self, mx):
        return {'fixture': True}


@unittest.skipIf(mx is None, 'MLX unavailable')
class CheckpointTests(unittest.TestCase):
    def test_relative_dynamic_export_passes_real_source_admission(self):
        # Use the actual byte-preserving tiny export fixture, including receipt,
        # shards and runtime metadata. Do not mock verify_export/_source_entries.
        from test_merge import MergeExportTests
        fixture = MergeExportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        spec = fixture.base / 'edit.json'
        spec.write_text(json.dumps(fixture.spec))
        relative = Path(os.path.relpath(fixture.source, Path.cwd()))
        self.assertFalse(relative.is_absolute())
        data = SimpleNamespace(fingerprint='data', splits={'final': [], 'capability': []})
        with patch('mlx_abliterate.checkpoint_evaluation.load_dataset_splits', return_value=data), \
                patch('mlx_abliterate.checkpoint_evaluation.load_rubrics', return_value={'rubrics': {}, 'sha256': 'rubric'}), \
                patch('mlx_abliterate.checkpoint_evaluation._tokenizer', side_effect=RuntimeError('admitted')) as tokenizer:
            with self.assertRaisesRegex(RuntimeError, '^admitted$'):
                evaluate_checkpoint(relative, 'data', 'rubric', edit_spec=spec)
        tokenizer.assert_called_once_with(fixture.source.resolve())

    def test_all_capability_rows_hash_only_reports_and_logits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{'id': f'c{i}', 'prompt': f'PRIVATE PROMPT {i}'} for i in range(7)]
            data = SimpleNamespace(fingerprint='data', splits={'final': rows[:1], 'capability': rows[1:]})
            tokenizer = SimpleNamespace(eos_token_id=2)
            resets = []
            adapter = SimpleNamespace(last_logits=lambda _: mx.array([[1., 2., 3.]]),
                                      reset_edits=lambda: resets.append(True))
            results = [{'id': r['id'], 'kind': 'exact', 'status': 'completed',
                        'task_check_passed': True, 'response_sha256': 'hash'} for r in rows]
            with patch('mlx_abliterate.checkpoint_evaluation.load_dataset_splits', return_value=data), \
                    patch('mlx_abliterate.checkpoint_evaluation.load_rubrics', return_value={'rubrics': {}, 'sha256': 'rubric'}), \
                    patch('mlx_abliterate.checkpoint_evaluation.verify_export', return_value={'content_fingerprint': 'edited'}) as verify, \
                    patch('mlx_abliterate.checkpoint_evaluation._tokenizer', return_value=tokenizer), \
                    patch('mlx_abliterate.checkpoint_evaluation._tokens', return_value=[1, 2]), \
                    patch('mlx_abliterate.checkpoint_evaluation.MemoryGuard', Guard), \
                    patch('mlx_abliterate.mlx_adapter.load_export', return_value=adapter), \
                    patch('mlx_abliterate.checkpoint_evaluation.evaluate_tasks', return_value=results) as tasks:
                report = evaluate_checkpoint(root, 'data', 'rubric', logits_dir=root / 'logits')
            self.assertEqual(report['row_order'], [r['id'] for r in rows])
            self.assertEqual(tasks.call_args.args[2], rows)
            self.assertEqual(report['summary']['correct'], 7)
            self.assertEqual(resets, [True])
            self.assertEqual(len(list((root / 'logits').glob('*.npy'))), 7)
            manifest = json.loads((root / 'logits/manifest.json').read_text())
            self.assertEqual(len(manifest['records']), 7)
            self.assertNotIn('PRIVATE PROMPT', json.dumps(report) + json.dumps(manifest))
            verify.assert_called_once_with(root.resolve(), require_unmodified=False)
            with self.assertRaises(FileExistsError):
                evaluate_checkpoint(root, 'data', 'rubric', logits_dir=root / 'logits')


class CheckpointCliTests(unittest.TestCase):
    def test_failed_run_leaves_explicit_report_and_reraises(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'failed.json'
            arguments = ['export', 'data', 'rubrics', '--output', str(output)]
            with patch('mlx_abliterate.checkpoint_evaluation.evaluate_checkpoint',
                       side_effect=ValueError('private failure details')):
                with self.assertRaisesRegex(ValueError, 'private failure details'):
                    main(arguments)
            self.assertEqual(json.loads(output.read_text()),
                             {'schema_version': 1, 'status': 'failed', 'error_type': 'ValueError'})
            with patch('mlx_abliterate.checkpoint_evaluation.evaluate_checkpoint') as run:
                with self.assertRaises(FileExistsError):
                    main(arguments)
                run.assert_not_called()
