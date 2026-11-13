import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_abliterate.export import ExportError
from mlx_abliterate.merge_validation import quantization_control_spec, export_pair


class MergeValidationTests(unittest.TestCase):
    def test_control_preserves_selection_scale_and_original_spec(self):
        spec = {'scale_policy': 'preserve', 'source_manifest_digest': 'sha256:base',
                'projections': {'selected': {'strength': .7, 'direction': [1., 0.]}}}
        control = quantization_control_spec(spec)
        self.assertEqual(control['projections']['selected']['strength'], 0.)
        self.assertTrue(control['force_requantize'])
        self.assertEqual(control['scale_policy'], 'preserve')
        self.assertEqual(set(control['projections']), {'selected'})
        control['projections']['selected']['direction'][0] = 9.
        self.assertEqual(spec['projections']['selected'], {'strength': .7, 'direction': [1., 0.]})

    def test_pair_rejects_same_destination_without_writing(self):
        with TemporaryDirectory() as folder, patch('mlx_abliterate.merge_validation.merge_export') as merge:
            destination = Path(folder) / 'candidate'
            with self.assertRaises(ExportError):
                export_pair(Path(folder) / 'source', destination, destination, {})
            merge.assert_not_called()

    def test_existing_control_prevents_partial_candidate_write(self):
        with TemporaryDirectory() as folder, patch('mlx_abliterate.merge_validation.merge_export') as merge:
            control = Path(folder) / 'control'
            control.mkdir()
            with self.assertRaises(FileExistsError):
                export_pair(Path(folder) / 'source', Path(folder) / 'candidate', control, {})
            merge.assert_not_called()

    def test_pair_verifies_both_and_forces_only_control_requantization(self):
        with TemporaryDirectory() as folder, patch('mlx_abliterate.merge_validation.merge_export', return_value={}) as merge, patch('mlx_abliterate.merge_validation.verify_export', return_value={'verified': True}) as verify:
            source, candidate, control = [Path(folder) / name for name in ('source', 'candidate', 'control')]
            spec = {'projections': {'p': {'strength': .25, 'direction': [1.]}}}
            report = export_pair(source, candidate, control, spec)
            self.assertEqual(merge.call_count, 2)
            self.assertEqual(merge.call_args_list[0].args[2], spec)
            self.assertEqual(merge.call_args_list[1].args[2]['projections']['p']['strength'], 0.)
            self.assertTrue(merge.call_args_list[1].args[2]['force_requantize'])
            self.assertEqual([c.args[0] for c in verify.call_args_list], [candidate, control])
            self.assertFalse(report['quality_validated'])
            self.assertFalse(report['imported'])
