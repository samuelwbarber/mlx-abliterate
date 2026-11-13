import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from mlx_abliterate.checkpoint_comparison import compare_checkpoints, logit_metrics, _npy


def npy(values):
    header = repr({'descr':'<f4','fortran_order':False,'shape':(1,len(values))}).encode()+b'\n'
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<'+'f'*len(values),*values)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.artifacts={}
        for label in ('stock','dynamic','edited','control'):
            folder=self.root/label
            folder.mkdir()
            raw=npy([1.,2.,3.])
            (folder/'0000.npy').write_bytes(raw)
            identity={'source_identity':{'source_manifest_digest':'sha256:base','content_fingerprint':label if label in ('edited','control') else 'stock','modified':label in ('edited','control')},'dataset_fingerprint':'data','rubric_sha256':'rubric','runtime':{'mlx':'version'},'edit_spec_sha256':'edit' if label=='dynamic' else None}
            record={'id':'row','file':'0000.npy','prompt_sha256':'prompt','tokens_sha256':'tokens','shape':[1,3],'sha256':hashlib.sha256(raw).hexdigest()}
            manifest={'schema_version':1,**identity,'records':[record]}
            manifest_bytes=json.dumps(manifest).encode()
            (folder/'manifest.json').write_bytes(manifest_bytes)
            report={'schema_version':1,'status':'completed',**identity,'row_order':['row'],'splits':['final','capability'],'max_new_tokens':96,'results':[{'id':'row','kind':'exact','status':'completed','task_check_passed':True,'response_sha256':'answer'}],'logits':{'count':1,'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest()}}
            path=self.root/(label+'.json')
            path.write_text(json.dumps(report))
            self.artifacts[label]=(path,folder)

    def test_exact_four_way_control(self):
        report=compare_checkpoints(self.artifacts)
        self.assertTrue(report['quantization_control_exact'])
        self.assertTrue(report['comparisons']['dynamic_to_edited']['tasks']['retained_baseline_correct_tasks'])

    def test_corrupt_logits_fail_before_metrics(self):
        path=self.artifacts['edited'][1]/'0000.npy'
        path.write_bytes(path.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError,'file SHA'):
            compare_checkpoints(self.artifacts)

    def test_manifest_corruption_is_detected(self):
        path=self.artifacts['edited'][1]/'manifest.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'manifest SHA'):
            compare_checkpoints(self.artifacts)

    def test_prompt_tokens_must_match_even_with_valid_manifest_hash(self):
        report_path,folder=self.artifacts['edited']
        manifest=json.loads((folder/'manifest.json').read_text())
        manifest['records'][0]['tokens_sha256']='other'
        raw=json.dumps(manifest).encode()
        (folder/'manifest.json').write_bytes(raw)
        report=json.loads(report_path.read_text())
        report['logits']['manifest_sha256']=hashlib.sha256(raw).hexdigest()
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'tokens_sha256'):
            compare_checkpoints(self.artifacts)

    def test_retention_regression_reported(self):
        path,_=self.artifacts['edited']
        report=json.loads(path.read_text())
        report['results'][0]['task_check_passed']=False
        path.write_text(json.dumps(report))
        comparison=compare_checkpoints(self.artifacts)['comparisons']['stock_to_edited']['tasks']
        self.assertFalse(comparison['retained_baseline_correct_tasks'])
        self.assertEqual(comparison['regressed_task_ids'],['row'])

    def test_kl_shift_invariance_and_nonzero_drift(self):
        same=logit_metrics([1.,2.,3.],[101.,102.,103.])
        self.assertEqual(same['kl_base_to_candidate'],0.)
        self.assertEqual(same['max_abs_logit_delta'],100.)
        self.assertGreater(logit_metrics([1.,2.,3.],[3.,2.,1.])['kl_base_to_candidate'],0.)

    def test_npy_nonfinite_or_truncated_rejected(self):
        for raw in (npy([float('nan')]),npy([1.])[:-1]):
            with self.assertRaises(ValueError):
                _npy(raw,[1,1])

    def test_mlx_fortran_single_row_layout(self):
        header = repr({'descr':'<f4','fortran_order':True,'shape':(1,3)}).encode()+b'\n'
        raw = b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<fff',1.,2.,3.)
        self.assertEqual(list(_npy(raw,[1,3])),[1.,2.,3.])
