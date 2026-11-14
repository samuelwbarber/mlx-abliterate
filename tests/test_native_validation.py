import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlx_abliterate.native_validation import validate_isolation, compare_exact_responses, compare_native_speed, quality_gate, verify_native_import, run_speed


def benchmark(version='stock'):
    def sample(index,repetition,phase):
        return {'status':'completed','response_sha256':'a'*64,'thinking_sha256':'b'*64,
                'eval_count':10,'prompt_eval_count':5,'prompt_index':index,'repetition':repetition,
                'phase':phase,'decode_tokens_per_second':50.}
    return {'schema_version':1,'status':'complete','ollama_version':version,'settings':{'think':False},
            'prompt_sha256':['p1','p2'],'environment':{},'cold_samples':[sample(0,0,'cold')],
            'warmup_samples':[sample(i,0,'warmup') for i in range(2)],
            'samples':[sample(i,r,'warm') for r in range(5) for i in range(2)]}


class NativeValidationTests(unittest.TestCase):
    def test_balanced_speed_includes_every_block_and_reverses_order(self):
        with patch('mlx_abliterate.native_validation.Servers') as servers, \
             patch('mlx_abliterate.native_validation.run_benchmark', side_effect=lambda tag,*a,**kw: benchmark(tag)) as measure:
            servers.return_value.original='original';servers.return_value.isolated='isolated'
            servers.return_value.isolation={}
            servers.return_value.digest.side_effect=lambda url,tag: tag+'-digest'
            result=run_speed('original','isolated','stock','edited',['p1','p2'],blocks=4)
            self.assertTrue(result['passed'])
            self.assertEqual([call.args[0] for call in measure.call_args_list],
                             ['stock','edited','edited','stock','stock','edited','edited','stock'])
            for arm in ('stock','edited'):
                self.assertEqual(len(result['reports'][arm]['samples']),40)
                self.assertEqual({r['block'] for r in result['reports'][arm]['samples']},{0,1,2,3})
            self.assertEqual(servers.return_value.unload.call_count,8)

    def test_ports_and_store_must_be_isolated(self):
        with tempfile.TemporaryDirectory() as folder:
            original=Path(folder)/'original';isolated=Path(folder)/'isolated'
            self.assertTrue(validate_isolation('http://127.0.0.1:11434','http://127.0.0.1:11435',isolated,original)['stores_separate'])
            for url,store in [('http://localhost:11434',isolated),('http://remote:11435',isolated),('http://127.0.0.1:11435',original/'nested')]:
                with self.assertRaises(ValueError):validate_isolation('http://127.0.0.1:11434',url,store,original)

    def test_speed_rejects_incomplete_cold_generation(self):
        def measure(tag,*args,**kwargs):
            result=benchmark(tag)
            if tag=='stock':
                result['status']='incomplete';result['cold_samples'][0]['status']='truncated'
            return result
        with patch('mlx_abliterate.native_validation.Servers') as servers, \
             patch('mlx_abliterate.native_validation.run_benchmark',side_effect=measure):
            servers.return_value.original='original';servers.return_value.isolated='isolated'
            servers.return_value.isolation={}
            servers.return_value.digest.side_effect=lambda url,tag: tag+'-digest'
            result=run_speed('original','isolated','stock','edited',['p1','p2'])
            self.assertFalse(result['passed'])
            self.assertEqual(result['comparison']['status'],'inconclusive')

    def test_zero_requires_every_response_and_count(self):
        base=benchmark();edit=copy.deepcopy(base)
        self.assertTrue(compare_exact_responses(base,edit)['passed'])
        edit['samples'][3]['response_sha256']='c'*64
        self.assertFalse(compare_exact_responses(base,edit)['passed'])
        edit=copy.deepcopy(base);edit['samples'][0]['eval_count']=11
        self.assertFalse(compare_exact_responses(base,edit)['passed'])
        edit=copy.deepcopy(base);edit['cold_samples']=[]
        self.assertFalse(compare_exact_responses(base,edit)['passed'])

    def test_cross_runtime_speed_preserves_other_gates(self):
        base=benchmark();edit=benchmark('patched')
        self.assertEqual(compare_native_speed(base,edit)['status'],'pass')
        edit['settings']={'think':True}
        self.assertEqual(compare_native_speed(base,edit)['status'],'inconclusive')
        edit=benchmark('patched')
        for row in edit['samples']:row['decode_tokens_per_second']=44.
        self.assertEqual(compare_native_speed(base,edit)['status'],'fail')

    def test_import_requires_stable_manifest_and_explicit_store(self):
        with tempfile.TemporaryDirectory() as folder:
            models=Path(folder)/'isolated'
            with patch('mlx_abliterate.native_edit_export.verify_native_export',return_value={'base_weights_unchanged':True}), \
                 patch('mlx_abliterate.export.verify_import',return_value={'payloads_verified':True}) as imported, \
                 patch('mlx_abliterate.inventory.inspect_model',return_value={'manifest_digest':'sha256:stable'}):
                result=verify_native_import('fixture:27b',Path(folder)/'native',Path(folder)/'source',models)
                self.assertTrue(result['passed'])
                self.assertEqual(imported.call_args.args[2],models)
            with patch('mlx_abliterate.native_edit_export.verify_native_export',return_value={}), \
                 patch('mlx_abliterate.export.verify_import',return_value={}), \
                 patch('mlx_abliterate.inventory.inspect_model',side_effect=[{'manifest_digest':'before'},{'manifest_digest':'after'}]):
                with self.assertRaises(ValueError):
                    verify_native_import('fixture:27b',Path(folder)/'native',Path(folder)/'source',models)

    def test_quality_retention_not_aggregate(self):
        base=[{'id':'a','kind':'exact','status':'completed','task_check_passed':True},
              {'id':'b','kind':'exact','status':'completed','task_check_passed':False}]
        edit=copy.deepcopy(base)
        self.assertTrue(quality_gate(base,edit,{'a','b'})['passed'])
        edit[0]['task_check_passed']=False;edit[1]['task_check_passed']=True
        result=quality_gate(base,edit,{'a','b'})
        self.assertFalse(result['passed']);self.assertEqual(result['regressed_task_ids'],['a'])
        with self.assertRaises(ValueError):quality_gate(base,edit[:1],{'a','b'})
