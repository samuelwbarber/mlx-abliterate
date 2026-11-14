import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from mlx_abliterate.export import ExportError, verify_export, _publish_directory
from mlx_abliterate.native_edit_export import export_native_edits,verify_native_export,ARCHITECTURE


class NativeEditExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.source=self.base/'source';self.source.mkdir()
        self.dest=self.base/'native';meta=self.source/'.mlx-abliterate';meta.mkdir()
        self.paths=['model.language_model.layers.0.linear_attn.out_proj','model.language_model.layers.0.mlp.down_proj']
        header,records,payload={},[],bytearray()
        for path in self.paths:
            for suffix,dtype,shape,raw in [('.weight','U32',[5120,2],bytes(5120*8)),('.weight_scale','U8',[5120,1],bytes([56])*5120),('.weight_scale_2','F32',[],struct.pack('<f',.125))]:
                name=path+suffix;offset=len(payload);payload.extend(raw)
                header[name]={'dtype':dtype,'shape':shape,'data_offsets':[offset,len(payload)]}
                records.append({'name':name,'original_name':name,'dtype':dtype,'shape':shape,'size':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'shard':'weights.safetensors'})
        encoded=json.dumps(header).encode()
        (self.source/'weights.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
        (self.source/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{r['name']:'weights.safetensors' for r in records}}))
        assets=[]
        for name,value in [('config.json',{'architectures':['Qwen3_5ForConditionalGeneration'],'model_type':'qwen3_5'}),('tokenizer_config.json',{'test':'unchanged'})]:
            raw=json.dumps(value).encode();(self.source/name).write_bytes(raw)
            assets.append({'name':name,'size':len(raw),'digest':'sha256:'+hashlib.sha256(raw).hexdigest()})
        (self.source/'Modelfile').write_text('FROM ./\n');(meta/'create-options.json').write_text('{}')
        receipt={'schema_version':1,'source_manifest_digest':'sha256:'+'a'*64,'tensors':records,'assets':assets,'modified':False}
        (meta/'export-report.json').write_text(json.dumps(receipt))
        self.identity=verify_export(self.source,require_unmodified=True)
        self.spec={'schema_version':1,'source_manifest_digest':receipt['source_manifest_digest'],
                   'source_content_fingerprint':self.identity['content_fingerprint'],'scale_policy':'preserve','force_requantize':False,
                   'projections':{p:{'strength':1.575,'direction':[1.]+[0.]*5119} for p in self.paths}}
        self.report={'status':'complete','export_gate_passed':True,'source_identity':self.identity,
                     'selected':{'id':'chosen','strength':1.575,'start':0,'stop':1,'direction_mode':'global','direction_layer':24},
                     'standard':{'status':'complete','best_candidate_id':'chosen','trials':[{'candidate_id':'chosen','result':{'eligible':True,'constraints_met':True,'status':'completed','failures':0,'metrics':{'refusals':3}}}]}}
        self.spec_path=self.base/'spec.json';self.report_path=self.base/'selection.json'
        self.save()
        patched=patch('mlx_abliterate.native_edit_export.preflight_export',return_value={'projection_paths':self.paths})
        patched.start();self.addCleanup(patched.stop)

    def save(self):
        self.spec_path.write_text(json.dumps(self.spec));self.report_path.write_text(json.dumps(self.report))

    def export(self,**kwargs):
        return export_native_edits(self.source,self.dest,self.spec_path,self.report_path,**kwargs)

    def test_exact_base_payloads_and_explicit_architecture(self):
        result=self.export()
        verified=verify_native_export(self.dest,self.source)
        self.assertEqual(result['base_tensor_count'],6)
        self.assertEqual(result['direction_tensor_count'],2)
        self.assertEqual(result['extra_payload_bytes'],40960)
        self.assertTrue(verified['source_identity_reverified'])
        self.assertEqual((self.source/'weights.safetensors').read_bytes(),(self.dest/'weights.safetensors').read_bytes())
        config=json.loads((self.dest/'config.json').read_text())
        self.assertEqual(config['architectures'],[ARCHITECTURE])
        self.assertEqual(config['mlx_abliterate']['targets'][0]['path'],'model.layers.0.linear_attn.out_proj')
        self.assertEqual(config['mlx_abliterate']['targets'][0]['strength'],1.575)
        self.assertEqual((self.source/'tokenizer_config.json').read_bytes(),(self.dest/'tokenizer_config.json').read_bytes())

    def test_zero_control_retains_vectors_and_zeroes_only_effective_strengths(self):
        self.export(zero_strength=True)
        result=verify_native_export(self.dest,self.source)
        self.assertTrue(result['zero_control'])
        config=json.loads((self.dest/'config.json').read_text())
        self.assertTrue(all(t['strength']==0. for t in config['mlx_abliterate']['targets']))
        saved=json.loads((self.dest/'.mlx-abliterate/edit-spec.json').read_text())
        self.assertEqual(saved['projections'][self.paths[0]]['strength'],1.575)

    def test_unsuccessful_selection_does_not_write(self):
        self.report['export_gate_passed']=False;self.save()
        with self.assertRaisesRegex(ExportError,'Successful selection'):self.export()
        self.assertFalse(self.dest.exists())

    def test_six_refusals_rejected_even_if_report_claims_success(self):
        self.report['standard']['trials'][0]['result']['metrics']['refusals']=6;self.save()
        with self.assertRaisesRegex(ExportError,'standard/capability'):self.export()

    def test_shape_finiteness_and_unit_norm(self):
        for vector in ([1.]*5120,[float('nan')]+[0.]*5119,[1.]+[0.]*5118):
            self.spec['projections'][self.paths[0]]['direction']=vector;self.save()
            with self.assertRaises(ExportError):self.export()
            self.assertFalse(self.dest.exists())

    def test_schedule_or_strength_cannot_change_after_selection(self):
        self.spec['projections'][self.paths[0]]['strength']=1.6;self.save()
        with self.assertRaisesRegex(ExportError,'schedule'):self.export()

    def test_existing_destination_is_untouched(self):
        self.dest.mkdir();(self.dest/'keep').write_text('keep')
        with self.assertRaises(FileExistsError):self.export()
        self.assertEqual((self.dest/'keep').read_text(),'keep')

    def test_publication_race_does_not_replace_destination(self):
        def race(staging,destination):
            destination.mkdir();_publish_directory(staging,destination)
        with patch('mlx_abliterate.native_edit_export._publish_directory',side_effect=race):
            with self.assertRaises(FileExistsError):self.export()
        self.assertEqual(list(self.dest.iterdir()),[])
        self.assertEqual(list(self.base.glob('.native-*')),[])

    def test_corrupt_base_or_direction_payload_rejected(self):
        self.export()
        shard=self.dest/'mlx-abliterate-directions.safetensors'
        raw=bytearray(shard.read_bytes());raw[-1]^=1;shard.write_bytes(raw)
        with self.assertRaisesRegex(ExportError,'hash mismatch'):verify_native_export(self.dest)

    def test_changed_architecture_detected_even_with_updated_asset_receipt(self):
        self.export();path=self.dest/'config.json';config=json.loads(path.read_text())
        config['architectures']=['Qwen3_5ForConditionalGeneration'];raw=json.dumps(config).encode();path.write_bytes(raw)
        receipt_path=self.dest/'.mlx-abliterate/export-report.json';receipt=json.loads(receipt_path.read_text())
        asset=next(a for a in receipt['assets'] if a['name']=='config.json');asset.update(size=len(raw),digest='sha256:'+hashlib.sha256(raw).hexdigest())
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ExportError,'explicit custom architecture'):verify_native_export(self.dest)

    def test_unrelated_config_change_is_rejected(self):
        self.export();path=self.dest/'config.json';config=json.loads(path.read_text())
        config['unrelated_change']=True;raw=json.dumps(config).encode();path.write_bytes(raw)
        receipt_path=self.dest/'.mlx-abliterate/export-report.json';receipt=json.loads(receipt_path.read_text())
        asset=next(a for a in receipt['assets'] if a['name']=='config.json');asset.update(size=len(raw),digest='sha256:'+hashlib.sha256(raw).hexdigest())
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ExportError,'unrelated base configuration'):verify_native_export(self.dest)
