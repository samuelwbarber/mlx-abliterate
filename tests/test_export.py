import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from mlx_abliterate.export import ExportError, export_model, plan_export, verify_import, verify_export, _publish_directory
from mlx_abliterate.inventory import read_safetensors_header


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'models'
        (self.root / 'blobs').mkdir(parents=True)
        self.layers = []
        header = {'__metadata__': {'quant_type': 'nvfp4'},
                  'model.weight': {'dtype': 'U32', 'shape': [2, 2], 'data_offsets': [0, 16]},
                  'model.weight.scale': {'dtype': 'U8', 'shape': [2, 1], 'data_offsets': [16, 18]},
                  'model.weight.global_scale': {'dtype': 'F32', 'shape': [], 'data_offsets': [18, 22]}}
        encoded = json.dumps(header).encode()
        self.tensor_payload = bytes(range(18)) + struct.pack('<f', 0.75)
        self.tensor = self.blob(struct.pack('<Q', len(encoded)) + encoded + self.tensor_payload,
                                'application/vnd.ollama.image.tensor', 'model.weight')
        self.layers.append(self.tensor)
        self.layers.append(self.blob(b'{"model_type":"fixture"}', 'application/vnd.ollama.image.json', 'config.json'))
        self.layers.append(self.blob(b'{"tokenizer_class":"fixture"}', 'application/vnd.ollama.image.json', 'tokenizer_config.json'))
        self.layers.append(self.blob(b'{"stop":["end"],"temperature":0}', 'application/vnd.ollama.image.params'))
        config = self.blob(b'{"renderer":"qwen3.8","parser":"qwen3.5","capabilities":["completion"]}', 'application/vnd.docker.container.image.v1+json')
        manifest = {'schemaVersion': 2, 'config': config, 'layers': self.layers}
        path = self.root / 'manifests/registry.ollama.ai/library/fixture/latest'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(manifest))
        self.output = self.base / 'exported'

    def blob(self, data, media, name=None):
        digest = hashlib.sha256(data).hexdigest()
        (self.root / 'blobs' / ('sha256-' + digest)).write_bytes(data)
        descriptor = {'digest': 'sha256:' + digest, 'size': len(data), 'mediaType': media}
        if name:
            descriptor['name'] = name
        return descriptor

    def test_plan_does_not_write_and_export_preserves_payload(self):
        plan = plan_export('fixture', self.root)
        self.assertEqual(plan['tensor_count'], 3)
        self.assertFalse(self.output.exists())
        export_model('fixture', self.output, self.root)
        shard = next(self.output.glob('*.safetensors'))
        header = read_safetensors_header(shard)
        self.assertEqual(set(header), {'model.weight', 'model.weight_scale', 'model.weight_scale_2'})
        with shard.open('rb') as source:
            size = struct.unpack('<Q', source.read(8))[0]
            source.seek(8 + size)
            payload = source.read()
        self.assertEqual(payload, self.tensor_payload)
        result = verify_import('fixture', self.output, self.root)
        self.assertTrue(result['payloads_verified'])
        self.assertFalse((self.output / 'export-report.json').exists())
        self.assertFalse((self.output / 'create-options.json').exists())
        self.assertTrue((self.output / '.mlx-abliterate/export-report.json').is_file())
        self.assertIn('PARSER """qwen3.5"""', (self.output / 'Modelfile').read_text())

    def test_exclusive_publish_rejects_concurrent_empty_destination(self):
        staging = self.base / 'staging'
        staging.mkdir()
        (staging / 'payload').write_text('keep')
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            _publish_directory(staging, self.output)
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertTrue((staging / 'payload').is_file())

    def test_destination_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_text('existing')
        with self.assertRaises(FileExistsError):
            export_model('fixture', self.output, self.root)
        self.assertEqual(marker.read_text(), 'existing')

    def test_failed_copy_removes_staging(self):
        with patch('mlx_abliterate.export._copy_range', side_effect=ExportError('truncated')):
            with self.assertRaises(ExportError):
                export_model('fixture', self.output, self.root)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.base.glob('.exported-*')), [])

    def test_changed_payload_fails_verification(self):
        export_model('fixture', self.output, self.root)
        path = self.root / 'blobs' / self.tensor['digest'].replace(':', '-')
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        with self.assertRaisesRegex(ExportError, 'payload differs'):
            verify_import('fixture', self.output, self.root)

    def test_multiple_shards_and_exact_boundaries(self):
        with patch('mlx_abliterate.export.SHARD_BYTES', 16):
            export_model('fixture', self.output, self.root)
        self.assertEqual(len(list(self.output.glob('*.safetensors'))), 2)
        index = json.loads((self.output / 'model.safetensors.index.json').read_text())
        self.assertEqual(len(index['weight_map']), 3)

    def test_local_export_verification_and_content_identity(self):
        export_model('fixture', self.output, self.root)
        first = verify_export(self.output, require_unmodified=True)
        self.assertTrue(first['payloads_verified'])
        self.assertEqual(first['tensor_count'], 3)
        self.assertEqual(first['asset_count'], 2)
        self.assertEqual(first, verify_export(self.output, require_unmodified=True))
        # Runtime options and Modelfile changes retain ancestral model identity,
        # but must change the actual experiment fingerprint.
        modelfile = self.output / 'Modelfile'
        modelfile.write_text(modelfile.read_text() + '# changed runtime configuration\n')
        second = verify_export(self.output)
        self.assertEqual(first['source_manifest_digest'], second['source_manifest_digest'])
        self.assertNotEqual(first['content_fingerprint'], second['content_fingerprint'])
        options = self.output / '.mlx-abliterate/create-options.json'
        options.write_text(options.read_text() + ' ')
        self.assertNotEqual(second['content_fingerprint'], verify_export(self.output)['content_fingerprint'])

    def test_corrupted_local_tensor_rejected(self):
        export_model('fixture', self.output, self.root)
        shard = next(self.output.glob('*.safetensors'))
        with shard.open('r+b') as target:
            target.seek(-1, 2)
            old = target.read(1)
            target.seek(-1, 2)
            target.write(bytes([old[0] ^ 1]))
        with self.assertRaisesRegex(ExportError, 'payload hash'):
            verify_export(self.output)

    def test_corrupted_tokenizer_rejected(self):
        export_model('fixture', self.output, self.root)
        path = self.output / 'tokenizer_config.json'
        path.write_bytes(path.read_bytes().replace(b'fixture', b'changed'))
        with self.assertRaisesRegex(ExportError, 'asset hash'):
            verify_export(self.output)

    def test_modified_checkpoint_rejected_for_collection(self):
        export_model('fixture', self.output, self.root)
        path = self.output / '.mlx-abliterate/export-report.json'
        receipt = json.loads(path.read_text())
        receipt['modified'] = True
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ExportError, 'unmodified'):
            verify_export(self.output, require_unmodified=True)
        self.assertTrue(verify_export(self.output)['modified'])

    def test_undeclared_asset_rejected(self):
        export_model('fixture', self.output, self.root)
        (self.output / 'tokenizer.model').write_bytes(b'untracked tokenizer')
        with self.assertRaisesRegex(ExportError, 'undeclared'):
            verify_export(self.output)

    def test_changed_shape_and_index_disagreement_rejected(self):
        export_model('fixture', self.output, self.root)
        path = self.output / '.mlx-abliterate/export-report.json'
        receipt = json.loads(path.read_text())
        receipt['tensors'][0]['shape'] = [4, 1]
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ExportError, 'shape/dtype/size'):
            verify_export(self.output)

    def test_store_destination_rejected(self):
        with self.assertRaisesRegex(ExportError, 'outside'):
            export_model('fixture', self.root / 'new-export', self.root)


if __name__ == '__main__':
    unittest.main()
