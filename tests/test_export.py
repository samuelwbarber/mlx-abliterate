import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from mlx_abliterate.export import ExportError, export_model, plan_export, verify_import
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

    def test_store_destination_rejected(self):
        with self.assertRaisesRegex(ExportError, 'outside'):
            export_model('fixture', self.root / 'new-export', self.root)


if __name__ == '__main__':
    unittest.main()
