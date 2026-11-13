import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from mlx_abliterate.inventory import (
    InventoryError, blob_path, inspect_model, read_safetensors_header,
    resolve_model, tensor_layout,
)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'blobs').mkdir()

    def blob(self, data, media='application/vnd.ollama.image.json', name=None):
        digest = 'sha256:' + hashlib.sha256(data).hexdigest()
        (self.root / 'blobs' / digest.replace(':', '-')).write_bytes(data)
        descriptor = dict(digest=digest, size=len(data), mediaType=media)
        if name:
            descriptor['name'] = name
        return descriptor

    def manifest(self, layers, tag='tiny:latest'):
        config = self.blob(b'{"model_format":"safetensors"}')
        value = dict(schemaVersion=2, config=config, layers=layers)
        path = self.root / 'manifests/registry.ollama.ai/library' / tag.replace(':', '/')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return value

    def tensor(self, name='model.language_model.layers.0.mlp.down_proj.weight'):
        header = {'__metadata__': {'quant_type': 'nvfp4'},
                  name: dict(dtype='U32', shape=[2, 2], data_offsets=[0, 16]),
                  name + '.scale': dict(dtype='U8', shape=[2, 1], data_offsets=[16, 18]),
                  name + '.global_scale': dict(dtype='F32', shape=[], data_offsets=[18, 22])}
        data = json.dumps(header).encode()
        return self.blob(struct.pack('<Q', len(data)) + data + bytes(22),
                         'application/vnd.ollama.image.tensor', name)

    def test_inspect_nvfp4_and_nested_provenance(self):
        tensor = self.tensor()
        child = self.manifest([tensor])
        child_blob = self.blob(json.dumps(child).encode(), 'application/vnd.docker.distribution.manifest.v2+json')
        self.manifest([tensor, child_blob])
        result = inspect_model('tiny', self.root)
        self.assertEqual(result['summary']['manifest_count'], 2)
        self.assertEqual(result['summary']['active_tensor_count'], 1)
        self.assertEqual(result['summary']['editable_projection_count'], 1)
        self.assertEqual(result['layers'][0]['logical_shape'], [2, 16])
        self.assertFalse(result['summary']['tensor_payload_hashes_verified'])
        self.assertEqual(len(result['blobs']), 3)
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_v2_manifest_preferred_and_digest_checked(self):
        original = self.manifest([self.tensor()])
        newer = dict(original, layers=[])
        descriptor = self.blob(json.dumps(newer).encode(), 'application/vnd.docker.distribution.manifest.v2+json')
        link = self.root / 'manifests-v2/ollama.com/library/tiny/latest'
        link.parent.mkdir(parents=True)
        link.symlink_to(Path('../../../../blobs') / descriptor['digest'].replace(':', '-'))
        for name in ('tiny', 'ollama.com/library/tiny', 'registry.ollama.ai/library/tiny'):
            result = inspect_model(name, self.root)
            self.assertEqual(result['manifest_digest'], descriptor['digest'])
            self.assertEqual(result['summary']['active_tensor_count'], 0)
        blob_path(self.root, descriptor['digest']).write_text('{}')
        with self.assertRaisesRegex(InventoryError, 'hash mismatch'):
            inspect_model('tiny', self.root)

    def test_dangling_v2_manifest_falls_back_to_legacy(self):
        self.manifest([self.tensor()])
        link = self.root / 'manifests-v2/ollama.com/library/tiny/latest'
        link.parent.mkdir(parents=True)
        link.symlink_to(Path('../../../../blobs') / ('sha256-' + 'a' * 64))
        self.assertEqual(inspect_model('tiny', self.root)['summary']['active_tensor_count'], 1)

    def test_manifest_symlink_cannot_target_non_blob(self):
        self.manifest([self.tensor()])
        link = self.root / 'manifests-v2/ollama.com/library/tiny/latest'
        link.parent.mkdir(parents=True)
        link.symlink_to(self.root / 'manifests/registry.ollama.ai/library/tiny/latest')
        with self.assertRaisesRegex(InventoryError, 'blob directory'):
            inspect_model('tiny', self.root)

    def test_wrong_blob_size_rejected(self):
        tensor = self.tensor()
        tensor['size'] += 1
        self.manifest([tensor])
        with self.assertRaisesRegex(InventoryError, 'size mismatch'):
            inspect_model('tiny', self.root)

    def test_metadata_hash_mismatch_rejected(self):
        config = self.blob(b'{}', name='config.json')
        self.manifest([config])
        blob_path(self.root, config['digest']).write_bytes(b'[]')
        with self.assertRaisesRegex(InventoryError, 'hash mismatch'):
            inspect_model('tiny', self.root)

    def test_tensor_payload_not_read_or_hashed(self):
        tensor = self.tensor()
        self.manifest([tensor])
        path = blob_path(self.root, tensor['digest'])
        with path.open('r+b') as stream:
            stream.seek(-1, 2)
            stream.write(b'\x01')
        result = inspect_model('tiny', self.root)
        record = next(x for x in result['blobs'] if x['digest'] == tensor['digest'])
        self.assertFalse(record['content_hash_verified'])
        with self.assertRaisesRegex(InventoryError, 'payload content hash mismatch'):
            inspect_model('tiny', self.root, verify_payloads=True)

    def test_optional_full_payload_verification(self):
        self.manifest([self.tensor()])
        result = inspect_model('tiny', self.root, verify_payloads=True)
        self.assertTrue(result['summary']['tensor_payload_hashes_verified'])
        self.assertTrue(all(x['content_hash_verified'] for x in result['blobs']))

    def test_unknown_quantization_rejected(self):
        descriptor = self.tensor()
        header = read_safetensors_header(blob_path(self.root, descriptor['digest']))
        header['__metadata__']['quant_type'] = 'unknown'
        with self.assertRaisesRegex(InventoryError, 'Unsupported quantization'):
            tensor_layout(header, descriptor['name'])

    def test_wrong_scale_shape_rejected(self):
        descriptor = self.tensor()
        header = read_safetensors_header(blob_path(self.root, descriptor['digest']))
        header[descriptor['name'] + '.scale']['shape'] = [1, 2]
        with self.assertRaisesRegex(InventoryError, 'block-scale'):
            tensor_layout(header, descriptor['name'])

    def test_header_truncation_and_offset_overlap_rejected(self):
        path = self.root / 'bad'
        path.write_bytes(struct.pack('<Q', 1000))
        with self.assertRaises(InventoryError):
            read_safetensors_header(path)
        header = {'a': dict(dtype='F32', shape=[1], data_offsets=[0, 4]),
                  'b': dict(dtype='F32', shape=[1], data_offsets=[0, 4])}
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + bytes(4))
        with self.assertRaisesRegex(InventoryError, 'overlapping'):
            read_safetensors_header(path)

    def test_path_traversal_and_invalid_digest_rejected(self):
        for model in ('../tiny', '/tiny', 'tiny:', 'tiny:a:b', 'a/b/c/d', 'a\\b'):
            with self.subTest(model=model), self.assertRaises(InventoryError):
                resolve_model(model, self.root)
        with self.assertRaises(InventoryError):
            blob_path(self.root, 'sha256:../secret')

    def test_symlink_escape_rejected(self):
        outside = self.root.parent / (self.root.name + '-outside')
        outside.write_bytes(b'{}')
        self.addCleanup(outside.unlink)
        digest = 'sha256:' + 'a' * 64
        (self.root / 'blobs' / digest.replace(':', '-')).symlink_to(outside)
        with self.assertRaisesRegex(InventoryError, 'escapes'):
            blob_path(self.root, digest)

    def test_duplicate_tensor_names_rejected(self):
        tensor = self.tensor()
        self.manifest([tensor, tensor])
        with self.assertRaisesRegex(InventoryError, 'Duplicate'):
            inspect_model('tiny', self.root)


if __name__ == '__main__':
    unittest.main()
