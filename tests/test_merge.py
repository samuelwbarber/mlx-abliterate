import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

try:
    import mlx.core as mx
except ModuleNotFoundError:
    mx = None

from mlx_abliterate.export import ExportError, export_model, verify_export
from mlx_abliterate.merge import merge_export, merge_nvfp4, NVFP4_MAX


@unittest.skipIf(mx is None, 'MLX unavailable')
class MergeMathTests(unittest.TestCase):
    def setUp(self):
        self.weight = mx.sin(mx.arange(16 * 32, dtype=mx.float32)).reshape(16, 32)
        self.global_factor = mx.array(1 / NVFP4_MAX, dtype=mx.float32)
        self.packed, self.scales = mx.quantize(self.weight, mode='nvfp4', group_size=16, bits=4, global_scale=self.global_factor * NVFP4_MAX)
        self.direction = mx.cos(mx.arange(16, dtype=mx.float32))
        mx.eval(self.packed, self.scales)

    def decode(self, q, s, g):
        return mx.dequantize(q, s, mode='nvfp4', group_size=16, bits=4, dtype=mx.float32) * g

    def test_zero_is_exact_handle_identity(self):
        q, s, g, report = merge_nvfp4(self.packed, self.scales, self.global_factor, None, 0)
        self.assertIs(q, self.packed)
        self.assertIs(s, self.scales)
        self.assertIs(g, self.global_factor)
        self.assertFalse(report['requantized'])

    def test_explicit_weight_reference_and_dynamic_output(self):
        from mlx_abliterate.edits import normalize_direction, directional_output
        decoded = self.decode(self.packed, self.scales, self.global_factor)
        direction = normalize_direction(self.direction)
        reference = decoded - .7 * direction[:, None] * (direction @ decoded)[None, :]
        q, s, g, report = merge_nvfp4(self.packed, self.scales, self.global_factor, self.direction, .7)
        wanted_g = mx.max(mx.abs(reference)) / NVFP4_MAX
        wanted_q, wanted_s = mx.quantize(reference, mode='nvfp4', group_size=16, bits=4, global_scale=wanted_g * NVFP4_MAX)
        self.assertTrue(mx.array_equal(q, wanted_q).item())
        self.assertTrue(mx.array_equal(s, wanted_s).item())
        self.assertAlmostEqual(g.item(), wanted_g.item(), places=10)
        inputs = mx.sin(mx.arange(96, dtype=mx.float32)).reshape(3, 32)
        bias = mx.arange(16, dtype=mx.float32) / 100
        dynamic = directional_output(mx.matmul(inputs, decoded.T, stream=mx.cpu) + bias, direction, .7, bias)
        exact = mx.matmul(inputs, reference.T, stream=mx.cpu) + bias
        self.assertLess(mx.max(mx.abs(dynamic - exact)).item(), 2e-6)
        merged = inputs @ self.decode(q, s, g).T + bias
        relative_rms = mx.sqrt(mx.mean((merged - dynamic) ** 2)) / mx.sqrt(mx.mean(dynamic ** 2))
        self.assertLess(relative_rms.item(), .2)
        self.assertFalse(report['quality_validated'])

    def test_forced_quantization_control_and_native_global_semantics(self):
        q, s, g, report = merge_nvfp4(self.packed, self.scales, self.global_factor, None, 0, force_requantize=True, scale_policy='preserve')
        self.assertTrue(report['requantized'])
        self.assertTrue(report['quantization_only'])
        decoded = self.decode(q, s, g)
        correct = mx.dequantize(q, s, mode='nvfp4', group_size=16, bits=4, global_scale=g * NVFP4_MAX, dtype=mx.float32)
        wrong = mx.dequantize(q, s, mode='nvfp4', group_size=16, bits=4, global_scale=g, dtype=mx.float32)
        self.assertLess(mx.max(mx.abs(decoded - correct)).item(), 1e-6)
        self.assertGreater(mx.max(mx.abs(decoded - wrong)).item(), .5)

    def test_budget_and_bad_scales_fail(self):
        with self.assertRaisesRegex(ExportError, 'scratch'):
            merge_nvfp4(self.packed, self.scales, self.global_factor, self.direction, .5, scratch_limit_bytes=1)
        with self.assertRaisesRegex(ExportError, 'positive'):
            merge_nvfp4(self.packed, self.scales, mx.array(0., mx.float32), self.direction, .5)
        with self.assertRaisesRegex(ExportError, 'block-scale'):
            merge_nvfp4(self.packed, mx.full(self.scales.shape, 127, mx.uint8), self.global_factor, self.direction, .5)

    def test_invalid_budgets_cannot_bypass_admission(self):
        for budget in (float('nan'), float('inf'), True, 0, -1, 1024.5):
            with self.subTest(budget=budget), self.assertRaisesRegex(ExportError, 'scratch budget'):
                merge_nvfp4(self.packed, self.scales, self.global_factor, self.direction, .5,
                            scratch_limit_bytes=budget)


class MergeExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'models'
        (self.root / 'blobs').mkdir(parents=True)
        self.prefix = 'model.language_model.layers.0.mlp.down_proj'
        name = self.prefix + '.weight'
        header = {'__metadata__': {'quant_type': 'nvfp4'},
                  name: {'dtype': 'U32', 'shape': [2, 2], 'data_offsets': [0, 16]},
                  name + '.scale': {'dtype': 'U8', 'shape': [2, 1], 'data_offsets': [16, 18]},
                  name + '.global_scale': {'dtype': 'F32', 'shape': [], 'data_offsets': [18, 22]}}
        encoded = json.dumps(header).encode()
        payload = bytes([0x12, 0x34, 0x56, 0x76] * 4) + bytes([56, 56]) + struct.pack('<f', .125)
        tensor = self.blob(struct.pack('<Q', len(encoded)) + encoded + payload, 'application/vnd.ollama.image.tensor', name)
        config = self.blob(b'{"model_type":"fixture"}', 'application/vnd.ollama.image.json', 'config.json')
        modelconfig = self.blob(b'{"model_format":"safetensors"}', 'application/vnd.docker.container.image.v1+json')
        norm_header = json.dumps({'model.norm.weight': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}).encode()
        norm = self.blob(struct.pack('<Q', len(norm_header)) + norm_header + struct.pack('<ff', 1., 2.), 'application/vnd.ollama.image.tensor', 'model.norm.weight')
        manifest = {'schemaVersion': 2, 'config': modelconfig, 'layers': [tensor, norm, config]}
        path = self.root / 'manifests/registry.ollama.ai/library/tiny/latest'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(manifest))
        self.source, self.dest = self.base / 'source', self.base / 'merged'
        export_model('tiny', self.source, self.root)
        receipt = json.loads((self.source / '.mlx-abliterate/export-report.json').read_text())
        self.spec = {'schema_version': 1, 'source_manifest_digest': receipt['source_manifest_digest'], 'scale_policy': 'absmax', 'projections': {self.prefix: {'strength': 0}}}

    def blob(self, data, media, name=None):
        digest = hashlib.sha256(data).hexdigest()
        (self.root / 'blobs' / ('sha256-' + digest)).write_bytes(data)
        descriptor = {'digest': 'sha256:' + digest, 'size': len(data), 'mediaType': media}
        if name:
            descriptor['name'] = name
        return descriptor

    def test_zero_merge_copies_exact_payloads_without_mlx(self):
        result = merge_export(self.source, self.dest, self.spec)
        self.assertEqual(result['source_content_fingerprint'], verify_export(self.source)['content_fingerprint'])
        self.assertEqual(result['edited_projection_count'], 0)
        before = json.loads((self.source / '.mlx-abliterate/export-report.json').read_text())
        after = json.loads((self.dest / '.mlx-abliterate/export-report.json').read_text())
        self.assertEqual({x['name']: x['payload_sha256'] for x in before['tensors']}, {x['name']: x['payload_sha256'] for x in after['tensors']})
        self.assertFalse((self.dest / '.merge-scratch').exists())

    def test_fingerprint_mismatch_fails(self):
        self.spec['source_manifest_digest'] = 'sha256:' + '0' * 64
        with self.assertRaisesRegex(ExportError, 'fingerprint'):
            merge_export(self.source, self.dest, self.spec)
        self.assertFalse(self.dest.exists())

    def test_ineligible_projection_fails(self):
        self.spec['projections'] = {'mtp.layers.0.mlp.down_proj': {'strength': 0}}
        with self.assertRaisesRegex(ExportError, 'language-tower'):
            merge_export(self.source, self.dest, self.spec)

    def test_corrupt_source_fails_and_cleans_staging(self):
        shard = next(self.source.glob('*.safetensors'))
        data = bytearray(shard.read_bytes())
        data[-1] ^= 1
        shard.write_bytes(data)
        with self.assertRaisesRegex(ExportError, 'hash mismatch'):
            merge_export(self.source, self.dest, self.spec)
        self.assertFalse(self.dest.exists())
        self.assertEqual(list(self.base.glob('.merged-*')), [])

    def test_appended_asset_bytes_are_rejected_before_staging(self):
        with (self.source / 'config.json').open('ab') as stream:
            stream.write(b'\n{}')
        with self.assertRaisesRegex(ExportError, 'asset size'):
            merge_export(self.source, self.dest, self.spec)
        self.assertFalse(self.dest.exists())
        self.assertEqual(list(self.base.glob('.merged-*')), [])

    def test_undeclared_loader_asset_is_rejected(self):
        (self.source / 'tokenizer_config.json').write_text('{}')
        with self.assertRaisesRegex(ExportError, 'undeclared'):
            merge_export(self.source, self.dest, self.spec)
        self.assertFalse(self.dest.exists())

    def test_runtime_metadata_cannot_escape_export(self):
        options = self.source / '.mlx-abliterate/create-options.json'
        outside = self.base / 'options.json'
        options.rename(outside)
        options.symlink_to(outside)
        with self.assertRaisesRegex(ExportError, 'escaping'):
            merge_export(self.source, self.dest, self.spec)
        self.assertFalse(self.dest.exists())

    def test_zero_export_rejects_invalid_budget(self):
        with self.assertRaisesRegex(ExportError, 'scratch budget'):
            merge_export(self.source, self.dest, self.spec, scratch_limit_bytes=float('nan'))

    @unittest.skipIf(mx is None, 'MLX unavailable')
    def test_selected_merge_rewrites_only_requested_bundle(self):
        self.spec['projections'][self.prefix] = {'direction': [1., -1.], 'strength': .5}
        result = merge_export(self.source, self.dest, self.spec)
        self.assertEqual(result['edited_projection_count'], 1)
        self.assertFalse(result['quality_validated'])
        before = json.loads((self.source / '.mlx-abliterate/export-report.json').read_text())
        after = json.loads((self.dest / '.mlx-abliterate/export-report.json').read_text())
        before_hashes = {x['name']: x['payload_sha256'] for x in before['tensors']}
        after_hashes = {x['name']: x['payload_sha256'] for x in after['tensors']}
        self.assertEqual(before_hashes['model.norm.weight'], after_hashes['model.norm.weight'])
        self.assertNotEqual(before_hashes, after_hashes)
        self.assertEqual((self.source / 'config.json').read_bytes(), (self.dest / 'config.json').read_bytes())
        self.assertFalse((self.dest / '.merge-scratch').exists())


if __name__ == '__main__':
    unittest.main()
