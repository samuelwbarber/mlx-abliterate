import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from mlx_abliterate.mlx_adapter import (
    FidelityError, decode_e4m3, decode_nvfp4_group, preflight_export,
    probe_nvfp4_blob, QwenHybridAdapter,
)


class NumericReferenceTests(unittest.TestCase):
    def test_signed_fp4_codes_and_global_scale(self):
        values = decode_nvfp4_group(bytes.fromhex('1032547698badcfe'), 56, 2.)
        self.assertEqual(values, [0., 1., 2., 3., 4., 6., 8., 12., -0., -1., -2., -3., -4., -6., -8., -12.])

    def test_fp8_subnormal_and_maximum(self):
        self.assertEqual(decode_e4m3(1), 1 / 512)
        self.assertEqual(decode_e4m3(126), 448)
        self.assertEqual(decode_e4m3(184), -1.)
        with self.assertRaises(FidelityError):
            decode_e4m3(127)

    def test_invalid_factors_fail(self):
        for factor in (0, -1, math.inf, math.nan):
            with self.subTest(factor=factor), self.assertRaises(FidelityError):
                decode_nvfp4_group(bytes(8), 56, factor)
        with self.assertRaises(FidelityError):
            decode_nvfp4_group(bytes(8), 184, 1.)

    def test_unsupported_architecture_fails_before_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text(json.dumps({'model_type': 'other'}))
            with self.assertRaisesRegex(FidelityError, '64-layer'):
                preflight_export(path)

    def test_mismatched_attention_schedule_fails(self):
        config = {'model_type': 'qwen3_5', 'text_config': {
            'num_hidden_layers': 64, 'hidden_size': 5120,
            'layer_types': ['full_attention'] * 64, 'full_attention_interval': 4}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(FidelityError, 'attention schedule'):
                preflight_export(path)

    @unittest.skipUnless(importlib.util.find_spec('mlx') is not None, 'Optional MLX runtime not installed')
    def test_projection_reset_and_invalid_update_are_atomic(self):
        from types import SimpleNamespace
        import mlx.core as mx
        import mlx.nn as nn
        base = nn.Linear(2, 5120, bias=False)
        parent = SimpleNamespace(down_proj=base)
        adapter = object.__new__(QwenHybridAdapter)
        adapter.model = SimpleNamespace(language_model=SimpleNamespace(layers=[SimpleNamespace(mlp=parent)]))
        path = 'model.language_model.layers.0.mlp.down_proj'
        adapter.projections = {path: base}
        weight = base.weight
        adapter.set_edits({path: mx.ones((5120,))}, 0.)
        wrapped = parent.down_proj
        self.assertIs(wrapped.base.weight, weight)
        with self.assertRaises(FidelityError):
            adapter.set_edits({path: mx.ones((3,))}, 1.)
        self.assertIs(parent.down_proj, wrapped)
        adapter.reset_edits()
        self.assertIs(parent.down_proj, base)
        self.assertIs(base.weight, weight)

    @unittest.skipUnless(importlib.util.find_spec('mlx') is not None, 'Optional MLX runtime not installed')
    def test_collection_restores_layers_on_error(self):
        from types import SimpleNamespace
        import mlx.nn as nn
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.is_linear = True
                self.linear_attn = nn.Identity()
        layers = [Layer() for _ in range(64)]
        originals = list(layers)
        adapter = object.__new__(QwenHybridAdapter)
        adapter.model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
        adapter.memory_limit = 1024 ** 3
        def fail(tokens):
            raise RuntimeError('inference failed')
        adapter.last_logits = fail
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            adapter.collect_last_hidden([1, 2])
        self.assertTrue(all(a is b for a, b in zip(layers, originals)))

    @unittest.skipUnless(importlib.util.find_spec('mlx') is not None, 'Optional MLX runtime not installed')
    def test_tiny_mlx_projection_matches_scalar_decode(self):
        name = 'projection.weight'
        header = {'__metadata__': {'quant_type': 'nvfp4'},
                  name: dict(dtype='U32', shape=[1, 2], data_offsets=[0, 8]),
                  name + '.scale': dict(dtype='U8', shape=[1, 1], data_offsets=[8, 9]),
                  name + '.global_scale': dict(dtype='F32', shape=[], data_offsets=[9, 13])}
        encoded = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tensor'
            path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + bytes.fromhex('1032547698badcfe') + bytes([56]) + struct.pack('<f', .25))
            report = probe_nvfp4_blob(path, name)
            self.assertTrue(report['passed'])
            self.assertLess(report['projection_abs_error'], 1e-6)


if __name__ == '__main__':
    unittest.main()
