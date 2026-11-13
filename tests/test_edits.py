import unittest
from mlx_abliterate.edits import DirectionalProjection, directional_output, normalize_direction, mx, nn


@unittest.skipIf(mx is None, 'optional MLX dependency unavailable')
class EditTests(unittest.TestCase):
    def test_explicit_weight_edit_with_ordinary_bias(self):
        # Explicit CPU reference avoids M5 reduced-precision matmul rounding.
        with mx.stream(mx.cpu):
            mx.random.seed(10)
            base = nn.Linear(32, 16, bias=True)
            x = mx.random.normal((2, 4, 32))
            r = normalize_direction(mx.random.normal((16,)))
            edited = DirectionalProjection(base, r, .8)
            expected_weight = base.weight - .8 * r[:, None] * (r @ base.weight)[None, :]
            expected = x @ expected_weight.T + base.bias
            self.assertTrue(mx.allclose(edited(x), expected, atol=2e-6, rtol=2e-5).item())

    def test_quantized_offsets_not_ordinary_bias(self):
        # Explicit CPU reference avoids M5 reduced-precision matmul rounding.
        with mx.stream(mx.cpu):
            mx.random.seed(5)
            base = nn.QuantizedLinear.from_linear(nn.Linear(64, 16, bias=True), group_size=32, bits=4)
            original = base.weight.tolist()
            x = mx.random.normal((2, 64))
            r = normalize_direction(mx.random.normal((16,)))
            edited = DirectionalProjection(base, r, .6)
            decoded = mx.dequantize(base.weight, base.scales, base.biases, group_size=32, bits=4)
            explicit = decoded - .6 * r[:, None] * (r @ decoded)[None, :]
            expected = x @ explicit.T + base.bias
            self.assertTrue(mx.allclose(edited(x), expected, atol=3e-6, rtol=3e-5).item())
            for strength in [0, .5, 1, 0]:
                edited.set_strength(strength)
                mx.eval(edited(x))
            self.assertEqual(original, base.weight.tolist())
            self.assertIs(edited.base, base)
            self.assertTrue(mx.array_equal(edited(x), base(x)).item())

    def test_zero_bypass_and_bfloat16(self):
        y = mx.array([[1., 2., 3.]], dtype=mx.bfloat16)
        self.assertIs(directional_output(y, [1, 0, 0], 0), y)
        result = directional_output(y, [1, 0, 0], 1)
        self.assertEqual(result.dtype, mx.bfloat16)
        self.assertEqual(result.tolist(), [[0, 2, 3]])

    def test_validation(self):
        for direction in [[0, 0], [float('nan'), 1], [[1, 2]], []]:
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                normalize_direction(direction)
        for strength in [-1, 3, float('inf'), True]:
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                directional_output(mx.array([1., 2.]), [1, 0], strength)
        with self.assertRaises(ValueError):
            directional_output(mx.array([1., 2.]), [1, 0, 0], 1)
        normalized = normalize_direction([1e30, 1e30])
        self.assertAlmostEqual(mx.sum(normalized * normalized).item(), 1, places=6)

    def test_gpu_against_measured_unedited_rounding_control(self):
        mx.random.seed(10)
        base = nn.Linear(32, 16)
        x = mx.random.normal((2, 4, 32))
        r = normalize_direction(mx.random.normal((16,)))
        mx.eval(base.parameters(), x, r)
        with mx.stream(mx.cpu):
            exact_base = x @ base.weight.T + base.bias
            edited_weight = base.weight - .8 * r[:, None] * (r @ base.weight)[None, :]
            exact_edited = x @ edited_weight.T + base.bias
            mx.eval(exact_base, edited_weight, exact_edited)
        with mx.stream(mx.gpu):
            observed_base = base(x)
            observed_repacked = x @ edited_weight.T + base.bias
            observed_dynamic = DirectionalProjection(base, r, .8)(x)
            mx.eval(observed_base, observed_repacked, observed_dynamic)
        # A projection with strength .8 has operator norm <=1. GPU base-kernel
        # rounding can therefore propagate, but should not be amplified greatly.
        control = mx.sqrt(mx.sum(mx.square(observed_base - exact_base))).item()
        actual = mx.sqrt(mx.sum(mx.square(observed_dynamic - exact_edited))).item()
        self.assertLessEqual(actual, 2 * control + 2e-6)
        self.assertTrue(mx.all(mx.isfinite(observed_repacked)).item())
