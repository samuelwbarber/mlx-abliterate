import unittest
from mlx_abliterate.collect import OnlineLastTokenMean
from mlx_abliterate.edits import mx


@unittest.skipIf(mx is None, 'optional MLX dependency unavailable')
class CollectTests(unittest.TestCase):
    def test_right_padding_and_streaming_mean(self):
        stats = OnlineLastTokenMean(2)
        hidden = mx.array([[[1., 2.], [3., 4.], [999., 999.]],
                           [[5., 6.], [999., 999.], [999., 999.]]])
        stats.update(hidden, lengths=[2, 1])
        self.assertEqual(stats.mean().tolist(), [4, 5])
        stats.update(mx.array([[[10., 11.]]]))
        self.assertEqual(stats.mean().tolist(), [6, 7])
        self.assertEqual(stats.count, 3)
        self.assertEqual(stats.accumulator_bytes, 8)
        self.assertEqual(set(vars(stats)), {'width', 'count', '_sum'})

    def test_positions_dtype_and_reset(self):
        stats = OnlineLastTokenMean(1)
        stats.update(mx.array([[[999.], [3.], [4.]]], dtype=mx.bfloat16), positions=[1])
        self.assertEqual(stats.mean().dtype, mx.float32)
        self.assertEqual(stats.mean().tolist(), [3])
        stats.reset()
        self.assertEqual(stats.count, 0)
        with self.assertRaises(ValueError):
            stats.mean()

    def test_reject_invalid_without_changing_state(self):
        stats = OnlineLastTokenMean(1)
        hidden = mx.array([[[2.], [3.]]])
        for kwargs in [{'lengths': [0]}, {'positions': [2]}, {'lengths': [True]},
                       {'positions': [1], 'lengths': [2]}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                stats.update(hidden, **kwargs)
        with self.assertRaises(ValueError):
            stats.update(mx.array([[[float('nan')]]]))
        self.assertEqual(stats.count, 0)
