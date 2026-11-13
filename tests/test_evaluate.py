import json
import math
from pathlib import Path
import tempfile
import unittest

from mlx_abliterate.evaluate import load_dataset_splits, first_token_kl
from mlx_abliterate.edits import mx


class DatasetTests(unittest.TestCase):
    def rows(self):
        return [
            {'id': 'a', 'prompt': 'first', 'split': 'direction', 'group': 'a'},
            {'id': 'b', 'prompt': 'second', 'split': 'direction', 'group': 'b'},
            {'id': 'c', 'prompt': 'third', 'split': 'search'},
            {'id': 'd', 'prompt': 'fourth', 'split': 'final'},
            {'id': 'e', 'prompt': 'fifth', 'split': 'capability'},
        ]

    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'examples.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            return load_dataset_splits(path)

    def test_fingerprint_and_content_free_lock(self):
        dataset = self.load(self.rows())
        self.assertEqual(dataset.fingerprint, self.load(self.rows()).fingerprint)
        self.assertNotIn('prompt', dataset.lock()['records'][0])
        rows = self.rows()
        rows[0]['prompt'] += ' changed'
        self.assertNotEqual(dataset.fingerprint, self.load(rows).fingerprint)
        self.assertEqual(dataset.splits['direction'][0]['prompt'], 'first')
        with self.assertRaises(TypeError):
            dataset.splits['direction'][0]['prompt'] = 'mutated'

    def test_duplicate_normalized_cross_split_rejected(self):
        rows = self.rows()
        rows[0]['prompt'] = 'the  same\ntext'
        rows[3]['prompt'] = 'the same text'
        with self.assertRaisesRegex(ValueError, 'duplicate normalized'):
            self.load(rows)

    def test_missing_split_groups_and_duplicate_ids(self):
        for change in ('split', 'groups', 'id'):
            rows = self.rows()
            if change == 'split':
                rows.pop()
            elif change == 'groups':
                rows[1]['group'] = 'a'
            else:
                rows[1]['id'] = rows[0]['id']
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(rows)


@unittest.skipIf(mx is None, 'optional MLX unavailable')
class KLTests(unittest.TestCase):
    def test_known_distribution_and_direction(self):
        p, q = [.8, .2], [.5, .5]
        expected = sum(a * math.log(a / b) for a, b in zip(p, q))
        value = first_token_kl([math.log(a) for a in p], [math.log(b) for b in q])
        self.assertAlmostEqual(value.item(), expected, places=6)
        reverse = first_token_kl([math.log(b) for b in q], [math.log(a) for a in p])
        self.assertNotAlmostEqual(value.item(), reverse.item(), places=3)

    def test_large_equal_logits_remain_normalized(self):
        expected = .5 * math.log(.5 / .8) + .5 * math.log(.5 / .2)
        actual = first_token_kl([1e30, 1e30], [math.log(.8), math.log(.2)])
        self.assertAlmostEqual(actual.item(), expected, places=6)

    def test_batch_offset_and_invalid(self):
        values = mx.array([[100., 99.], [-1., -2.]])
        self.assertTrue(mx.allclose(first_token_kl(values, values + 10), mx.zeros((2,)), atol=1e-5).item())
        for other in ([[1, 2, 3]], [[float('nan'), 1], [1, 2]]):
            with self.subTest(other=other), self.assertRaises(ValueError):
                first_token_kl(values, other)
