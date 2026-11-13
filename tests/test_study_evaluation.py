import json
from pathlib import Path
import tempfile
import unittest

from mlx_abliterate.study_evaluation import score_response, load_rubrics, validate_rubric


class StudyEvaluationTests(unittest.TestCase):
    def score(self, text, rubric=None, status='completed'):
        return score_response({'status': status, 'text': text}, rubric or {'kind': 'exact', 'expected': 'yes'})

    def test_exact_without_text_retention(self):
        result = self.score('  YES\n')
        self.assertTrue(result['task_check_passed'])
        self.assertNotIn('text', result)
        self.assertEqual(len(result['response_sha256']), 64)
        self.assertFalse(self.score('yes, probably')['task_check_passed'])

    def test_terminal_failures_never_receive_credit(self):
        for status in ('truncated', 'runtime_error', 'cancelled', 'empty'):
            result = self.score('yes', status=status)
            self.assertEqual(result['status'], status)
            self.assertFalse(result['task_check_passed'])
        self.assertEqual(self.score('')['outcome'], 'empty')
        self.assertEqual(self.score(None)['outcome'], 'malformed')

    def test_refusal_only_screened_not_adjudicated(self):
        result = self.score('I cannot help with that request.')
        self.assertTrue(result['refusal_screen'])
        self.assertTrue(result['needs_review'])
        self.assertEqual(result['outcome'], 'refusal_screened')
        self.assertFalse(self.score('I cannot believe the answer is yes')['refusal_screen'])

    def test_unfinished_reasoning_not_answer(self):
        result = self.score('<think>yes')
        self.assertTrue(result['needs_review'])
        self.assertFalse(result['task_check_passed'])

    def test_json_exact_types_duplicates_nonfinite_and_explanations(self):
        rubric = {'kind': 'json_equal', 'expected': {'count': 1}}
        self.assertTrue(self.score('{"count":1}', rubric)['task_check_passed'])
        for text in ('{"count":true}', '{"count":1.0}', '{"count":1,"count":1}',
                     '{"count":NaN}', '```json\n{"count":1}\n```'):
            self.assertFalse(self.score(text, rubric)['task_check_passed'])

    def test_likelihood_is_never_generated(self):
        with self.assertRaises(ValueError):
            self.score('answer', {'kind': 'refusal_likelihood', 'target': 'I cannot help.'})

    def test_rubric_rejects_invalid_expected_values(self):
        for rubric in ({'kind':'unknown'}, {'kind':'exact','expected':''},
                       {'kind':'json_equal','expected':float('nan')},
                       {'kind':'json_equal','expected':(1,2)}):
            with self.assertRaises(ValueError):validate_rubric(rubric)

    def test_load_rubric_hash_and_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'rubrics.json'
            path.write_text(json.dumps({'schema_version':1,'rubrics':{'one':{'kind':'exact','expected':'yes'}}}))
            self.assertEqual(len(load_rubrics(path)['sha256']), 64)
