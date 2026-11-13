import json
from pathlib import Path
import tempfile
import unittest
from mlx_abliterate.search import run_candidates


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'study'
        self.candidates = [{'id': 'zero', 'strength': 0}, {'id': 'edited', 'strength': .5}]
        self.provenance = {'model': 'sha256:base', 'dataset': 'sha256:data',
                           'runtime': {'mlx': 'fixture'}, 'edit_schema': 1}
        self.events = []

    def reset(self):
        self.events.append('reset')

    def evaluate(self, candidate):
        self.events.append(candidate['id'])
        return {'status': 'completed', 'score': 1 - candidate['strength'],
                'constraints_met': True, 'metrics': {'kl': candidate['strength']},
                'response': 'must never be serialized'}

    def run_study(self, **kwargs):
        return run_candidates(self.path, self.candidates, self.provenance,
                              evaluate=kwargs.pop('evaluate', self.evaluate),
                              reset_state=kwargs.pop('reset_state', self.reset), **kwargs)

    def test_order_resume_and_no_overwrite(self):
        result = self.run_study()
        self.assertEqual(result['best_candidate_id'], 'edited')
        self.assertEqual(self.events, ['reset', 'zero', 'reset', 'reset', 'edited', 'reset'])
        files = {path: path.read_bytes() for path in (self.path / 'trials').glob('*.json')}
        self.events.clear()
        self.assertEqual(self.run_study()['best_candidate_id'], 'edited')
        self.assertEqual(self.events, [])
        self.assertEqual(files, {path: path.read_bytes() for path in files})
        self.assertNotIn('must never', json.dumps(result))

    def test_resume_provenance_mismatch(self):
        self.run_study()
        self.provenance['dataset'] = 'different'
        with self.assertRaisesRegex(ValueError, 'resume refused'):
            self.run_study()

    def test_errors_and_constraints_cannot_win(self):
        def evaluate(candidate):
            if candidate['id'] == 'zero':
                return {'status': 'completed', 'score': -100, 'constraints_met': True, 'failures': 1}
            raise RuntimeError('private error details')
        result = self.run_study(evaluate=evaluate)
        self.assertEqual(result['eligible_candidates'], 0)
        self.assertIsNone(result['best_candidate_id'])
        self.assertIsNone(result['trials'][0]['result']['score'])
        self.assertNotIn('private error', json.dumps(result))
        self.assertEqual(self.run_study(evaluate=evaluate)['eligible_candidates'], 0)

    def test_cancelled_attempt_retried_in_new_file(self):
        def interrupt(candidate):
            raise KeyboardInterrupt()
        first = self.run_study(evaluate=interrupt)
        self.assertEqual(first['status'], 'cancelled')
        self.assertEqual(self.events, ['reset', 'reset'])
        original = (self.path / 'trials' / '000000-000001.json').read_bytes()
        second = self.run_study()
        self.assertEqual(second['status'], 'complete')
        self.assertEqual((self.path / 'trials' / '000000-000001.json').read_bytes(), original)
        self.assertTrue((self.path / 'trials' / '000000-000002.json').exists())

    def test_cancel_before_attempt_then_resume(self):
        self.assertEqual(self.run_study(cancelled=lambda: True)['status'], 'cancelled')
        self.assertEqual(self.events, [])
        self.assertEqual(self.run_study()['status'], 'complete')

    def test_reset_failure_blocks_remaining_candidates(self):
        calls = []
        def broken_reset():
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('cache cannot clear')
        result = self.run_study(reset_state=broken_reset)
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['evaluated_candidates'], 1)
        self.assertEqual(result['eligible_candidates'], 0)
        with self.assertRaisesRegex(RuntimeError, 'cache-reset'):
            self.run_study()

    def test_nan_or_missing_constraints_is_invalid(self):
        for output in [{'status': 'completed', 'score': float('nan'), 'constraints_met': True},
                       {'status': 'completed', 'score': 0}]:
            with tempfile.TemporaryDirectory() as directory:
                report = run_candidates(directory, self.candidates, self.provenance,
                                        evaluate=lambda _: output, reset_state=lambda: None)
                self.assertEqual(report['eligible_candidates'], 0)
                self.assertEqual(report['trials'][0]['result']['status'], 'runtime_error')

    def test_interrupt_during_initial_reset_blocks_resume(self):
        def interrupted_reset():
            raise KeyboardInterrupt()
        report = self.run_study(reset_state=interrupted_reset)
        self.assertEqual(report['status'], 'blocked')
        self.assertEqual(report['eligible_candidates'], 0)
        self.assertEqual(self.events, [])
        self.assertTrue(report['trials'][0]['reset_failed'])
        with self.assertRaisesRegex(RuntimeError, 'cache-reset'):
            self.run_study()

    def test_concurrent_owner_rejected(self):
        def nested(candidate):
            with self.assertRaisesRegex(RuntimeError, 'another worker'):
                self.run_study()
            return self.evaluate(candidate)
        self.assertEqual(self.run_study(evaluate=nested)['status'], 'complete')

    def test_corrupted_record_provenance_rejected(self):
        self.run_study()
        path = self.path / 'trials' / '000000-000001.json'
        record = json.loads(path.read_text())
        record['study_sha256'] = 'wrong'
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'trial provenance mismatch'):
            self.run_study()
