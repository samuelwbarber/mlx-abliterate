import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import subprocess

from mlx_abliterate.stability import run_trials, validate_config, supervise

try:
    import mlx.core as mx
except ImportError:
    mx = None


class ConfigTests(unittest.TestCase):
    def test_invalid_admission(self):
        for lengths, cycles, tolerance in [([], 5, 0), ([1, 1], 5, 0), ([4097], 5, 0),
                                            ([True], 5, 0), ([1], 2, 0), ([1], 5, -1)]:
            with self.assertRaises(ValueError):
                validate_config(lengths, cycles, tolerance)
        with self.assertRaises(ValueError):
            supervise('unused', timeout_seconds=float('nan'))

    def test_watchdog_failures_return_reports(self):
        for failure in (OSError('unavailable'), subprocess.TimeoutExpired('ps', 2),
                        SimpleNamespace(returncode=0, stdout='invalid')):
            process = MagicMock(pid=123, exitcode=-15)
            process.is_alive.return_value = True
            context = MagicMock()
            context.Process.return_value = process
            kwargs = ({'side_effect': failure} if isinstance(failure, Exception)
                      else {'return_value': failure})
            with patch('mlx_abliterate.stability.multiprocessing.get_context', return_value=context), \
                    patch('mlx_abliterate.stability.subprocess.run', **kwargs):
                report = supervise('unused', lengths=[1], cycles=3)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error_type'], 'watchdog_unavailable')
            process.terminate.assert_called_once()

    def test_worker_failure_returns_failed_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report = supervise(Path(folder) / 'missing', lengths=[1], cycles=3, timeout_seconds=20)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('error_type', report)
        self.assertEqual(report['watchdog']['exitcode'], 0)


@unittest.skipIf(mx is None, 'MLX unavailable')
class StabilityTests(unittest.TestCase):
    class Adapter:
        def __init__(self, *, broken_zero=False, broken_restore=False, crash=False):
            self.preflight = {'config': {'text_config': {'num_hidden_layers': 2}}}
            self.projections = {'tiny': SimpleNamespace(weight=mx.ones((4, 4)))}
            self.strength = None
            self.broken_zero = broken_zero
            self.broken_restore = broken_restore
            self.crash = crash
            self.edited = False

        def collect_last_hidden(self, tokens):
            return {i: mx.ones((4,)) for i in range(3)}

        def set_edits(self, directions, strength):
            self.strength = strength
            self.edited = True

        def reset_edits(self):
            self.strength = None

        def last_logits(self, tokens):
            if self.crash and self.strength == .05:
                raise RuntimeError('injected')
            error = self.broken_zero if self.strength == 0 else 0
            if self.strength is None and self.edited and self.broken_restore:
                error += 1
            return mx.array([[1., 2.]]) + (self.strength or 0) + error

    class Guard:
        def __init__(self, leak=False):
            self.calls = 0
            self.leak = leak

        def check(self, mx):
            self.calls += 1
            return {'mlx_active_bytes': self.calls * 100 if self.leak else 100}

    def test_plateau_and_warmup(self):
        result = run_trials(self.Adapter(), mx, [[1], [1, 2]], cycles=3,
                            guard=self.Guard(), tolerance_bytes=0)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(result['lengths'][0]['cycles']), 4)
        self.assertTrue(result['lengths'][0]['cycles'][0]['warmup'])

    def test_collection_phase(self):
        result = run_trials(self.Adapter(), mx, [[1]], cycles=3,
                            guard=self.Guard(), collect=True)
        phase = result['lengths'][0]['collection']
        self.assertEqual(phase['residual_count'], 3)
        self.assertEqual(phase['restored_max_logit_delta'], 0.)
        self.assertIn('memory_after_release', phase)

    def test_collection_rejects_invalid_residuals(self):
        for values in ({0: mx.ones((4,))},
                       {i: mx.ones((3,)) for i in range(3)},
                       {i: mx.array([float('nan')] * 4) for i in range(3)}):
            adapter = self.Adapter()
            adapter.collect_last_hidden = lambda tokens: values
            with self.assertRaises(ValueError):
                run_trials(adapter, mx, [[1]], cycles=3, guard=self.Guard(), collect=True)
            self.assertIsNone(adapter.strength)

    def test_post_collection_fidelity(self):
        adapter = self.Adapter(broken_restore=True)
        original = adapter.collect_last_hidden
        def corrupt(tokens):
            adapter.edited = True
            return original(tokens)
        adapter.collect_last_hidden = corrupt
        with self.assertRaisesRegex(ValueError, 'Post-collection'):
            run_trials(adapter, mx, [[1]], cycles=3, guard=self.Guard(), collect=True)
        self.assertIsNone(adapter.strength)

    def test_leak_rejected(self):
        result = run_trials(self.Adapter(), mx, [[1]], cycles=3,
                            guard=self.Guard(True), tolerance_bytes=0)
        self.assertEqual(result['status'], 'failed')

    def test_zero_and_restore_gates(self):
        for kwargs in ({'broken_zero': True}, {'broken_restore': True}):
            adapter = self.Adapter(**kwargs)
            with self.assertRaises(ValueError):
                run_trials(adapter, mx, [[1]], cycles=3, guard=self.Guard())
            self.assertIsNone(adapter.strength)

    def test_reset_after_exception(self):
        adapter = self.Adapter(crash=True)
        with self.assertRaises(RuntimeError):
            run_trials(adapter, mx, [[1]], cycles=3, guard=self.Guard())
        self.assertIsNone(adapter.strength)
