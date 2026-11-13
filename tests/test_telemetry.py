import unittest
from unittest.mock import patch

from mlx_abliterate.telemetry import MemoryGuard, MemoryBudgetExceeded, estimate_remaining


class TelemetryTests(unittest.TestCase):
    def test_memory_pools_not_added(self):
        guard = MemoryGuard(1000, 900, 100)
        with patch('mlx_abliterate.telemetry.snapshot', return_value={
            'process_peak_rss_bytes': 950, 'mlx_active_bytes': 850}):
            self.assertEqual(guard.check()['process_peak_rss_bytes'], 950)
        with patch('mlx_abliterate.telemetry.snapshot', return_value={'process_peak_rss_bytes': 1001}):
            with self.assertRaises(MemoryBudgetExceeded):
                guard.check()

    def test_no_eta_before_pilot(self):
        with self.assertRaises(ValueError):
            estimate_remaining([1, 2], 50)
        result = estimate_remaining([2, 3, 10], 5)
        self.assertEqual(result['estimated_trial_seconds'], 15)
        self.assertEqual(result['observed_range_seconds'], [10, 50])
