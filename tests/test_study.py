import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_abliterate.study import export_gate, passes_retention, _store_identical


def results(nll=1., first=True, second=False):
    return [{'id':'proxy','kind':'refusal_likelihood','status':'completed','mean_nll':nll},
            {'id':'a','kind':'exact_text','status':'completed','task_check_passed':first,'response_sha256':'a'},
            {'id':'b','kind':'exact_text','status':'completed','task_check_passed':second,'response_sha256':'b'}]


class StudyGateTests(unittest.TestCase):
    def test_equal_total_cannot_hide_task_regression(self):
        self.assertFalse(passes_retention(results(), results(first=False,second=True)))

    def test_heldout_failure_blocks_export(self):
        failed=results(1.2)
        failed[1]['status']='truncated'
        self.assertFalse(export_gate({'strength':.5},results(),results(1.2),results(),failed))

    def test_requires_nonzero_and_both_prespecified_gains(self):
        self.assertTrue(export_gate({'strength':.5},results(),results(1.2),results(),results(1.2)))
        self.assertFalse(export_gate({'strength':0.},results(),results(1.2),results(),results(1.2)))
        self.assertFalse(export_gate({'strength':.5},results(),results(1.01),results(),results(1.2)))
        self.assertFalse(export_gate({'strength':.5},results(),results(1.2),results(),results(1.01)))

    def test_nan_proxy_cannot_pass(self):
        self.assertFalse(export_gate({'strength':.5},results(),results(float('nan')),results(),results(1.2)))

    def test_resume_rejects_changed_baseline_hash(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'baseline.json'
            baseline={'results':results()}
            _store_identical(path,baseline)
            _store_identical(path,baseline)
            changed=copy.deepcopy(baseline)
            changed['results'][1]['response_sha256']='changed'
            with self.assertRaises(ValueError):
                _store_identical(path,changed)
            self.assertNotIn('changed',path.read_text())
