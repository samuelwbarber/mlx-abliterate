import copy
import unittest
from mlx_abliterate.ablation_search import (default_candidates,validate_candidates,projection_directions,
                                           summarize_windows,export_eligible)


def windows(refusals):
    return [{'status':'truncated','nonempty':True,'keyword_refusal':i<refusals} for i in range(100)]


class AblationSearchTests(unittest.TestCase):
    def test_default_schedule_and_strength_bounds(self):
        schedule=default_candidates()
        self.assertEqual(validate_candidates(schedule),schedule)
        for bad in (True,float('nan'),-1,2.01):
            changed=copy.deepcopy(schedule)
            changed[1]['strength']=bad
            with self.assertRaises(ValueError):validate_candidates(changed)

    def test_direction_layer_global_or_per_layer_with_window(self):
        paths=[f'model.language_model.layers.{i}.mlp.down_proj' for i in (0,16,24,48)]
        arrays={f'layer.{i}':i for i in range(64)}
        candidate={'start':16,'stop':40,'direction_mode':'global','direction_layer':32}
        self.assertEqual(list(projection_directions(candidate,paths,arrays).values()),[32,32])
        candidate['direction_mode']='per_layer'
        self.assertEqual(list(projection_directions(candidate,paths,arrays).values()),[16,24])

    def test_truncation_valid_but_empty_and_errors_not(self):
        self.assertTrue(summarize_windows(windows(0))['valid'])
        rows=windows(0);rows[0]['status']='runtime_error'
        self.assertFalse(summarize_windows(rows)['valid'])
        rows=windows(0);rows[0]['nonempty']=False
        self.assertFalse(summarize_windows(rows)['valid'])
        rows=windows(0);rows[0].update(status='matched',keyword_refusal=True)
        self.assertTrue(summarize_windows(rows)['valid'])
        self.assertEqual(summarize_windows(rows)['refusals'],1)

    def test_five_percent_is_inclusive_requires_retention_and_nonzero(self):
        cap=[{'id':'a','kind':'exact','status':'completed','task_check_passed':True}]
        self.assertTrue(export_eligible({'strength':1.},windows(90),windows(5),cap,cap))
        self.assertFalse(export_eligible({'strength':1.},windows(90),windows(6),cap,cap))
        self.assertFalse(export_eligible({'strength':0.},windows(90),windows(5),cap,cap))
        lost=[{**cap[0],'task_check_passed':False}]
        self.assertFalse(export_eligible({'strength':1.},windows(90),windows(5),cap,lost))

    def test_path_escape_candidates_rejected(self):
        schedule=default_candidates();schedule[1]['id']='../escape'
        with self.assertRaises(ValueError):validate_candidates(schedule)


class BenignKLTests(unittest.TestCase):
    def test_zero_kl_requires_exact_logits_not_only_shift_invariant_kl(self):
        from mlx_abliterate.ablation_search import require_zero_kl
        require_zero_kl(True,0.)
        for exact,value in ((False,0.),(True,1e-9),(True,float('nan'))):
            with self.assertRaisesRegex(ValueError,'Zero benign'):
                require_zero_kl(exact,value)

    def test_summary_mean_and_nearest_rank_tail(self):
        from mlx_abliterate.ablation_search import summarize_kl
        rows=[{'kl_base_to_candidate':i/100,'max_abs_logit_delta':float(i),'exact_logits':i==0} for i in range(100)]
        summary=summarize_kl(rows)
        self.assertAlmostEqual(summary['mean'],.495)
        self.assertEqual(summary['p95'],.94)
        self.assertEqual(summary['max'],.99)
        self.assertFalse(summary['all_logits_exact'])

    def test_refusal_count_primary_and_lower_kl_breaks_ties(self):
        from mlx_abliterate.ablation_search import standard_score
        self.assertLess(standard_score(5,.1),standard_score(5,.2))
        for count in range(100):
            self.assertLess(standard_score(count,1e300),standard_score(count+1,0.))
        for value in (float('nan'),float('inf'),-1):
            with self.assertRaises(ValueError):standard_score(0,value)

    def test_benign_file_and_ordered_prompt_hashes_are_verified(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from mlx_abliterate.ablation_search import load_benign_test
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            prompts=[f'benign {i}' for i in range(100)]
            raw=json.dumps({'prompts':prompts}).encode()
            (root/'harmless-test100.json').write_bytes(raw)
            subset={'sha256':hashlib.sha256(raw).hexdigest(),'ordered_prompt_sha256':[hashlib.sha256(p.encode()).hexdigest() for p in prompts]}
            lock={'datasets':{'harmless':{'subsets':{'test[:100]':subset}}}}
            (root/'dataset-lock.json').write_text(json.dumps(lock))
            self.assertEqual(load_benign_test(root)[0],prompts)
            subset['ordered_prompt_sha256'][0]='wrong'
            (root/'dataset-lock.json').write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError,'identity mismatch'):load_benign_test(root)
