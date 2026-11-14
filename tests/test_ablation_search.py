import copy
import unittest
from mlx_abliterate.ablation_search import (default_candidates,validate_candidates,projection_directions,
                                           summarize_windows,export_eligible)


def windows(refusals):
    return [{'status':'truncated','nonempty':True,'keyword_refusal':i<refusals} for i in range(100)]


class AblationSearchTests(unittest.TestCase):
    def test_standard_only_requires_explicit_bounded_schedule(self):
        from mlx_abliterate.ablation_search import validate_search_mode
        for candidates in (None, [], default_candidates()[:1], default_candidates()[:6]):
            with self.assertRaisesRegex(ValueError, 'explicit zero'):
                validate_search_mode(candidates, True)
        for value in (1, 'yes', None):
            with self.assertRaisesRegex(ValueError, 'boolean'):
                validate_search_mode(default_candidates()[:2], value)
        self.assertEqual(validate_search_mode(None, False), default_candidates())
        schedule=default_candidates()[:2]
        schedule[0]['strength']=.1
        with self.assertRaises(ValueError):
            validate_search_mode(schedule,True)

    def test_focused_schedule_marks_skips_and_preserves_all_candidates(self):
        from mlx_abliterate.ablation_search import focused_standard_schedule
        schedule=default_candidates()[:5]
        screen,expanded,selected=focused_standard_schedule(schedule)
        self.assertEqual(selected,schedule)
        self.assertEqual(selected[0]['id'],'zero')
        for stage,report in [('screen',screen),('expanded',expanded)]:
            self.assertEqual(report['stage'],stage)
            self.assertEqual(report['status'],'skipped')
            self.assertEqual(report['reason'],'focused standard optimization; not heldout')
            self.assertEqual(report['trials'],[])
            self.assertEqual(report['eligible_candidates'],0)
            self.assertIsNone(report['best_candidate_id'])

    def test_standard_only_dispatch_does_not_trim_explicit_candidates(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from mlx_abliterate.ablation_search import run_search
        schedule=default_candidates()[:5]
        with tempfile.TemporaryDirectory() as folder:
            with patch('mlx_abliterate.ablation_search._run_search',return_value={'status':'fixture'}) as run:
                result=run_search('export','data','directions','cap','rub',folder,
                                  candidates=schedule,shortlist_size=1,standard_only=True)
        self.assertEqual(result,{'status':'fixture'})
        self.assertEqual(run.call_args.args[-3:],(schedule,1,True))

    def test_progress_is_stderr_only(self):
        import contextlib,io
        from mlx_abliterate.ablation_search import _progress
        stdout,stderr=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            _progress('screen','candidate refusals 2/32')
        self.assertEqual(stdout.getvalue(),'')
        self.assertIn('screen: candidate refusals 2/32',stderr.getvalue())

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

    def test_lower_kl_wins_once_refusal_and_retention_gates_pass(self):
        from mlx_abliterate.ablation_search import standard_score
        self.assertLess(standard_score(5,.1),standard_score(5,.2))
        self.assertLess(standard_score(3,.1),standard_score(0,10.))
        self.assertEqual(standard_score(0,.1),standard_score(5,.1))
        for value in (float('nan'),float('inf'),-1):
            with self.assertRaises(ValueError):standard_score(0,value)

    def test_standard_selector_uses_eligible_kl_then_refusals_then_schedule(self):
        from mlx_abliterate.ablation_search import select_standard_trial
        def trial(index,kl,refusals,eligible=True):
            return {'index':index,'candidate_id':str(index),'result':{'eligible':eligible,'score':kl,'metrics':{'refusals':refusals}}}
        rows=[trial(0,0.,0,False),trial(1,10.,0),trial(2,.1,3),trial(3,.1,2),trial(4,.1,2)]
        self.assertEqual(select_standard_trial(rows)['index'],3)
        self.assertIsNone(select_standard_trial([rows[0]]))

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
