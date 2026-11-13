import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlx_abliterate.heretic_benchmark import keyword_refusal, run, sample, summarize


class KeywordTests(unittest.TestCase):
    def test_upstream_normalization_and_empty(self):
        for text in (' I **CAN’T**\n\tASSIST with that.', '', '   '):
            self.assertTrue(keyword_refusal(text, ["can't assist"]))
        self.assertFalse(keyword_refusal('Here is the answer', ["can't assist"]))
        self.assertFalse(keyword_refusal('***', ["can't assist"]))

    def test_truncated_windows_score_but_errors_cannot_lower_rate(self):
        records = [{'status': 'truncated', 'keyword_refusal': False} for _ in range(100)]
        self.assertTrue(summarize(records)['keyword_rate_target_passed'])
        records[0] = {'status': 'runtime_error', 'keyword_refusal': False}
        report = summarize(records)
        self.assertFalse(report['keyword_rate_target_passed'])
        self.assertEqual(report['refusal_rate_upper_bound'], .01)
        records[0] = {'status': 'empty', 'keyword_refusal': True}
        report = summarize(records)
        self.assertEqual(report['standard_refusal_rate'], .01)
        self.assertFalse(report['keyword_rate_target_passed'])

    def test_stream_scoring_and_explicit_system(self):
        events = [{'message': {'content': 'I **cannot** '}, 'done': False},
                  {'message': {'content': 'assist.'}, 'done': True, 'done_reason': 'length', 'eval_count':100}]
        config = {'system_prompt':'You are a helpful assistant.', 'max_new_tokens':100, 'keywords':['cannot assist']}
        with patch('mlx_abliterate.heretic_benchmark._request') as request, \
                patch('mlx_abliterate.heretic_benchmark._stream_events', return_value=iter(events)):
            result = sample('model:tag','private prompt',config,base_url='http://localhost:11434',timeout=1)
        self.assertEqual(result['status'], 'truncated')
        self.assertTrue(result['keyword_refusal'])
        self.assertEqual(request.call_args.args[1]['messages'][0]['content'],config['system_prompt'])
        self.assertEqual(request.call_args.args[1]['options']['num_predict'],100)

    def test_resume_checks_identity_and_public_report_excludes_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, config = root/'dataset.json', root/'config.json'
            dataset.write_text(json.dumps({'prompts':[f'PRIVATE PROMPT {i}' for i in range(100)]}))
            config.write_text(json.dumps({'keywords':['cannot assist'],'system_prompt':'You are a helpful assistant.','max_new_tokens':100}))
            import hashlib
            record = {'status':'truncated','response':'PRIVATE RESPONSE','response_sha256':hashlib.sha256(b'PRIVATE RESPONSE').hexdigest(),
                      'keyword_refusal':False}
            metadata = [{'models':[{'name':'model:tag','digest':'sha256:fixture'}]},{'version':'fixture'}]
            with patch('mlx_abliterate.heretic_benchmark._json_request',side_effect=metadata*2), \
                    patch('mlx_abliterate.heretic_benchmark.sample',side_effect=lambda *a,**k:dict(record)) as generate:
                first = run('model:tag',dataset,config,root/'run')
                second = run('model:tag',dataset,config,root/'run')
                self.assertEqual(generate.call_count,100)
            self.assertEqual(first,second)
            self.assertNotIn('PRIVATE',json.dumps(first))
            self.assertEqual(len(first['samples']),100)
            with patch('mlx_abliterate.heretic_benchmark._json_request',side_effect=[
                    {'models':[{'name':'model:tag','digest':'sha256:changed'}]},{'version':'fixture'}]):
                with self.assertRaisesRegex(ValueError,'provenance'):
                    run('model:tag',dataset,config,root/'run')
