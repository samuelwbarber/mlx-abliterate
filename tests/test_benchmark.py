import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mlx_abliterate.benchmark import compare_reports, load_prompts, run_benchmark


class Server(BaseHTTPRequestHandler):
    requests = []
    mode = 'ok'

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"version":"fixture-1"}')

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.requests.append(payload)
        self.send_response(200)
        self.end_headers()
        if payload.get('keep_alive') == 0:
            self.wfile.write(b'{"done":true}')
            return
        if self.mode == 'error':
            self.wfile.write(b'{"error":"private server data"}\n')
            return
        self.wfile.write(b'{"response":"private generated text","done":false}\n')
        if self.mode == 'incomplete':
            return
        final = {'done': True, 'done_reason': 'stop', 'eval_count': 10,
                 'eval_duration': 200_000_000, 'prompt_eval_count': 8,
                 'prompt_eval_duration': 100_000_000, 'load_duration': 0,
                 'total_duration': 300_000_000}
        if self.mode == 'truncated':
            final['done_reason'] = 'length'
        if self.mode == 'invalid':
            final['eval_duration'] = -1
        self.wfile.write(json.dumps(final).encode() + b'\n')


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Server)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        Server.mode = 'ok'
        Server.requests = []

    def run_report(self, **kwargs):
        return run_benchmark('fixture:model', ['first prompt', 'second prompt'],
                             base_url=self.url, **kwargs)

    def test_measurements_and_no_raw_text(self):
        report = self.run_report(cold=True)
        self.assertEqual(len(report['cold_samples']), 1)
        self.assertEqual(len(report['warmup_samples']), 2)
        self.assertEqual(len(report['samples']), 10)
        self.assertEqual(report['summary']['median_decode_tokens_per_second'], 50)
        self.assertEqual(Server.requests[0]['keep_alive'], 0)
        self.assertIsNotNone(report['samples'][0]['first_token_seconds'])
        serialized = json.dumps(report)
        self.assertNotIn('private generated text', serialized)
        self.assertNotIn('first prompt', serialized)
        self.assertEqual(compare_reports(report, copy.deepcopy(report))['status'], 'pass')

    def test_errors_and_truncation_cannot_pass(self):
        for mode, status in [('error', 'runtime_error'), ('invalid', 'runtime_error'),
                             ('incomplete', 'runtime_error'), ('truncated', 'truncated')]:
            with self.subTest(mode=mode):
                Server.mode = mode
                report = self.run_report(repeats=1)
                self.assertEqual(report['samples'][0]['status'], status)
                self.assertEqual(report['summary']['valid_samples'], 0)
                self.assertEqual(compare_reports(report, report)['status'], 'inconclusive')
                self.assertNotIn('private server data', json.dumps(report))

    def test_speed_gate_and_mismatched_settings(self):
        report = self.run_report()
        slower = copy.deepcopy(report)
        for sample in slower['samples']:
            sample['decode_tokens_per_second'] *= .8
        self.assertEqual(compare_reports(report, slower)['status'], 'fail')
        slower['settings']['think'] = True
        self.assertEqual(compare_reports(report, slower)['status'], 'inconclusive')

    def test_prompt_token_count_mismatch(self):
        report = self.run_report()
        candidate = copy.deepcopy(report)
        candidate['samples'][0]['prompt_eval_count'] += 1
        self.assertEqual(compare_reports(report, candidate)['status'], 'inconclusive')

    def test_prompt_file_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prompts.json'
            path.write_text('["one", "two"]')
            self.assertEqual(load_prompts(path), ['one', 'two'])
            path.write_text('[""]')
            with self.assertRaises(ValueError):
                load_prompts(path)
        for kwargs in [{'timeout': float('nan')}, {'warmups': 0},
                       {'options': {'num_predict': -1}}, {'repeats': True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_report(**kwargs)
        with self.assertRaises(ValueError):
            run_benchmark('x', ['y'], base_url='http://name:secret@localhost')


if __name__ == '__main__':
    unittest.main()
