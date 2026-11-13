import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlx_abliterate.cli import main, parser, write_report


class ReportTests(unittest.TestCase):
    def test_atomic_report_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run.json"
            write_report({"ok": True}, output)
            with self.assertRaises(FileExistsError):
                write_report({"ok": False}, output)
            self.assertEqual(json.loads(output.read_text()), {"ok": True})
            self.assertEqual(list(Path(folder).iterdir()), [output])

    def test_reject_nan(self):
        with self.assertRaises(ValueError):
            write_report({"rate": float("nan")}, None)

    def test_inspect_parse(self):
        args = parser().parse_args(["inspect", "example:tag"])
        self.assertEqual(args.model, "example:tag")
        self.assertIsNone(args.models_dir)

    def test_collect_and_pilot_parse(self):
        args = parser().parse_args(['collect', 'export', 'data.jsonl', 'directions',
                                   '--positive-group', 'a', '--negative-group', 'b'])
        self.assertEqual(args.positive_group, 'a')
        args = parser().parse_args(['pilot', 'export', 'data.jsonl', 'directions',
                                   '--output', 'pilot.json'])
        self.assertEqual(args.strengths, [0., .05, .1])

    def test_generation_commands_and_exit_status(self):
        with tempfile.TemporaryDirectory() as folder:
            for command, status, expected in [('generate', 'completed', 0),
                                               ('generate', 'truncated', 2),
                                               ('evaluate-nll', 'runtime_error', 2)]:
                output = Path(folder) / f'{command}-{status}.json'
                arguments = [command, 'export', '--prompt', 'prompt.txt', '--output', str(output)]
                if command == 'evaluate-nll':
                    arguments.extend(['--target', 'target.txt'])
                else:
                    arguments.extend(['--max-new-tokens', '64'])
                with patch('mlx_abliterate.generation.evaluate_export', return_value={'status': status}) as run:
                    self.assertEqual(main(arguments), expected)
                self.assertEqual(json.loads(output.read_text())['status'], status)
                self.assertEqual(run.call_args.kwargs['max_new_tokens'], 64 if command == 'generate' else 128)
                self.assertEqual(run.call_args.kwargs['target_file'],
                                 Path('target.txt') if command == 'evaluate-nll' else None)
                with patch('mlx_abliterate.generation.evaluate_export') as run:
                    self.assertEqual(main(arguments), 1)
                    run.assert_not_called()
