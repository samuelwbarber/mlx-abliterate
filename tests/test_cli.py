import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlx_abliterate.cli import main, parser, write_report


class ReportTests(unittest.TestCase):
    def test_heretic_prepare_parse_and_dispatch(self):
        arguments = ['heretic-prepare', 'datasets/heretic-standard']
        args = parser().parse_args(arguments)
        self.assertEqual(args.destination, Path('datasets/heretic-standard'))
        self.assertIsNone(args.output)
        summary = {'counts': {'harmful_test': 100}}
        with patch('mlx_abliterate.heretic_data.prepare_standard_data', return_value=summary) as prepare, \
                patch('mlx_abliterate.cli.write_report') as write:
            self.assertEqual(main(arguments), 0)
        prepare.assert_called_once_with(Path('datasets/heretic-standard'))
        write.assert_called_once_with(summary, None)

    def test_heretic_collect_and_search_parse_and_dispatch(self):
        collect_args = ['heretic-collect', 'export', 'datasets/standard', 'outputs/directions']
        args = parser().parse_args(collect_args)
        self.assertFalse(args.no_orthogonalize)
        self.assertIsNone(args.output)
        with patch('mlx_abliterate.heretic_collection.collect_standard', return_value={'counts': {}}) as collect, \
                patch('mlx_abliterate.cli.write_report') as write:
            self.assertEqual(main(collect_args + ['--no-orthogonalize']), 0)
        collect.assert_called_once_with(Path('export'), Path('datasets/standard'), Path('outputs/directions'),
                                        orthogonalize=False)
        write.assert_called_once_with({'counts': {}}, None)
        search_args = ['heretic-search', 'export', 'benchmark', 'directions', 'capability.jsonl',
                       'rubrics.json', 'runs/search', '--shortlist-size', '2']
        args = parser().parse_args(search_args)
        self.assertEqual(args.shortlist_size, 2)
        self.assertIsNone(args.candidates)
        with patch('mlx_abliterate.ablation_search.run_search', return_value={'export_gate_passed': False}) as search, \
                patch('mlx_abliterate.cli.write_report'):
            self.assertEqual(main(search_args), 2)
        search.assert_called_once_with(Path('export'), Path('benchmark'), Path('directions'),
                                       Path('capability.jsonl'), Path('rubrics.json'), Path('runs/search'),
                                       candidates=None, shortlist_size=2)

    def test_heretic_benchmark_parse_and_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'summary.json'
            arguments = ['heretic-benchmark', 'model:tag', 'prompts.json', 'config.json', 'runs/private',
                         '--output', str(output)]
            args = parser().parse_args(arguments)
            self.assertEqual(args.target_rate, .05)
            self.assertEqual(args.timeout, 120.)
            with patch('mlx_abliterate.heretic_benchmark.run',
                       return_value={'summary': {'keyword_rate_target_passed': False}}) as run:
                self.assertEqual(main(arguments), 2)
            run.assert_called_once_with('model:tag', Path('prompts.json'), Path('config.json'), Path('runs/private'),
                                        base_url='http://127.0.0.1:11434', timeout=120., target_rate=.05)
            self.assertTrue(output.exists())

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
