import json
from pathlib import Path
import tempfile
import unittest

from mlx_abliterate.cli import parser, write_report


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
