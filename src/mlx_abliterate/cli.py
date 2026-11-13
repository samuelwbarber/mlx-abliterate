"""Command-line entry points. Model-changing commands are explicit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from . import __version__


def write_report(report: dict, output: Path | None) -> None:
    """Atomically create a report, refusing to overwrite a prior experiment."""
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output is None:
        print(payload, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".report-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic and, unlike replace, refuses an existing destination.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="MLX model fidelity and ablation research tools")
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    for name, description in (("inspect", "Read model metadata without loading weights"),
                              ("plan-export", "Plan an unchanged export without writing weights"),
                              ("export", "Write an unchanged interchange artifact to a new directory")):
        sub = commands.add_parser(name, help=description)
        sub.add_argument("model")
        sub.add_argument("--models-dir", type=Path)
        sub.add_argument("--output", type=Path, help="New JSON report path (default: stdout)")
        if name == "export":
            sub.add_argument("destination", type=Path)
        if name == "inspect":
            sub.add_argument("--verify-payloads", action="store_true",
                             help="Stream SHA256 over all model blobs")
    verify = commands.add_parser("verify-import", help="Check imported payloads against export receipt")
    verify.add_argument("model")
    verify.add_argument("export_directory", type=Path)
    verify.add_argument("--models-dir", type=Path)
    verify.add_argument("--output", type=Path)
    verify_export = commands.add_parser("verify-export", help="Verify local export bytes and content identity")
    verify_export.add_argument("export_directory", type=Path)
    verify_export.add_argument("--require-unmodified", action="store_true")
    verify_export.add_argument("--output", type=Path)
    bench = commands.add_parser("baseline", help="Measure Ollama generation; does not execute generated code")
    bench.add_argument("model")
    bench.add_argument("--prompts", required=True, type=Path)
    bench.add_argument("--output", required=True, type=Path)
    bench.add_argument("--base-url", default="http://127.0.0.1:11434")
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--warmups", type=int, default=1)
    bench.add_argument("--num-predict", type=int, default=512)
    bench.add_argument("--num-ctx", type=int, default=4096)
    bench.add_argument("--timeout", type=float, default=120)
    bench.add_argument("--think", action="store_true")
    bench.add_argument("--cold", action="store_true", help="Unload this model before the first request")
    bench.add_argument("--environment", type=Path, help="JSON with observed power/thermal/MTP settings")
    compare = commands.add_parser("compare", help="Compare matched completed benchmark reports")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path)
    collect = commands.add_parser("collect", help="Extract contrast directions from local, disjoint data")
    collect.add_argument("export_directory", type=Path)
    collect.add_argument("dataset", type=Path)
    collect.add_argument("destination", type=Path)
    collect.add_argument("--positive-group", required=True)
    collect.add_argument("--negative-group", required=True)
    collect.add_argument("--output", type=Path)
    pilot = commands.add_parser("pilot", help="Measure first-token drift; does not select an edited model")
    pilot.add_argument("export_directory", type=Path)
    pilot.add_argument("dataset", type=Path)
    pilot.add_argument("directions_directory", type=Path)
    pilot.add_argument("--strengths", type=float, nargs='+', default=[0., .05, .1])
    pilot.add_argument("--max-prompts", type=int, default=8)
    pilot.add_argument("--output", type=Path, required=True)
    merge = commands.add_parser("merge", help="Write explicitly selected NVFP4 edits to a new export")
    merge.add_argument("export_directory", type=Path)
    merge.add_argument("destination", type=Path)
    merge.add_argument("--edit-spec", required=True, type=Path)
    merge.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"Report already exists: {args.output}")
        if args.command == "inspect":
            from .inventory import inspect_model
            report = inspect_model(args.model, models_dir=args.models_dir,
                                   verify_payloads=args.verify_payloads)
        elif args.command == "verify-export":
            from .export import verify_export
            report = verify_export(args.export_directory, require_unmodified=args.require_unmodified)
        elif args.command == "verify-import":
            from .export import verify_import
            report = verify_import(args.model, args.export_directory, models_dir=args.models_dir)
        elif args.command in {"plan-export", "export"}:
            from .export import export_model, plan_export
            if args.command == "plan-export":
                report = plan_export(args.model, models_dir=args.models_dir)
            else:
                report = export_model(args.model, args.destination, models_dir=args.models_dir)
        elif args.command == "baseline":
            from .benchmark import load_prompts, run_benchmark
            report = run_benchmark(
                args.model, load_prompts(args.prompts), base_url=args.base_url,
                repeats=args.repeats, warmups=args.warmups, timeout=args.timeout,
                think=args.think, cold=args.cold,
                options={"temperature": 0, "seed": 42, "num_predict": args.num_predict,
                         "num_ctx": args.num_ctx},
                environment=json.loads(args.environment.read_text()) if args.environment else None,
            )
        elif args.command == "merge":
            from .merge import merge_export
            if args.edit_spec.stat().st_size > 8 * 1024 * 1024:
                raise ValueError('Edit specification exceeds 8 MiB')
            report = merge_export(args.export_directory, args.destination,
                                  json.loads(args.edit_spec.read_text()))
        elif args.command == "collect":
            from .workflow import collect_directions
            report = collect_directions(args.export_directory, args.dataset, args.destination,
                                        positive_group=args.positive_group,
                                        negative_group=args.negative_group)
        elif args.command == "pilot":
            from .workflow import drift_pilot
            report = drift_pilot(args.export_directory, args.dataset, args.directions_directory,
                                 strengths=args.strengths, max_prompts=args.max_prompts)
        else:
            from .benchmark import compare_reports
            report = compare_reports(json.loads(args.baseline.read_text()),
                                     json.loads(args.candidate.read_text()))
        write_report(report, args.output)
        if args.command == "compare" and report.get("status") != "pass":
            return 2
        if args.command == "baseline" and report.get("status") != "complete":
            return 2
        return 0
    except (ValueError, OSError, RuntimeError, ModuleNotFoundError) as exc:
        print(f"mlx-abliterate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
