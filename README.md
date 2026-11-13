# MLX Abliterate

Research and build plan for modifying a quantized 27B language model on a 48 GB Apple Silicon Mac, while preserving its usefulness for coding and its fast Ollama deployment.

**Status: experimental research toolkit.** The unchanged 27B export/import and bounded MLX loader have been tested on the target Mac. Direction collection, reversible edits, drift pilots, resumable trial primitives, and selected-projection merging are implemented. There is no validated abliterated 27B checkpoint or claim of preserved coding quality yet. No model weights are included. The working name is `mlx-abliterate`; this is not an official Heretic project.

The first target is the user's existing `qwen3.8:27b-mlx`: Qwen3.8-27B in Ollama's NVFP4 format, on an M5 Pro with 20 GPU cores. A prior same-prompt comparison measured 65.17 generated tokens/s on its warmed Ollama run, versus 12.35 tokens/s for a different RANA 6-bit checkpoint in `mlx-vlm`. Those results motivate preserving the existing serving path; they do not isolate the cause of the difference or establish relative quality.

## Approach

Keep the original quantized weights immutable during search. Evaluate small, reversible output-projection edits in MLX. Export only the selected modification through a bounded-memory writer, then use Ollama for everyday inference.

The first implementation gate is an **unchanged-model round trip**. All 2,017 tensor payloads and runtime metadata survived export/import. Ten paired short-prompt outputs matched exactly; median warm decode was 67.68 tokens/s for stock and 70.76 for the round trip. This passes the initial short serving check, not the full tool/vision/long-context evaluation. [Measured results and limitations](docs/RESULTS.md).

## Install and inspect

Python 3.11 or newer. Read-only inventory, no-op export, and Ollama benchmarks use the standard library; Apple Silicon MLX is optional.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
mlx-abliterate inspect qwen3.8:27b-mlx --verify-payloads --output runs/inventory.json
mlx-abliterate plan-export qwen3.8:27b-mlx
```

For model collection, edits, and merging on Apple Silicon, install `pip install -e '.[runtime]'`. Keep only one large model resident. Outputs and run records are excluded from Git.

```bash
mlx-abliterate export qwen3.8:27b-mlx outputs/stock-noop
ollama create mlx-abliterate-noop:27b -f outputs/stock-noop/Modelfile
mlx-abliterate verify-import mlx-abliterate-noop:27b outputs/stock-noop
```

Use a new tag for imports. Export destinations and reports refuse overwrites. See [usage and dataset requirements](docs/USAGE.md), [MLX fidelity](docs/MLX_FIDELITY.md), and [merge semantics](docs/MERGE.md) before running experiments. `pilot` reports first-token drift and timing; it does not automatically select a safe or capable checkpoint. The search module exposes a Python callback interface requiring an explicit evaluator.

## Test

```bash
python -m unittest discover -s tests -v
```

Optional numerical tests run when MLX is installed and otherwise skip. Linux CI checks the standard-library paths; local Apple Silicon tests are required for numerical changes.

## Read the plan

- [Build plan](docs/PLAN.md): scope, milestones, acceptance gates, and first implementation slice.
- [Technical design](docs/DESIGN.md): quantized edits, memory budget, hybrid-model handling, and export.
- [Research findings](docs/RESEARCH.md): source-code audit, existing approaches, and unresolved questions.
- [Evaluation protocol](docs/EVALUATION.md): capability retention, refusal measurement, runtime comparisons, and ETA calculation.
- [Sources](research/SOURCES.md): primary references and pinned source revisions.
- [Stability probe](docs/STABILITY.md): bounded repeated-prefill validation and remaining memory gates.
- [Local evidence](research/evidence/): hardware inventory and prior benchmark measurements, without credentials or device identifiers.

## What success means

A reproducible local experiment that changes the intended refusal behavior, stays within the Mac's memory budget, preserves coding and tool-use performance within predeclared tolerances, and produces a separate model that remains close to the original model's measured Ollama speed.

This is a systems and evaluation project. It is not a claim that abliteration improves intelligence, that all refusals have a single cause, or that a small KL divergence proves capability retention.

## Repository and licensing

Original implementation and documentation are licensed under MIT. No upstream implementation is vendored. Heretic is AGPL-3.0-or-later; the inspected MLX port has an Apache-2.0 license file. Their licenses do not change with this project's license. Record provenance before incorporating upstream code; see the source register.

Development follows the staged plan, with public incremental commits. Model weights, local run artifacts, and private datasets are excluded.
