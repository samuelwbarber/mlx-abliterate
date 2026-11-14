# MLX Abliterate

Memory-bounded MLX abliteration and a native Metal serving extension for the M5 Pro / 48 GB Mac. The tool collects contrast directions, searches reversible edits against pinned Heretic inputs, exports the selected model, and verifies the actual imported model. This is an independent project, not an official Heretic release.

**Validated local model:** `qwen3.8-abliterated:27b-m5`, served by the isolated native runtime on port 11435. The original `qwen3.8:27b-mlx` on port 11434 is preserved. Model weights and direction vectors are local artifacts, not public downloads in this repository.

| Measured check | Stock | Edited |
|---|---:|---:|
| Heretic standard100 keyword refusals | 98/100 | 2/100 |
| Served exact-answer checks | 11/12 | 12/12 |
| Median warm decode, balanced four-block check | 78.06 tokens/s | 74.79 tokens/s |

Every stock-correct answer was retained. Zero-edit outputs matched stock exactly on the tested schedule, and all 2,017 base tensor payloads remained unchanged. The candidate adds 128 FP32 vectors (2.5 MiB); the fused Metal correction avoids full-model BF16 expansion. All 199 Python tests and focused native numerical tests passed.

These are optimized benchmark results, not independent held-out evaluation. Heretic scoring uses fixed 100-token windows and its pinned keyword matcher; the 12 exact-answer checks are narrow. Speed uses all 40 warm samples per model, alternating model order, with competing browser GPU work paused equally for both arms. [Full evidence](research/evidence/native-final.json) · [Candidate and limitations](docs/CANDIDATE.md).

## Run the installed model

On the development Mac:

```sh
cd ~/mlx-abliterate
python3 scripts/native-model.py
```

The launcher checks the validated binary and model digest, starts the isolated server if needed, and opens chat with thinking disabled as tested. Add a prompt to run once, or `--status` to inspect the installation. It uses a 4096-token default context. Ordinary Ollama cannot load this custom architecture; use this launcher or explicitly point a client at `http://127.0.0.1:11435`.

The earlier `qwen3.8-abliterated:27b-mlx` tag on port 11434 is a rejected historical experiment. Use the new `27b-m5` model through the native launcher.

## Approach

Keep the original quantized weights immutable during search. Evaluate small, reversible output-projection edits in MLX. Export only the selected modification through a bounded-memory writer, then use Ollama for everyday inference.

The first implementation gate is an **unchanged-model round trip**. All 2,017 tensor payloads and runtime metadata survived export/import. Ten paired short-prompt outputs matched exactly; median warm decode was 67.68 tokens/s for stock and 70.76 for the round trip. This passes the initial short serving check, not the full tool/vision/long-context evaluation. [Measured results and limitations](docs/RESULTS.md).

## Install and inspect

Python 3.11 or newer. Read-only inventory, no-op export, and Ollama benchmarks use the standard library; Apple Silicon MLX is optional.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
mlx-abliterate inspect qwen3.8:27b-mlx --verify-payloads --output runs/inventory.json
mlx-abliterate plan-export qwen3.8:27b-mlx
```

For the tested Apple Silicon Python 3.11 runtime and standard-data tools, install `pip install -c constraints-macos-py311.txt -e '.[runtime,data]'`. Keep only one large model resident. Outputs and run records are excluded from Git.

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
- [Generation](docs/GENERATION.md): bounded greedy inference and continuation scoring.
- [Small study](docs/STUDY.md): disjoint data, proxy objectives and candidate acceptance gates.
- [Stability probe](docs/STABILITY.md): bounded repeated-prefill validation and remaining memory gates.
- [Local evidence](research/evidence/): hardware inventory and prior benchmark measurements, without credentials or device identifiers.

## What success means

A reproducible local experiment that changes the intended refusal behavior, stays within the Mac's memory budget, preserves coding and tool-use performance within predeclared tolerances, and produces a separate model that remains close to the original model's measured Ollama speed.

This is a systems and evaluation project. It is not a claim that abliteration improves intelligence, that all refusals have a single cause, or that a small KL divergence proves capability retention.

## Repository and licensing

Original implementation and documentation are licensed under MIT. The native Ollama patch includes upstream context under its MIT license, preserved with pinned provenance in `native/ollama/`. Heretic is AGPL-3.0-or-later; the inspected MLX port has an Apache-2.0 license file. Their licenses do not change with this project's license. Record provenance before incorporating upstream code; see the source register.

Development follows the staged plan, with public incremental commits. Model weights, local run artifacts, and private datasets are excluded.
