# MLX Abliterate

Research and build plan for modifying a quantized 27B language model on a 48 GB Apple Silicon Mac, while preserving its usefulness for coding and its fast Ollama deployment.

**Status: early implementation.** Inventory, format fidelity, and benchmark tooling are under development. There is no completed ablation experiment or validated edited 27B model yet. No model weights are included. The working name is `mlx-abliterate`; this is not an official Heretic project.

The first target is the user's existing `qwen3.8:27b-mlx`: Qwen3.8-27B in Ollama's NVFP4 format, on an M5 Pro with 20 GPU cores. A prior same-prompt comparison measured 65.17 generated tokens/s on its warmed Ollama run, versus 12.35 tokens/s for a different RANA 6-bit checkpoint in `mlx-vlm`. Those results motivate preserving the existing serving path; they do not isolate the cause of the difference or establish relative quality.

## Proposed approach

Keep the original quantized weights immutable during search. Evaluate small, reversible output-projection edits in MLX. Export only the selected modification through a bounded-memory writer, then use Ollama for everyday inference.

The first implementation gate is an **unchanged-model round trip**. Before optimizing any refusal direction, demonstrate that the exact installed model can be inspected, represented correctly, and imported as a separate Ollama tag without losing its speed, template, tools, vision metadata, or speculative decoding support.

## Read the plan

- [Build plan](docs/PLAN.md): scope, milestones, acceptance gates, and first implementation slice.
- [Technical design](docs/DESIGN.md): quantized edits, memory budget, hybrid-model handling, and export.
- [Research findings](docs/RESEARCH.md): source-code audit, existing approaches, and unresolved questions.
- [Evaluation protocol](docs/EVALUATION.md): capability retention, refusal measurement, runtime comparisons, and ETA calculation.
- [Sources](research/SOURCES.md): primary references and pinned source revisions.
- [Local evidence](research/evidence/): hardware inventory and prior benchmark measurements, without credentials or device identifiers.

## What success means

A reproducible local experiment that changes the intended refusal behavior, stays within the Mac's memory budget, preserves coding and tool-use performance within predeclared tolerances, and produces a separate model that remains close to the original model's measured Ollama speed.

This is a systems and evaluation project. It is not a claim that abliteration improves intelligence, that all refusals have a single cause, or that a small KL divergence proves capability retention.

## Repository and licensing

Original implementation and documentation are licensed under MIT. No upstream implementation is vendored. Heretic is AGPL-3.0-or-later; the inspected MLX port has an Apache-2.0 license file. Their licenses do not change with this project's license. Record provenance before incorporating upstream code; see the source register.

Development follows the staged plan, with public incremental commits. Model weights, local run artifacts, and private datasets are excluded.
