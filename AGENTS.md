# Project instructions

## Current phase

The user has approved implementation with coordinated subagents and a public GitHub repository with incremental commits and pushes. Follow the staged plan: build and validate inventory and the unchanged-model round trip first. Do not bypass failed fidelity gates or modify the stock model. Agents own separate files; the lead integrates, commits, and publishes.

## Project priorities

- Target Qwen3.8-27B on the documented M5 Pro / 48 GB machine first.
- Preserve the existing `qwen3.8:27b-mlx` installation. Treat its manifest and content-addressed blobs as read-only.
- Make format fidelity, capability retention, memory use, and serving speed independently measurable.
- Keep facts, local observations, derived calculations, hypotheses, and acceptance targets distinct.
- Prefer pinned primary sources and source inspection over model-card marketing or cross-hardware extrapolation.
- Do not conflate MLX research throughput with Ollama serving throughput.
- Do not copy implementation from upstream without recording its license and provenance.

## Future implementation constraints

- No full-model BF16 expansion or per-trial weight copies on this 48 GB target.
- No silent CPU fallback, unknown tensor layouts, swallowed generation failures, or cross-trial reuse of invalid caches.
- No overwrite of the stock model; export to new paths and new tags through supported import mechanisms.
- Keep weights, prompt corpora, generated responses, checkpoints, secrets, and runtime logs out of Git.
- Treat model-generated code as untrusted in execution benchmarks; use a separate disposable test environment with explicit limits.
- Implement only the next accepted milestone; do not bypass a failed fidelity or export gate to produce a superficially working result.
