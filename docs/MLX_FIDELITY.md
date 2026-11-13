# Exact-source MLX adapter

The adapter in `src/mlx_abliterate/mlx_adapter.py` loads the project's no-op interchange export using MLX-VLM's installed Qwen3.5/3.8 implementation. It does not reproduce the transformer forward loop. This is a narrow, text-only research interface for the inspected dense 64-layer, width-5120 model, not a general Hugging Face loader.

## Format mapping

The no-op export retains Ollama's packed U32 weight arrays. The inspected MLX-VLM ModelOpt importer expects U8 arrays and subsequently views them as U32. Passing this export blindly to that importer would fail or misinterpret the format.

The adapter instead verifies every indexed shard header and maps `.weight_scale` to native `.scales`, preserving U32 weights and scalar `.weight_scale_2`. It constructs the upstream model configuration with native NVFP4 quantization, calls upstream model sanitization, then uses strict `load_model_weights`. `ScaledQuantizedLinear` applies the scalar multiplier before conversion back to the input dtype. Source files are never rewritten.

Upstream sanitization handles the Qwen tensor-name mapping, zero-centered RMSNorm offsets, convolution-axis layout, and vision patch layout. The same original checkpoint layout triggers the RMSNorm offset in the pinned Ollama implementation. MTP assets remain in the export; the research loader excludes MTP tensors and uses no drafter. Vision weights are loaded and preserved, but vision behavior has not been evaluated by this adapter pilot.

The adapter checks all 128 expected output projections and requires NVFP4 modules with global-scale handling. It rejects a different attention schedule, architecture, missing tensors, escaping shard paths, unsupported packing, and insufficient configured memory headroom.

## Numerical checks

A small independent scalar reference decodes FP4 nibbles, E4M3FN block-scale bytes, and the ordinary global multiplier. `probe_nvfp4_blob` reads one group of 16 weights from the original store and compares it with MLX dequantization and the installed scaled projection. It reads the header and 13 payload bytes, not a whole weight matrix.

On the first group of `model.language_model.layers.0.mlp.down_proj.weight`, the local probe observed:

- Maximum decoded-value absolute error: `4.656612873077393e-10`.
- Projection absolute error: `9.313225746154785e-10`.
- Declared absolute tolerance for this group: `2e-6`.

This validates that sampled group's packing and scaling. It does not prove numerical parity for all tensors, all kernels, or complete generations across engines.

## Local 27B pilot

After the no-op import integrity and short-serving-speed checks passed, the exact export was loaded with MLX 0.32.4 and MLX-VLM 0.7.12 on the documented M5 Pro. Ollama and the other model server were unloaded first. A separate watchdog terminated the process if its observed resident high-water mark exceeded 34 GiB; the adapter used a 32 GiB MLX memory guideline and 1 GiB allocator-cache limit.

The final pilot observed:

| Measurement | Result |
|---|---:|
| Strict load | 2.52 seconds |
| Editable global-scaled NVFP4 projections | 128 |
| Nine-token prompt prefill | 0.96 seconds |
| Last-position logits | `[1, 248320]`, all finite |
| Repeated prompt with fresh state | Maximum absolute logit difference 0 |
| Zero-strength wrappers on all projections | Maximum absolute logit difference 0 |
| Collected last-token residuals | 65 vectors of width 5120 |
| Prompt after removing collection wrappers | Maximum absolute logit difference 0 |
| MLX peak allocation | 18,180,346,622 bytes, approximately 16.93 GiB |
| Process resident high-water mark | 14,004,322,304 bytes, approximately 13.04 GiB |

These memory counters overlap and must not be added. File-cache warmth affects load time. This short prompt does not establish long-context memory behavior or coding quality. Raw local results live in ignored `runs/mlx-loader-pilot.json`; user prompts and model outputs are not committed.

## Research interface

```python
from pathlib import Path
from mlx_abliterate.mlx_adapter import load_export

adapter = load_export(Path('outputs/stock-noop'), memory_limit=32 * 1024**3)
logits = adapter.last_logits(token_ids)
states = adapter.collect_last_hidden(token_ids)
adapter.set_edits(directions_by_projection_path, strengths=0.0)
adapter.reset_edits()
```

`last_logits` accepts one unpadded prompt of 1–4096 tokens, creates fresh KV/Gated DeltaNet state, resets positional state, and selects the last hidden position before the vocabulary projection through the upstream `logits_to_keep=1` option. It is a prefill scoring primitive, not a generation API.

`collect_last_hidden` temporarily wraps existing decoder layers, calls their original forward implementations, and materializes one float32 last-token mean per layer. Position 0 is the input embedding; positions 1–64 are unnormalized decoder outputs. Thus decoder layer `i` corresponds to position `i+1`. Wrappers are removed in `finally`, including when inference fails. Synchronizing each layer prioritizes bounded intermediate lifetime over peak throughput.

`set_edits` validates all requested wrappers before changing module slots. Base quantized modules remain immutable and are restored by identity on reset. Every inference creates fresh state, so candidate edits cannot inherit prior KV/recurrent caches. These methods assume a single caller and are not thread-safe.

## Memory and remaining limits

MLX's `set_memory_limit` is explicitly a guideline, not a hard memory cap. The adapter checks resident and active allocation counters between shard loads and after inference, and resident high-water marks during collection. A process watchdog remains appropriate for large experiments because a single kernel or allocation can exceed a budget between checks. It does not raise wired-memory limits or enable a CPU fallback.

Further work must test longer contexts, multiple prompt lengths, real data collection, nonzero edits, generation trajectories, post-export quality, and allocation stability across repeated trials. Passing the zero-edit pilot is permission to investigate those stages, not a claim that the model has been successfully abliterated.

The implementation invokes installed APIs and contains original orchestration and scalar test code. No upstream implementation has been copied. Relevant pinned sources and their licenses remain recorded in [SOURCES.md](../research/SOURCES.md).

## End-to-end collection and drift smoke

The real `collect` and `pilot` CLI commands subsequently passed on a tiny, entirely benign fixture contrasting formal and casual explanations. The direction split had one example per group, and the search split had one distinct question; final and capability splits were distinct and unused. This checks plumbing, not useful direction estimation or model quality.

Collection took 6.51 seconds including startup/load, with 2.73 seconds recorded for statistics. The three-trial pilot took 6.40 seconds including startup/load. Strengths `0`, `0.05`, and `0.1` produced first-token `KL(base || candidate)` values `0`, `0.00234954`, and `0.00776785`. Individual trials took approximately 0.38, 0.25, and 0.25 seconds; these single-prompt scoring rates must not be extrapolated to generation or refusal optimization.

MLX peak allocation was 18,257,958,244 bytes (about 17.00 GiB). Active allocations after the three trials were 17,920,087,198, 17,920,087,250, and 17,920,087,198 bytes, respectively. This short run showed an allocation plateau; it does not replace the planned longer repeated-trial test. Summary records are in ignored `runs/workflow-smoke.json`.

## Experiment identity and integrity

Collection and pilot entry points now stream-verify all indexed tensor payloads, shapes, dtypes, declared assets (including tokenizer/config files), and runtime metadata before loading. They reject edited exports as direction-extraction or pilot bases. The recorded content fingerprint includes actual tensor and asset hashes, the source manifest digest, create options, and Modelfile bytes. This prevents an ancestral manifest digest from making a changed or edited checkpoint appear identical during resume. Undeclared top-level files are rejected because they could alter tokenizer or model loading.

A full verification of the local no-op artifact checked 2,017 tensors and eight assets in 6.43 seconds with a 4 MiB copy buffer. This verification is repeated at collection/pilot admission; the directory must remain immutable until the run finishes. The earlier mechanical pilot predates this additional identity field, so its direction artifact intentionally cannot resume under the strengthened guard without recollection. Guard integration was checked with tiny fixtures and actual full-artifact verification, without repeating the large-model pilot.
