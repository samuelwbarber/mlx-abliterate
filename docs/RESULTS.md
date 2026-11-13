# Local implementation results

Measured on 2026-11-13, M5 Pro / 20 GPU cores / 48 GB, Ollama 0.40.16, MLX 0.32.4, MLX-VLM 0.7.12. These are engineering checks, not an abliteration effectiveness or coding benchmark.

## Unchanged export and import

The installed model was exported through a bounded 4 MiB copy buffer to ModelOpt-compatible Safetensors, then imported through `ollama create` as a separate `mlx-abliterate-noop:27b` tag. No source tensors were decoded or edited.

All **2,017 tensor payloads**, their shapes and dtypes, JSON asset hashes, and runtime metadata matched after import. This includes packed weights, block scales, global scales, and retained MTP assets. Source model blobs were treated as read-only. The current Ollama v2 manifest resolves to `sha256:5642e97495e1a088883805981563dcdc4a040c2f53388b7a41d1f24d3622cf7e`; the earlier legacy manifest also included a provenance layer.

The importer reads every top-level JSON file. Export receipts therefore live in `.mlx-abliterate/` so they do not become model configuration assets.

## Short serving test

Two short coding prompts, five warm repetitions each, temperature zero, seed 42, thinking disabled, context limit 4,096, output cap 256. Cold and warm-up samples are recorded separately. RANA was stopped and only one Ollama model was loaded at a time.

| Model | Completed warm requests | Median decode tokens/s |
|---|---:|---:|
| Stock | 10 / 10 | 67.68 |
| Unchanged round trip | 10 / 10 | 70.76 |

All ten paired response hashes and generated token counts matched. The speed ratio was 1.045, passing the initial 90% threshold. This does **not** establish a speed improvement: the run order was stock then candidate, prompts were short, and power/thermal state and MTP acceptance were not independently measured. No generated code was executed. Long contexts, cache-miss behavior, tools, vision, and the full capability suite remain separate tests.

## Python MLX loader

The strict loader retained 128 NVFP4 output projections with their global scales. It uses the installed model implementation rather than a copied transformer loop. MTP assets remain in the export but are excluded from the search runtime.

The first pilot loaded in 2.24 seconds and processed a 9-token prompt in 0.75 seconds. Last-position logits had shape `[1, 248320]`, were finite, and two calls with fresh caches matched exactly. Peak MLX allocation was 18,171,024,126 bytes (16.92 GiB); process peak RSS was 15,745,499,136 bytes (14.66 GiB). These overlapping counters must not be added. This short pilot does not establish the peak for long prompts or an entire optimization study.

An independent scalar decoder checked a representative stock NVFP4 group against MLX: maximum decode error `4.66e-10`, projection error `9.31e-10`. Tiny matrix tests also cover ordinary biases, affine quantization offsets, zero-strength identity, and CPU/GPU rounding differences.

## End-to-end collection and drift pilot

The actual CLI completed `collect` then `pilot` on the 27B model with two benign contrasting style prompts and one search prompt. This is a mechanical smoke test, not a refusal dataset. Collection took 6.51 seconds including startup/load; the three-trial pilot took 6.40 seconds including startup/load. Zero strength produced KL exactly zero. Strengths 0.05 and 0.1 produced first-token KL values approximately 0.00235 and 0.00777.

MLX peak allocation was 18,257,958,244 bytes (17.00 GiB). Active allocations stayed approximately 17.92 GB across the three trials. This is too short a run to establish a long-study memory plateau or predict generation-heavy search time.

## Largest-projection merge component

One real 5,120 × 17,408 down-projection was merged with a synthetic direction and strength 0.01, then re-quantized. Arithmetic took 0.118 seconds excluding input loading. Additional peak MLX allocation was 713,211,908 bytes (680 MiB), below the 1 GiB scratch target. No resulting weights were retained, and no edited full checkpoint was imported. This validates a component's memory behavior, not edited-model quality.

## Repeated prefill and collection stability

With RANA stopped and Ollama empty, the 27B model passed synthetic 32-, 256-, and 1,024-token validation. Each length had one warm-up and five measured cycles, with strengths 0, 0.05 and 0.1 followed by restoration. Zero-edit and restored logits matched exactly. All 65 collected residual vectors were finite and had width 5,120; collection left baseline logits unchanged.

Post-reset live MLX allocations had **zero measured spread** across the five cycles at each length. Peak MLX allocation was 20,337,631,316 bytes (18.94 GiB), and process peak RSS was 18,256,232,448 bytes (17.00 GiB); these overlap. All 114 system pressure samples reported level 1, with no additional swapouts. Existing swap usage and some swapins were present; these system counters cannot be attributed solely to this process.

The supervised run took 113.75 seconds. A small unit suite ran concurrently early, so timings are not an isolated performance benchmark. Synthetic directions and repeated prefill do not establish behavioral effectiveness, generation stability, or capability retention. See [sanitized evidence](../research/evidence/stability-27b.json) and [probe usage](STABILITY.md).

## What is not yet established

No held-out refusal reduction, coding retention, tool-use retention, edited-model serving speed, or optimal edit strength is claimed. A successful format round trip and small numerical tests do not answer those questions. Use disjoint user-supplied data and the evaluation protocol before selecting or deploying an edited checkpoint.

Raw timing fields and content hashes, without prompts or responses, are in [the evidence directory](../research/evidence/).
