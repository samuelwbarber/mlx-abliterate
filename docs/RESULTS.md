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

## What is not yet established

No held-out refusal reduction, coding retention, tool-use retention, edited-model serving speed, or optimal edit strength is claimed. A successful format round trip and small numerical tests do not answer those questions. Use disjoint user-supplied data and the evaluation protocol before selecting or deploying an edited checkpoint.

Raw timing fields and content hashes, without prompts or responses, are in [the evidence directory](../research/evidence/).
