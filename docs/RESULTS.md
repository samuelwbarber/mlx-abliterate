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

A separate run at the adapter's **4,096-token limit** also passed with collection, one warm-up and three measured cycles. Its live allocation spread was zero, MLX peak was 24,299,738,174 bytes (22.63 GiB), and process peak RSS was 16,729,702,400 bytes (15.58 GiB). Collection took 12.38 seconds; each measured cycle (three edited prefills and one restored prefill) took about 48–49 seconds. Total supervised time was 239.92 seconds. Pressure stayed at level 1 and swapouts did not increase. See [4,096-token evidence](../research/evidence/stability-4096.json).

These runs support proceeding from the bounded synthetic prefill/collection gate to a small study. They do not establish long-generation stability or make the first-token drift score a behavioral/capability evaluator.

## First experimental edited checkpoint

A 13-candidate study selected strength 1.0 on layers 16–47, preserving every baseline-correct search and held-out answer. The dynamic candidate improved the small held-out answer count from 7/8 to 8/8 and increased refusal-continuation NLL from 1.1042 to 2.6788. These are a narrow authored diagnostic and a likelihood proxy, not observed refusal-rate reduction.

The serialized checkpoint retains all 11/12 baseline-correct answers in the expanded final-plus-capability check. It loses the dynamic candidate's extra correct answer (12/12 before serialization). Its refusal-continuation NLL is 2.5504. Quantization-only control logits/results match stock exactly. Dynamic-to-serialized first-token KL is mean 0.1229, maximum 0.8731 across 16 rows; this drift is reported, not treated as zero.

The separate Ollama tag `qwen3.8-abliterated:27b-mlx` passed 2,017-tensor import verification. Ten completed warm serving samples per model yielded medians 75.06 tokens/s stock and 76.65 edited, ratio 1.0212. Fixed stock-then-edited order and short prompts preclude claims of speed improvement or long-context performance. See [candidate details](CANDIDATE.md), [study evidence](../research/evidence/study-v1.json), and [checkpoint evidence](../research/evidence/checkpoint-comparison-v1.json).

## What is not yet established

No measured behavioral refusal-rate reduction, broad coding/tool-use retention, long-context edited-model serving speed, or optimal edit strength is claimed. A successful format round trip and small numerical tests do not answer those questions. Use disjoint user-supplied data and the evaluation protocol before selecting or deploying an edited checkpoint.

Raw timing fields and content hashes, without prompts or responses, are in [the evidence directory](../research/evidence/).
