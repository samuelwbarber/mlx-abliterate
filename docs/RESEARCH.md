# Research findings

Reviewed 13 November 2026. References below resolve through the [source register](../research/SOURCES.md). Code-level conclusions apply to the pinned revisions, not every project version.

## Conclusions that determine the design

1. A full BF16 27B model needs roughly 54–56 GB for weights alone. This machine has 48 GiB of physical unified memory; its currently reported recommended GPU working set is about 37.44 GiB. The search must operate on quantized weights, with explicit working-memory limits.
2. The original model is already in the desired NVFP4 deployment format and contains 15 MTP tensors. Its actual tensor headers include packed U32 weights, U8 block scales, and F32 global scales. An exact, read-only import of this artifact is preferable to silently substituting another quantization.
3. Current Heretic already uses adapters rather than directly rewriting the full model during every trial. The proposed contribution is an MLX implementation and bounded-memory export for this model size, not the invention of adapter-based ablation.
4. The small MLX port inspected here is useful reference material but does not establish quantized 27B support. Its implementation assumes ordinary floating-point projection weights.
5. Ollama 0.40.16 supports several prequantized layouts, but explicitly rejects `--quantize` on an already-quantized source. A 6-bit-to-NVFP4 command is not an export design.
6. Export fidelity and serving performance must be tested before investing in a large search. A successful refusal score is insufficient if the resulting model becomes slow or loses reliable tool use.

## Existing approaches

| Approach | What the inspected evidence supports | Relevance to this project |
|---|---|---|
| Arditi et al. and accompanying code [S1] | Direction extraction and interventions on residual-stream-writing weights; results across multiple earlier model families | Mathematical baseline, not proof of Qwen3.8 retention |
| Heretic [S2] | PyTorch/Transformers, pluggable ARA and directional modifiers, Optuna search, adapter-based edits, optional bitsandbytes loading | Reference search/evaluation design; not a demonstrated MLX/Ollama path |
| `akaiHuang/mlx-heretic` [S3] | Small-model MLX experiment with Qwen hybrid-layer handling and per-layer directional edits | Learn from it, but audit and redesign memory handling and evaluation |
| MLX core and neural-network primitives [S4] | Packed quantized matmul, dequantization, NVFP4 support, memory/caching controls | Numerical and execution foundation |
| Ollama 0.40.16 [S5] | Existing fast serving path and importer source available for inspection | Deployment target and no-op control |
| `dexloom/mlx_fun` [S7] | Adjacent quantization, steering, and streaming-conversion work, with a broader MoE focus | Review before implementing conversion helpers; not established as this project's dense-27B solution |

## Heretic code audit

At revision `a14b8543537b50ce8db4db7eaae4eb295d5d5142`, `model.py:apply_lora` attaches adapters to discovered projection modules and resets adapter weights between trials. Its optional quantized load path uses bitsandbytes NF4. Neither that backend nor a CUDA-oriented workflow should be assumed to work efficiently on this Mac.

Its quantized-model merge path reloads the base on the CPU in full precision before merging. On unified memory this does not remove the total-memory problem; streaming export must be designed separately. The directional modifier includes projected directions and optional row-magnitude preservation. The first-token KL scorer measures distribution drift, not successful coding or tool execution. [S2]

The main config inspected has 100 trials and a 100-token response limit. Those defaults and the upstream small-model timing example do not justify an ETA for a 27B Mac run. Earlier conversational estimates of several hours or overnight were unmeasured planning guesses; replace them with a timed pilot.

## MLX port code audit

At revision `27f369eaac86c3708a66b109f93c5d46c45e82dd`:

- `ablation.py` snapshots selected projection weights and later constructs modified weight arrays. It does not implement a quantized adapter path. Its cast of a direction to `mod.weight.dtype` and direct multiplication by `mod.weight` assume floating-point logical weights; packed integer storage must not be treated that way.
- `model_io.py` constructs a list of per-layer hidden states and computes sequence logits before selecting the final position. Lazy evaluation may eliminate some unused work, so the physical peak needs profiling; the API still does not enforce a bounded streaming collection strategy.
- `run.py` takes direction and evaluation examples from overlapping prefixes of the same input lists. A separate held-out evaluation is needed.
- `evaluator.py:count_refusals` catches generation exceptions and replaces them with an empty string. Empty output can evade its refusal-string detector. Errors must be a separate failed outcome, not a better refusal score.
- The KL implementation computes baseline-to-candidate KL, although its explanatory docstring is inconsistent about direction. Specify and test the equation explicitly.

These are observations about the pinned code, not a claim that every fork has the same behavior. The README's approximately eleven-minute run concerns a 2B checkpoint and fifteen short trials, not 27B. Its linked `Hey-Meadow` repository returned 404 during this review; record the accessible fork and revision rather than treating the link as verified provenance. [S3]

## Quantization and the exact installed checkpoint

The local manifest has a wrapper/provenance relationship to another manifest, so a reader must handle a manifest graph, deduplicate tensors by digest, and distinguish model payloads from metadata. It must not assume every nested manifest is a drafter. MTP tensors in this installation have explicit `mtp.*` names.

The text config declares 64 layers, hidden width 5,120, vocabulary 248,320, and full attention every fourth layer. Other layers use Gated DeltaNet. The architecture name is `Qwen3_5ForConditionalGeneration` despite the Qwen3.8 product name. Dispatch must use inspected config and validated tensor layouts, not string matching on the tag.

MLX and Ollama both support NVFP4-related operations, but that does not prove their packing, scale conventions, mixed precision, or input-quantization behavior match. Scalar global scales, FP8 scale-byte interpretation, matrix orientation, and any transformations applied by the Ollama loader are explicit parity-test targets. [S4, S5]

Ollama's `create/classify.go` refuses re-quantization of prequantized sources. `create/prequant.go` recognizes MLX, compressed-tensors NVFP4, and ModelOpt layouts. `create/qwen35.go` preserves sensitive low-rank linear-attention projections at source precision during fresh quantization. Do not globally convert every tensor to four bits or discard per-tensor policy. [S5]

## What the prior speed comparison establishes

The [recorded comparison](../research/evidence/prior-comparison.json) used one coding prompt, temperature zero, thinking disabled, and 256 generated tokens. Only one model was loaded at a time.

| Configuration | Observed decode |
|---|---:|
| Existing Ollama NVFP4 model, first request | 48.99 tokens/s |
| Same model, second request | 65.17 tokens/s |
| RANA 6-bit, `mlx-vlm` with MTP | 12.35 tokens/s |

This is evidence of a deployment gap, not a controlled experiment isolating quantization, kernel compilation, MTP, or weight editing. The models and engines differ. Preserve this limitation wherever the numbers are reused. Ollama documents fused execution and conversation-state caching; those are reasons to investigate its serving path, not a numerical attribution of this gap. [S6]

## Scientific and product risks

A single extracted direction is a useful baseline, not a universal account of refusal. Later research distinguishes different refusal-related directions and tradeoffs. Multi-direction or higher-rank methods should follow measured failures of the simple method, not be assumed necessary from the outset. [S8]

Reduced refusal can coincide with empty output, incoherence, lost reasoning, fabricated tool results, or reduced willingness to acknowledge uncertainty. Evaluation needs separate labels for these cases. A low KL score, especially only at the first answer token, cannot establish retention on long coding-agent trajectories.

Quantizing a modified already-quantized weight introduces another approximation. Compare against a control that re-quantizes the same selected matrices without ablation. Otherwise quantization damage can be misattributed to the edit.

## Questions left for implementation spikes

1. Can the exact installed NVFP4 checkpoint be read and represented in Python MLX without altering its effective weights or materializing a full BF16 copy?
2. Can a no-op export/import preserve Ollama's supported format, templates, MTP behavior, and warm speed?
3. Does an output-projection wrapper remain efficient under this hybrid model's compiled execution path?
4. Is simple projected directional ablation adequate, or do retained capabilities require row normalization or a higher-rank method?
5. Does keeping the original MTP head preserve useful draft acceptance after editing the target?
6. What are the actual peak memory and per-trial runtime with short prompt batches on this Mac?

These questions are milestone gates, not implementation details to defer until after an overnight run.
