# Build plan

**Decision proposed:** build a narrow MLX research tool around the exact installed Qwen3.8-27B NVFP4 model, with immutable quantized weights during search and a separately tested Ollama export path. Start with directional ablation; add more elaborate methods only when measured results justify them.

**Document status:** original staged plan, retained as the acceptance framework. Implementation is now underway; see [measured results](RESULTS.md) and [usage](USAGE.md) for what exists and which checks have passed. Criteria below are targets unless supported by those result records.

## Scope

The first release targets one dense hybrid architecture, one 48 GB Apple Silicon machine class, local text-based search/evaluation, and export into a new Ollama tag. Preserve the stock checkpoint, its tokenizer/template, unedited vision assets, and MTP assets. Avoid model fine-tuning, cloud infrastructure, a web UI, broad architecture claims, and custom GPU kernels in the MVP.

The useful deliverable is a model that still codes and uses tools well, not merely a lower refusal counter. The [evaluation protocol](EVALUATION.md) defines distinct quality, behavior, memory, and speed criteria.

## Milestones and gates

| Stage | Deliverable | Pass criteria | If it fails |
|---|---|---|---|
| 0. Inventory and baseline | Read-only model inspector; reproducible baseline harness | Exact manifest/tensor/config fingerprints; documented warm/cold and MTP settings; repeatable stock measurements | Resolve model provenance or benchmark inconsistency before editing |
| 1. Unchanged-model round trip | Supported interchange artifact and new Ollama tag, without ablation | All logical tensor values/scales preserved; templates and MTP present; no-op tag within 10% of stock median warm decode at matched settings | Investigate importer/layout fidelity; do not continue to a large search |
| 2. MLX fidelity and tiny edit reference | Loader plus Qwen adapter; reversible rank-one module | Zero edit preserves outputs; packed-weight semantics verified; nonzero edit agrees with explicitly modified tiny matrices under dtype-specific tolerances | Fix layout, masks, layer mapping, or math; no silent full-model dequantization |
| 3. Bounded 27B execution | Streaming statistics, baseline distributions, one timed candidate | One loaded model; peak below 34 GiB process target; stable allocations after warm-up; no sustained pressure or unexplained fallback | Reduce batch/context/cache limits or revisit loader; publish the failed feasibility result |
| 4. Small search | 10–20 candidate study with disjoint data and resume | All outcomes classified; one selected candidate meets preliminary quality constraints; pilot-derived ETA | Broaden or change method only after diagnosing the failure, not by chasing the test set |
| 5. Edited export | Matrix-at-a-time NVFP4 writer and new model tag | Full integrity verification; dynamic-vs-export drift quantified; quantization-only control measured; no-op and rollback tests pass | Keep adapter as a research artifact; do not claim fast deployable output |
| 6. Final comparison | Capability report and matched Ollama benchmark | Prespecified behavior/quality gates; at least 90% of stock median warm decode at each tested context; useful MTP behavior measured | Report regression and retain stock as default |

Stages 1 and 2 share format work but have different responsibilities: Ollama's no-op import can pass while the Python MLX representation remains wrong. Both must pass.

## First implementation slice

Build Stage 0 and the smallest Stage 1 experiment first:

1. Inventory the exact current tag, resolve nested manifests, identify quantized and nonquantized tensors, and record MTP and template metadata.
2. Produce a dry-run memory/disk estimate and a list of unsupported layouts. No model mutation or large download is necessary for inventory.
3. Read one representative NVFP4 projection and validate packing, scale bytes, global scale, and logical shape against the pinned Ollama loader.
4. Establish an unchanged export/import route under a disposable tag. Retain original payload values and demonstrate that Ollama still selects its MLX runner.
5. Re-run a small fixed coding prompt suite with stock and no-op tags, one loaded at a time. Separate first request, warm generation, and prefix reuse.
6. Write a short go/no-go report before implementing direction search.

This slice tests the project's largest product risk: losing the original deployment properties while producing a technically valid checkpoint.

## Method development sequence

Start with toy matrices and a small quantized model, then the exact 27B checkpoint. Testing a small model verifies implementation mechanics; it does not establish 27B performance or effectiveness.

The first modifier uses a direction estimated from contrasting prompt groups, optional orthogonalization against a benign mean, and bounded per-component strength profiles. Keep its zero-edit control mandatory. Predeclare search bounds and seeds. Do not copy an arbitrary aggressive default from another model.

Search uses a cheap screening tier and full evaluation of finalists. Screening cannot label missing or truncated outputs as successes. Preserve all trial results and Pareto tradeoffs rather than optimizing only a refusal percentage. Start with no speculative decoding during search.

Only consider magnitude-preserving edits or ARA after the simple method's failure mode is identified. Row normalization changes the mathematical operator and export behavior; it deserves a separate test suite and source/provenance decision.

## Engineering decisions

| Decision | Reason |
|---|---|
| Python 3.11 and a project-local environment | Matches the currently installed working MLX stack; no global package changes |
| Pin MLX/model-runtime revisions after the fidelity spike | The known installation is MLX 0.32.4 and `mlx-vlm` 0.7.12; these are observations, not final dependency choices |
| Reuse validated model kernels and forward logic | Avoid a second implementation of Gated DeltaNet or attention |
| Implement original bounded-memory orchestration | The existing port is not proven for quantized 27B models |
| Use Optuna only after the single-candidate path works | Optimization should not obscure a loader or evaluation bug |
| Keep inference through Ollama after export | Preserve the user's working workflow; research engine speed is not the deployment target |
| Default to sequential GPU trials | Avoid duplicate weights and cache state on a 48 GB machine |
| Keep license/provenance decision before code reuse | Heretic and its MLX port have different declared licenses; no upstream code is currently included |

The CLI now implements `inspect`, `baseline`, `collect`, `pilot`, unchanged `export`, selected-projection `merge`, and integrity/comparison commands. `search` remains a Python callback runner; generation-based evaluation and automatic candidate selection are not implemented. See [usage](USAGE.md) for current commands.

## Expected artifacts

- A source lock containing model digests, configs, tokenizer/template hashes, package versions, and hardware details without device identifiers.
- A data lock containing licensed dataset sources, immutable revisions, normalized-example hashes, and disjoint splits.
- Small direction/edit artifacts linked to their exact base model.
- A resumable study and per-trial timing/memory/failure records.
- A no-op artifact, quantization-only control, and selected edited export.
- A report that distinguishes research throughput, cold latency, warm decode, cache reuse, capability scores, and refusal behavior.

Large artifacts and prompts remain outside Git. Model tags and output paths are new; stock weights are never overwritten. Public incremental source commits are authorized; this does not imply publication of model weights or private datasets.

## Effort and runtime estimates

These are planning allowances, not promises:

- Inventory and no-op compatibility investigation: roughly **1–2 focused engineering days**.
- If that gate passes, a bounded 27B pilot and small search prototype: roughly **3–6 additional days**.
- Reliable export, regression evaluation, documentation, and failure recovery: roughly **2–5 additional days**.

Format incompatibility or a required new model adapter could expand those ranges. Do not schedule an overnight optimization run until the single-trial pilot has produced an ETA and memory trace. The [runtime calculation](EVALUATION.md#runtime-estimation) replaces the earlier conversational guesses.

## Main risks and planned responses

| Risk | Response |
|---|---|
| Exact Ollama artifact cannot be faithfully represented in Python MLX | Stop at the fidelity gate; investigate format conversion or an identical upstream artifact with explicit provenance |
| Export unexpectedly changes precision or selects another runner | Reject the result; inspect per-tensor metadata and importer behavior |
| Ablation search loses coding/tool reliability | Use real execution/trajectory checks and Pareto constraints; retain zero edit as a valid best result |
| MTP acceptance falls after editing | Measure acceptance and end-to-end speed; preserve original drafter for comparison, rather than promising retained acceleration |
| Search scores improve because generation failed or ended early | Track invalid outputs separately and penalize them |
| Lazy graphs or cached tensors grow across trials | Warm-up plateau test, per-trial telemetry, explicit lifetime boundaries, checkpoint-and-exit on budget breach |
| Changed-weight re-quantization creates drift | Quantization-only control and post-export re-evaluation |
| Search overfits prompts or a refusal keyword list | Disjoint held-out splits, richer response labels, manual adjudication of borderline cases |
| A large run gives a misleading ETA | Report phase-specific rates and recompute after representative trials |

## Review point

Stages 0–2 have initial local evidence, including unchanged-model import fidelity, short serving measurements, and the exact-source MLX pilot. The current next gate is **Stage 3 repeated-trial stability and longer-context bounded execution**. The three-trial mechanical smoke does not establish that gate. Stage 4 still requires disjoint study data and an explicit behavioral/capability evaluator; export components do not establish Stage 5 quality. See [measured results](RESULTS.md) for the limits of the completed checks.
