# Using the current research tools

This is an experimental, single-architecture workflow for the inspected dense 64-layer Qwen hybrid model. Inventory, export and benchmark commands use Python's standard library. Collection, drift measurement and edit primitives need Apple Silicon and the pinned MLX runtime. The current CLI does not automatically judge refusal behavior, execute coding benchmarks, select a useful edited model, or establish capability retention.

## Install and inspect

Use a project environment so the existing serving installation remains separate:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/mlx-abliterate inspect qwen3.8:27b-mlx --output runs/inventory.json
.venv/bin/mlx-abliterate plan-export qwen3.8:27b-mlx --output runs/export-plan.json
```

`inspect` reads manifests and tensor headers without loading model weights. Add `--verify-payloads` to stream hashes over the complete blobs. `--models-dir` selects a nondefault Ollama store. The stock store is read-only.

The unchanged export is a separate artifact:

```sh
.venv/bin/mlx-abliterate export qwen3.8:27b-mlx outputs/unchanged --output runs/export.json
```

Use the generated import instructions/Modelfile to create a **new** Ollama tag, then check that imported tag with `verify-import`. Do not use the stock tag as the destination. Preserve the export receipt under `.mlx-abliterate/`; collection and pilot provenance refer to it. An unchanged export must pass the format and serving gates before a real edit study.

## Serving measurements

Provide a local JSON array of nonblank prompt strings. Several distinct prompts should complete comfortably within the output budget. The file and generated text stay outside Git.

```sh
.venv/bin/mlx-abliterate baseline qwen3.8:27b-mlx \
  --prompts datasets/serving-prompts.json --repeats 5 --warmups 1 \
  --num-ctx 4096 --num-predict 512 --cold --output runs/stock.json
```

Run the same command for the imported tag with a different output filename. Ensure only one large model is resident. `--cold` unloads only the named model before its first request; it does not stop other services. `--environment` accepts a JSON object recording observed power, thermal and MTP settings. No undocumented MTP toggle is used.

```sh
.venv/bin/mlx-abliterate compare runs/stock.json runs/unchanged.json --output runs/comparison.json
```

The comparison requires matching settings, prompt hashes, prompt token counts, Ollama version and environment, at least two distinct prompts, and at least five valid warm samples per prompt. Truncated, empty or failed responses cannot pass the gate. A separate summary reports throughput including truncated responses for performance diagnosis only. Neither rate demonstrates correctness. Exit code `2` means an incomplete baseline or a comparison that did not pass; exit code `1` means a command error.

Reports contain prompt/response hashes and timing metadata, not the actual generated text. Reasoning-versus-final token counts, MTP acceptance, memory and cache-hit status are not supplied by this harness. Prefix reuse can affect warm requests. Repeat at the required contexts and use the full [evaluation protocol](EVALUATION.md) for release decisions.

## Dataset format and provenance

Collection and pilot take one user-supplied UTF-8 JSONL file. Each row has:

| Field | Requirement |
|---|---|
| `id` | Unique nonblank string |
| `prompt` | Nonblank string, preserved for actual tokenization |
| `split` | One of `direction`, `search`, `final`, `capability` |
| `group` | Required for direction rows; a user-defined contrast-group name |
| `source`, `license`, `revision` | Optional nonblank provenance strings; supply them for reproducible studies |

All four splits must be nonempty; the direction split needs at least two named groups. Supply actual data and its provenance rather than treating fixture examples as an evaluation set. Group names have no built-in meaning: a label is not evidence that an example elicits refusal.

Normalized duplicate prompts are rejected both within and across splits. Normalization is Unicode NFKC plus whitespace collapse, preserving case. This catches textual duplicates, not paraphrases or semantic leakage. The dataset lock records raw and normalized hashes, row order, split/group membership and supplied provenance. Any change to those records changes the fingerprint. Limits are 16 MiB per file and 64 KiB per JSONL row.

`DatasetSplits.splits` exposes immutable local records for callbacks; `.lock()` returns metadata/hashes without prompt text. Source/license fields are caller-provided declarations, not verified permissions.

## Collect directions and measure drift

Install the optional runtime into the project environment:

```sh
.venv/bin/python -m pip install -e '.[runtime]'
.venv/bin/mlx-abliterate collect outputs/unchanged datasets/study.jsonl outputs/directions \
  --positive-group positive --negative-group negative --output runs/collection.json
.venv/bin/mlx-abliterate pilot outputs/unchanged datasets/study.jsonl outputs/directions \
  --strengths 0 0.05 0.1 --max-prompts 8 --output runs/pilot.json
```

Replace the contrast-group arguments with names present in your dataset. Collection computes normalized positive-minus-negative last-token residual means. Residual position zero is the embedding and is excluded; positions 1–64 map to decoder layers 0–63. It keeps per-layer vectors, not every prompt's activation history. The model's ordinary forward implementation is used with bounded-memory checks.

Collection and pilot stream-verify every exported tensor payload and declared asset before loading. They reject modified exports and fingerprint actual content, runtime options and the Modelfile; an ancestral model digest alone is insufficient. You can run this check independently with `mlx-abliterate verify-export outputs/unchanged --require-unmodified`. Keep the verified input directory immutable throughout inference. Direction artifacts include this export-content fingerprint, source manifest digest, dataset lock/fingerprint, runtime versions and a tensor-file hash. Pilot rejects mismatched provenance and malformed vectors before loading the model. Output directories and report filenames must be new; existing artifacts are never intentionally overwritten. For collection, `directions.json` is published last and marks completion. A directory without it is incomplete.

Pilot always starts with strength zero and requires at least three strengths. Zero-strength logits must match the baseline exactly; any mismatch stops the pilot before nonzero trials and restores the base modules. Strengths are finite values within `[0, 2]`; defaults are small engineering probes, not tuned recommendations. It uses at most 64 search-split prompts, each at most 4,096 tokenized tokens, with thinking disabled. Baseline last-token logits are spooled to temporary files; each inference uses fresh model state.

**Pilot measures first-token `KL(base || candidate)` only.** It does not generate responses, score refusals, run coding/tool tasks, choose a candidate, or validate a merged model. The timing forecast covers similar drift trials only; it is not an estimate for a full generation-based optimization run.

## Python callback studies

`mlx_abliterate.search.run_candidates` supplies sequential orchestration without a model-specific generation judge:

```python
report = run_candidates(
    run_dir,
    candidates,        # JSON dictionaries with unique nonblank `id` fields
    provenance,        # nonempty model/dataset/runtime/edit_schema entries
    evaluate=evaluate_candidate,
    reset_state=discard_all_model_state,
    cancelled=should_cancel,
)
```

`evaluate_candidate(candidate)` returns a dictionary with `status`; for a completed evaluation it must also include a finite lower-is-better `score` and explicit boolean `constraints_met`. Optional `failures` is a nonnegative count, and `metrics` contains finite numeric values. Recognized statuses are `completed`, `runtime_error`, `invalid`, `empty`, `truncated`, `malformed`, and `cancelled`. Any failure, noncompleted status or failed constraint removes the candidate from selection and gives it no eligible score. Raw callback text and exception messages are not written into trial records.

The callback owns the rubric and must count generation errors honestly. `completed` means the callback evaluation completed; it is not automatically a substantive answer or a coding success. A zero-edit baseline is an ordinary candidate and can remain the best result. No keyword-based automatic success label is provided.

`reset_state()` runs before and after every attempt, including exceptions. Its contract is to discard **all** KV, recurrent and prefix caches and restore the immutable model/edit state; merely resetting a random seed is insufficient. A reset failure stops the study and prevents resume until investigated in a new run.

A process lock prevents simultaneous workers. The study manifest fixes provenance and candidate order. Resume refuses any mismatch, skips existing terminal outcomes, and never overwrites a trial. Cancellation before a trial leaves it pending; interruption during a trial writes a cancelled attempt, then resume appends a fresh attempt. Checkpoint files contain parameters and metrics, so keep run directories outside Git. This is a fixed-candidate runner, not yet an Optuna optimizer or an end-to-end 27B behavioral search.

## Merge an evaluated candidate

After separately evaluating a dynamic candidate, provide an explicit edit specification as described in [MERGE.md](MERGE.md):

```sh
.venv/bin/mlx-abliterate merge outputs/unchanged outputs/edited \
  --edit-spec runs/accepted-edit.json --output runs/merge.json
```

This writes a new artifact and does not change a serving default or import automatically. A zero-strength selected projection stays byte-identical unless the specification explicitly requests the quantization-only control. Validate the new artifact with `verify-export`, import into a new tag, then run `verify-import` and the post-export evaluation. A successful merge is not a quality approval.

See [DESIGN.md](DESIGN.md) for numerical assumptions and [EVALUATION.md](EVALUATION.md) for the still-required capability, behavior and post-export checks.
