# Experimental text inference

`generate` runs deterministic greedy text decoding on a verified unchanged or edited export. `evaluate-nll` scores a prescribed continuation token by token. Both use the installed model forward implementation with fresh KV/recurrent caches and positional state, and preserve the caller's selected weights. They do not execute generated code, select a model, or establish capability retention.

Use the project runtime environment with the pinned MLX dependencies. Unload other large models before inference. Memory checks run between forward calls; a separate process watchdog is still needed to enforce an external memory or elapsed-time budget.

```sh
mlx-abliterate generate outputs/stock-noop \
  --prompt runs/prompt.txt --max-new-tokens 64 --output runs/generation.json
mlx-abliterate evaluate-nll outputs/stock-noop \
  --prompt runs/prompt.txt --target runs/continuation.txt --output runs/likelihood.json
```

Prompt and continuation files are local UTF-8 text, each nonblank and at most 64 KiB. The prompt uses the export's chat template with thinking disabled. The continuation is tokenized as plain text without added special tokens. Prompt plus output budget is limited to 4,096 tokens; output or continuation is limited to 1,024 tokens. EOS IDs come from the tokenizer and the verified generation configuration. No speculative drafter is used.

Every export's tensor and asset integrity is checked before loading. Edited exports are accepted for inference; this verification is not a quality approval. Existing report files are never overwritten. Reports require an explicit local `--output` path. Generation reports include raw decoded text and output token IDs; keep them in ignored local run directories. They contain prompt hashes, not raw prompt text. Continuation reports contain per-token and mean negative log likelihood, with a target hash.

Generation outcomes distinguish `completed` (EOS), `empty` (EOS without substantive decoded text), `truncated` (token budget exhausted), and `runtime_error` (explicit error type). Truncated outputs never become completed merely because they look complete. Runtime errors retain partial generated token IDs but cannot produce a valid likelihood score. CLI exit code 0 means completed, 2 means another recorded outcome, and 1 means admission or command failure. Completion alone does not establish correctness or non-refusal.

The Python functions `generate(adapter, token_ids, eos_token_ids=[...])` and `teacher_forced_nll(adapter, token_ids, target_ids)` can evaluate caller-installed dynamic edits. They do not reset those edits, but discard inference caches after every call. Require a `completed` result before using `mean_nll`. Teacher-forced likelihood is a diagnostic proxy, not measured behavioral effectiveness.

## Checkpoint comparison evidence

The separate checkpoint evaluator loads one model per process and evaluates the final split plus every capability row against the local rubric file:

```sh
python -m mlx_abliterate.checkpoint_evaluation outputs/edited \
  runs/dataset.jsonl runs/rubrics.json \
  --output runs/edited-evaluation.json --logits-dir runs/edited-logits
```

For dynamic-versus-export comparisons, run the same command against the unchanged source with `--edit-spec runs/selected-edit.json`; dynamic edits require a verified unchanged base. Run these processes sequentially so only one model is resident. All runs should use identical dataset, rubrics, and output budgets (default 96 tokens). Dynamic evaluation cannot simulate forced requantization: load the separately merged quantization-only control for that comparison.

Reports contain result hashes and scores, runtime, memory, source identity, dataset/rubric fingerprints, and the exact edit-file hash where applicable. They contain no prompt or generated response text. Optional `.npy` files hold one float32 last-token logit distribution per evaluated prompt; keep them out of Git. The logits directory must be new. Its `manifest.json`, written last, identifies prompt/token hashes, row order, file hashes, shapes, and evaluation provenance. A directory without the manifest is incomplete.

Edit specification files are bounded at 16 MiB across merge, merge validation, and checkpoint evaluation, allowing the 128 width-5,120 projection vectors of the supported architecture. Checkpoint completion means inference completed, not that answer checks or a release criterion passed; inspect the per-task results and comparison gates.
