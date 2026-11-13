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
