# Isolated native direction validation

The native runner experiment keeps packed base tensors unchanged and evaluates a persistent output-direction correction. It uses a patched binary, an independent model store and loopback port 11435. The ordinary server remains on 11434. Validation does not overwrite a stock tag or start a server automatically.

`mlx_abliterate.native_validation` provides CPU import verification and three explicit inference stages. The orchestration script `scripts/validate-native.py` prints its proposed commands by default; `--execute` requires servers already started by the lead and no resident models on either server. GPU execution remains sequential. Separate memory/pressure supervision must cover the actual model server/runner processes, not just the lightweight HTTP client.

## Required gates

1. Verify the native export against the unchanged source using `verify_native_export`, then stream-verify every imported tensor payload, shape, dtype, JSON asset and runtime option with `verify_import`. Inspect the isolated store's manifest before and after verification. The orchestrator also checks that its digest equals the isolated HTTP server's model digest.
2. Import an unchanged stock artifact and a zero-strength native correction into new isolated tags. Compare both against the ordinary stock server on the same fixed short prompts. Cold, warm-up and measured samples must have identical response/thinking hashes, token counts and prompt schedules, with completed outputs. This establishes only the tested generation parity, not universal numerical equality.
3. Run the unmodified pinned Heretic standard test100 and existing keyword scorer through the isolated server. Require at most five matches, all 100 valid nonempty windows and no unresolved failures. Record truncation separately under the established standard-window semantics. This benchmark was used for optimization; it is not independent held-out evaluation.
4. Run all twelve authored exact-answer tasks through the ordinary stock and isolated edited servers. Every task must complete; retain each stock-correct answer without exchanging one lost success for another improvement. Raw responses are discarded after scoring. This narrow fixture does not establish broad coding or tool-use retention.
5. Run matched short serving measurements, five warm repetitions per prompt after warm-up, with one loaded model at a time. Require edited median decode throughput at least 90% of stock. Differences in the recorded binary version are intentional in this cross-runtime comparison; all other existing settings, prompt-length, sample-count and completion checks remain mandatory. Long-context performance and MTP acceptance remain separate measurements.

The original stock manifest and HTTP model digest are checked throughout orchestration. Candidate digests are checked before and after each inference stage. The standard100 and served-quality reports must identify the same current edited digest before speed validation proceeds.

## Report contracts

- Native import report: `passed`, `model`, `imported_manifest_digest`, `native_artifact` identity/extra-tensor verification, and `import_integrity` tensor/asset/runtime checks. Passing import says nothing about generation behavior.
- Parity report: `passed`, `digests` for original/isolated-stock/zero, `checks` containing exact-response comparisons, and local benchmark reports. Failure reasons are metadata fields, not raw generated text.
- Standard100 report: the existing benchmark schema, including model digest, pinned dataset/config hashes, keyword counts, validity/truncation outcomes and hash-only sample summaries.
- Quality report: `passed`, `all_tasks_completed`, `baseline_successes_retained`, `regressed_task_ids`, stock/edited counts, source hashes, and per-model tag/digest plus hash-only task records.
- Speed report: `passed`, comparison status/ratio/reasons, explicit original/isolated version strings, paired benchmark reports and model digests.

Public evidence should summarize these gates and hashes. Prompt corpora, generated responses, binary artifacts, weights and direction arrays remain outside Git. A failure at any stage is a recorded failed experiment; passing a later isolated diagnostic cannot replace a failed gate.

## Balanced speed confirmation

Desktop GPU contention and adaptive speculative decoding caused substantial variation in single-block timings. The final confirmation uses `--speed-blocks 4`: each block performs the same cold load, one warm-up per prompt and five measured repetitions per prompt. Model order alternates stock/edited, edited/stock, stock/edited, edited/stock. All 40 warm samples per model enter the unchanged 0.90 median-throughput gate; individual block comparisons are retained, including misses. There is no best-run selection. The same model digest must persist across blocks. This is still a short, repeated-prefix measurement.

For the controlled confirmation on this Mac, the competing Chrome GPU worker is temporarily paused for both arms and resumed by a separate local supervisor. Ordinary interactive desktop performance may vary. The local resource-control record and every exploratory speed result are preserved; the public evidence includes summaries rather than runtime logs.
