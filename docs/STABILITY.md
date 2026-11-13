# Bounded prefill stability validation

The standalone probe exercises the next Stage 3 memory checks with one immutable loaded model, synthetic token sequences of multiple lengths, and reversible edits. It does not select a candidate or estimate refusal reduction. Run only after other model servers have been unloaded, using an environment with the pinned MLX runtime installed:

```sh
python -m mlx_abliterate.stability --export outputs/stock-noop \
  --output runs/stability.json --lengths 32 256 1024 --cycles 5 --collect
```

`--collect` additionally checks all residual positions and vector shapes, rejects nonfinite residuals, and requires baseline logits to remain identical after collection. It records collection timing and memory before, during, and after releasing the residual vectors. The output file must not exist. The worker verifies the unchanged export's payloads and content identity before loading. Each token length gets one warm-up cycle followed by five measured cycles. Each cycle evaluates strengths 0, 0.05 and 0.1 on all 128 output projections using a deterministic synthetic direction, then restores the original modules and checks the unchanged logits. Every forward call uses the adapter's fresh KV and recurrent state. Exact zero-edit and restoration equality, finite logits, and memory-budget checks are mandatory. The direction is an engineering fixture, not a learned behavioral direction.

The prespecified allocation target is at most 16 MiB between the largest and smallest post-reset MLX live allocation across measured cycles of each token length. Allocator cache is recorded separately and is not manually flushed between trials. This target checks retained allocation growth over the finite run; it does not prove absence of every leak. RSS, active allocations, cache allocations and peak allocations overlap and must not be added.

An independent parent watches only its own worker's RSS every approximately 0.2 seconds and terminates that worker above 34 GiB or the default 600-second timeout. The loader uses a 32 GiB guideline and 1 GiB cache limit, with phase-boundary checks. These checks are not an OOM guarantee: a kernel can exceed a budget between observations. The probe does not monitor system compression or memory pressure, so a passed report alone cannot close the complete Stage 3 pressure gate.

The report contains token counts, input hashes, identity, timings, numerical differences and memory counters; no prompts, token IDs, weights or generated text are published. The workload is synthetic prefill only. Long generation trajectories, behavioral datasets, coding retention and Ollama serving remain separate validation work. Collection is checked only when `--collect` is requested.

The initial run was deferred while RANA was resident. After the user authorized stopping it, the 32/256/1,024-token run passed with collection enabled and five measured cycles per length. All 107 tests pass in the installed MLX environment. See [measured results](RESULTS.md) and [evidence](../research/evidence/stability-27b.json). RANA remains stopped at the user’s request.
