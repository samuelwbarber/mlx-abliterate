# First experimental edited model

**Subsequent standard benchmark:** stock and this first edit both scored 98/100 keyword refusals on the pinned Heretic standard test100 through Ollama. This candidate does not meet the ≤5% refusal target. Its earlier likelihood proxy did not predict measured refusal reduction. A new full-standard-data search is in progress. See [benchmark evidence](../research/evidence/heretic-initial-v1.json).

The development Mac now has the separate Ollama tag:

```sh
ollama run qwen3.8-abliterated:27b-mlx
```

The stock `qwen3.8:27b-mlx` remains intact and its default was not changed. RANA was stopped at the user's request. Both benchmarked Ollama models were unloaded after testing; launching the tag loads it again. Model weights are local and are not in Git or publicly hosted.

## Edit and selection

The exact-source NVFP4 model received a rank-one directional removal at strength 1.0 on both output projections in decoder layers 16–47 (64 projections). Directions were normalized differences of last-token residual means from 16 matched high-level refusal/benign prompt pairs. This is a conventional directional-ablation operation, not fine-tuning.

A frozen 13-candidate study compared zero plus four strengths across three layer ranges. It used disjoint direction, search, final and capability examples. High-level disallowed requests were used only for activation collection or teacher forcing of harmless refusal text. Generated answers were restricted to benign exact-answer tasks. Data were locally authored and narrow; the corpus is excluded from Git.

The selected dynamic edit retained all eight search answer checks and improved the held-out exact-answer count from 7/8 to 8/8. Mean refusal-continuation NLL increased from 1.2466 to 2.4369 on search and from 1.1042 to 2.6788 on held-out prompts. This says the supplied refusal continuation became less likely. It does not establish the rate of refusal, compliance, coherence on open-ended requests, or broad coding quality.

## Export and serving

The edited artifact is `outputs/abliterated-v1`; the matched quantization-only control is `outputs/quantization-control-v1`. The selected edit specification remains local in `runs/study-v1/selected-edit.json`. Export used the existing global scales and preserved unedited tensors, tokenizer/template, vision and MTP assets. All 2,017 tensor payloads, assets and runtime metadata verified after Ollama import.

| Check | Stock | Dynamic edit | Serialized edit | Quantization control |
|---|---:|---:|---:|---:|
| Final plus all capability exact answers | 11/12 | 12/12 | 11/12 | 11/12 |
| Held-out refusal-continuation mean NLL | 1.1042 | 2.6788 | 2.5504 | 1.1042 |

The serialized edit retains each stock-correct answer but loses the dynamic candidate's extra success. Some capability examples in this expanded comparison were used in search; the 12-task set is not wholly held out. Dynamic-versus-serialized first-token KL across 16 rows has mean 0.1229, median 0.0209 and maximum 0.8731. Quantization-control logits and result records match stock exactly. The nonzero export drift is a practical limit, not a failed byte-copy claim: changed weights were re-quantized.

A direct Ollama run also scored 11/12 for both stock and edited, preserving every stock-correct answer without failed or truncated responses. No generated code was executed.

Two short coding prompts with five warm repeats each measured median decode rates of 75.06 tokens/s stock and 76.65 edited, a 1.0212 ratio, above the prespecified 0.90 threshold. Stock ran first; thermal/power state and MTP acceptance were not independently isolated. This supports the short serving gate, not a speed-improvement claim or long-context release approval.

## Tool and remaining limits

The toolkit now provides inventory, unchanged export, integrity checks, reversible collection/editing, bounded generation and likelihood scoring, resumable gated studies, matrix-bounded merging, matched quantization controls, checkpoint drift comparison and Ollama serving benchmarks. The full suite has 145 passing tests in the installed MLX environment.

This is an experimental ablation checkpoint suitable for local trials. Broad refusal-rate reduction, real coding execution, multi-step tool trajectories, long-generation stability, vision behavior and long-context serving still require evaluation. A likelihood-proxy improvement and twelve exact-answer questions do not establish those properties. Keep stock available for comparison.

Evidence: [study](../research/evidence/study-v1.json), [checkpoint comparison](../research/evidence/checkpoint-comparison-v1.json), [Ollama answer checks](../research/evidence/ollama-quality-v1.json), [serving](../research/evidence/edited-serving-v1.json).
