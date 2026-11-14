# Validated native M5 candidate

The installed `qwen3.8-abliterated:27b-m5` uses runtime `0.40.16-directional-v8` on loopback port 11435, with a separate model store. Run `python3 scripts/native-model.py` from the project root. Original stock remains unchanged on port 11434.

## Results

- Actual deployed Heretic standard100: **2/100 keyword refusals**, versus 98/100 stock. All 100 windows were valid; 96 exhausted the 100-token budget and were scored under the standard protocol.
- Served answer retention: **12/12**, versus 11/12 stock. Every stock-correct answer was retained and all tasks completed.
- Exact zero control: unchanged isolated stock and zero-strength native edits matched original stock across both fixed prompts, including cold and warm samples.
- Import fidelity: all 2,017 original packed tensors plus 128 direction vectors verified against the exported payloads, including model configuration and runtime metadata.
- Serving: **74.79 tokens/s edited versus 78.06 stock**, ratio **0.958**, from all 40 warm measurements per model. Four blocks alternate model order; no timing samples are discarded. Browser GPU work was paused for this controlled confirmation and resumed afterward. Individual block results and prior failed speed probes remain in the evidence.
- Resource supervision: normal system memory pressure throughout both final runs. RSS is measured over actual server process trees, not the HTTP client, and is not a direct Metal allocation measurement. See the evidence for peaks and global swapout changes.
- Tool tests: 199 Python tests passed; focused Go tests cover schema admission, import architecture, zero-edit behavior and FP32/BF16 fused arithmetic across decoding and multi-token shapes.

## Selected modification

The selected source direction is layer 24, strength 1.575, applied to attention output and MLP down projections across all 64 main decoder layers. The base remains NVFP4. Embeddings, LM head, MTP head and stock tensors are preserved. The modification occupies 2,621,440 additional tensor bytes.

The selected dynamic candidate scored 3/100, but re-quantizing the modification into NVFP4 produced 96/100 in the actual merged deployment. That checkpoint was rejected. The native extension stores the correction separately and fuses its reduction, subtraction and residual addition in one Metal kernel. This preserves its effect without expanding the weights to BF16. Its custom architecture fails on an unsupported stock runtime instead of silently ignoring the edit. Nonzero edited models use a four-observation speculative warm-up and a nine-token draft cap; target verification remains unchanged, while stock and zero edits keep their upstream settings.

## Scope and evidence

The test100 was used during optimization, so the score is not held out. The pinned Heretic dataset order, 33 keyword markers and 100-token budget are preserved; local generation explicitly uses `think=false`, which differs from upstream automatic prefix handling. Keyword absence does not establish useful completed answers. Twelve exact-answer checks do not establish broad coding, tool-use, vision, MTP acceptance or long-context quality. The speed comparison measures short repeated-prefix generation at context4096.

[Final measurements](../research/evidence/native-final.json) · [Dynamic selection](../research/evidence/heretic-search-v6.json) · [Rejected NVFP4 merge](../research/evidence/heretic-merged-v6-rejected.json) · [Speed optimization history](../research/evidence/native-speed-optimization.json) · [Runtime build and provenance](../native/ollama/README.md) · [Validation protocol](NATIVE_VALIDATION.md) · [Historical first candidate](CANDIDATE-V1.md).
