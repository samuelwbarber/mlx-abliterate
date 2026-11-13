# Source register

Reviewed 13 November 2026. Prefer the pinned code links for implementation findings. Project READMEs and papers provide context; benchmark claims retain their original hardware/model scope. No upstream source code is included in this repository.

## S1 Refusal direction research

- [Arditi et al., Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717), 2024.
- [Accompanying code and results](https://github.com/andyrdt/refusal_direction).

Use: mathematical motivation for directional intervention. Does not establish effectiveness or capability retention for Qwen3.8-27B, this quantization, or this machine.

## S2 Upstream Heretic

Inspected revision: `a14b8543537b50ce8db4db7eaae4eb295d5d5142`. Declared source license: AGPL-3.0-or-later.

- [Repository README at inspected revision](https://github.com/p-e-w/heretic/blob/a14b8543537b50ce8db4db7eaae4eb295d5d5142/README.md).
- [Model loading, adapters, reset, and merge](https://github.com/p-e-w/heretic/blob/a14b8543537b50ce8db4db7eaae4eb295d5d5142/src/heretic/model.py).
- [Directional ablation modifier](https://github.com/p-e-w/heretic/blob/a14b8543537b50ce8db4db7eaae4eb295d5d5142/src/heretic/modifiers/abliteration.py).
- [ARA modifier](https://github.com/p-e-w/heretic/blob/a14b8543537b50ce8db4db7eaae4eb295d5d5142/src/heretic/modifiers/ara.py).
- [First-token KL scorer](https://github.com/p-e-w/heretic/blob/a14b8543537b50ce8db4db7eaae4eb295d5d5142/src/heretic/scorers/kl_divergence.py).
- [Default configuration](https://github.com/p-e-w/heretic/blob/a14b8543537b50ce8db4db7eaae4eb295d5d5142/config.default.toml).
- [License](https://github.com/p-e-w/heretic/blob/a14b8543537b50ce8db4db7eaae4eb295d5d5142/LICENSE).

Use: distinguish current adapter-based behavior from older full-weight implementations; identify the full-precision CPU merge in the bitsandbytes branch. CPU memory and GPU memory share the same physical pool on the target Mac.

## S3 Inspected MLX Heretic port

Accessible repository: `akaiHuang/mlx-heretic`; inspected revision `27f369eaac86c3708a66b109f93c5d46c45e82dd`. Its license file declares Apache-2.0 with a Hey-Meadow copyright notice. This record is not a conclusion about the provenance of every line.

- [README](https://github.com/akaiHuang/mlx-heretic/blob/27f369eaac86c3708a66b109f93c5d46c45e82dd/README.md).
- [Weight snapshot and direct matrix edit](https://github.com/akaiHuang/mlx-heretic/blob/27f369eaac86c3708a66b109f93c5d46c45e82dd/mlx_heretic/ablation.py).
- [Hidden-state and logits collection](https://github.com/akaiHuang/mlx-heretic/blob/27f369eaac86c3708a66b109f93c5d46c45e82dd/mlx_heretic/model_io.py).
- [Prompt subsets and search loop](https://github.com/akaiHuang/mlx-heretic/blob/27f369eaac86c3708a66b109f93c5d46c45e82dd/mlx_heretic/run.py).
- [KL and refusal evaluation](https://github.com/akaiHuang/mlx-heretic/blob/27f369eaac86c3708a66b109f93c5d46c45e82dd/mlx_heretic/evaluator.py).
- [License file](https://github.com/akaiHuang/mlx-heretic/blob/27f369eaac86c3708a66b109f93c5d46c45e82dd/LICENSE).

Use: code-level audit of quantized-weight assumptions, overlapping split prefixes, and swallowed generation errors. The advertised small-model timing is not a 27B estimate.

## S4 MLX and Apple hardware

- [MLX quantized matrix multiplication](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html).
- [MLX dequantization](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html).
- [MLX memory limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html).
- [MLX cache limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html).
- [MLX quantized layers, v0.32.4](https://github.com/ml-explore/mlx/blob/v0.32.4/python/mlx/nn/layers/quantized.py).
- [Apple research: MLX and M5 GPU Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).

The installed MLX 0.32.4 docstrings and quantized-layer source were also inspected locally. `mx.device_info()` supplied the hardware-memory values in the evidence file. M5 GPU Neural Accelerators are not the same thing as the separate Apple Neural Engine; no ANE execution is assumed in this plan.

Use: supported primitives and memory controls. Their availability does not establish format equivalence to an Ollama checkpoint or its inference throughput.

## S5 Ollama importer and runtime

Inspected tag `v0.40.16`, commit `d63b702609c4300a2ca07be2bdb0adb9834725e5`.

- [Source classification and re-quantization rejection](https://github.com/ollama/ollama/blob/d63b702609c4300a2ca07be2bdb0adb9834725e5/create/classify.go).
- [Prequantized producer layouts, scales, and global-scale handling](https://github.com/ollama/ollama/blob/d63b702609c4300a2ca07be2bdb0adb9834725e5/create/prequant.go).
- [Qwen hybrid quantization policy](https://github.com/ollama/ollama/blob/d63b702609c4300a2ca07be2bdb0adb9834725e5/create/qwen35.go).
- [Quantization format definitions](https://github.com/ollama/ollama/blob/d63b702609c4300a2ca07be2bdb0adb9834725e5/mlx/quant/quant.go).
- [Linear-layer loading](https://github.com/ollama/ollama/blob/d63b702609c4300a2ca07be2bdb0adb9834725e5/mlxrunner/model/linear.go).
- [Draft-model import](https://github.com/ollama/ollama/blob/d63b702609c4300a2ca07be2bdb0adb9834725e5/create/draft.go).
- [Parser and local model-file handling](https://github.com/ollama/ollama/blob/d63b702609c4300a2ca07be2bdb0adb9834725e5/parser/parser.go).
- [Official model import documentation](https://docs.ollama.com/import).
- [Official Modelfile reference](https://docs.ollama.com/modelfile).

Use: export/import constraints. The fact that a pattern is recognized by the importer is not proof that our eventual exported artifact will load or run correctly; that is Stage 1's purpose.

## S6 Ollama performance design

- [Ollama MLX performance and caching update](https://ollama.com/blog/mlx-performance), published 11 June 2026.

Use: context about fused execution, GPU sampling, NVFP4 support, and conversation-state caching. No percentage from this article is used to explain the locally measured 5.3× gap.

## S7 Adjacent MLX tooling

- [dexloom/mlx_fun](https://github.com/dexloom/mlx_fun).

README reviewed for adjacent quantization/steering/export work. No implementation audit or performance validation was performed for this project. Pin and inspect relevant conversion code and its licensing before reuse; do not treat README examples as tested 27B/Ollama support.

## S8 Limits of a single-direction explanation

- [Joad et al., There Is More to Refusal in Large Language Models than a Single Direction, v2](https://arxiv.org/abs/2602.02132v2), revised 15 September 2026.

Use: motivation to measure diverse behavior categories and distinguish refusal types. Does not prescribe a particular higher-rank implementation for this target.

## S9 Existing checkpoint metadata and local measurements

- [Official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B).
- [RANA MLX model card](https://huggingface.co/abliterant/Qwen3.8-27B-RANA-abliterated-MLX).
- [Hardware and source inventory](evidence/environment.json).
- [Earlier same-prompt comparison](evidence/prior-comparison.json).
- [Pinned source revisions](evidence/source-revisions.json).

Local files are the evidence for the installed artifact and observed throughput. Model cards describe related releases and cannot prove this checkpoint's quality, exact provenance equivalence, or performance. RANA is retained in the record only as the earlier slower deployment comparison, not as the proposed base for a new ablation search.
