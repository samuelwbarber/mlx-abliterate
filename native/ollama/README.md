# Native NVFP4 direction support

The local NVFP4 merge rounded away most of the selected correction: the dynamically edited candidate scored 3/100 Heretic keyword refusals, but its merged Ollama checkpoint scored 96/100. This extension preserves the 2,017 packed base tensors and stores 128 FP32 direction vectors separately (2.5 MiB). The runner applies each rank-one correction to the corresponding attention or MLP output. It does not expand the model to BF16.

The custom architecture `Qwen3_5DirectionalForConditionalGeneration` deliberately requires this runtime. Ordinary Ollama cannot load it. Zero-strength edits return the original output directly. The patch validates direction shape, norm, strength, target paths and tensor pairing; external export/import checks bind all base and direction payloads to their receipts.

## Rebuild

Upstream is MIT licensed; its license is included in this directory. The exact source revision, patch hash, build command and numerical tests are recorded in [provenance-v8.json](provenance-v8.json). The tested Go toolchain is 1.26.11; Go 1.26.10 or newer is required by this upstream revision.

```sh
git clone https://github.com/ollama/ollama.git ollama-directional
cd ollama-directional
git checkout d63b702609c4300a2ca07be2bdb0adb9834725e5
git apply /path/to/mlx-abliterate/native/ollama/directional-v8.patch
```

On the development Mac, the matching Ollama 0.40.16 native libraries were copied from `/Applications/Ollama.app/Contents/Resources` into `build/lib/ollama`, preserving library symlinks and the `mlx_metal_v3`/`mlx_metal_v4` directory layouts. The exact file sizes and hashes are recorded in [payload-manifest.json](payload-manifest.json). This reuses the installed native payload; it does not rebuild Metal or change the installed app. Other installations must supply compatible native libraries using upstream's build instructions.

```sh
go test ./mlxrunner ./mlxrunner/model/qwen3_5 ./mlxrunner/nn ./create -run 'TestDepth|TestAcceptance|TestCost|TestSpeculationMaximum|TestDirectional|TestValidateUnitDirection' -count=1
go build -ldflags '-X github.com/ollama/ollama/version.Version=0.40.16-directional-v8' -o ollama-directional-v8 .
```

The installed development runtime lives under `outputs/native-runtime-v8/{bin,lib/ollama}` with its own verified copy of the matching native libraries. Its isolated model store is `outputs/native-model-store-v1`, served only on loopback port 11435. The original app, port 11434 and stock model remain unchanged. Binaries, native libraries, model weights and direction vectors are not distributed by this repository.

## Export and validate

`mlx-abliterate native-export --help` documents the streaming exporter. It accepts a successful selected edit and its completed search report; use `--zero-strength` for the control artifact. It copies base shards without decoding and atomically publishes a new destination. `mlx-abliterate-native-check --help` exposes import, parity, quality and speed gates. The standard benchmark remains `mlx_abliterate.heretic_benchmark` with `--base-url http://127.0.0.1:11435`.

See [validation contracts](../../docs/NATIVE_VALIDATION.md). Run GPU checks sequentially and unload each model before loading the next.

## M5 optimization

Version 8 combines the FP32 direction reduction, correction, intermediate output-dtype rounding and residual addition in one Metal kernel per edited projection. Kernel dispatch and descriptor creation use one Go-to-C call. Main decoder layers retain their original packed projections; the small correction is applied at the residual boundary. Ordinary stock models and the MTP head keep their original path. A missing Metal backend fails explicitly.

The numerical tests compare FP32 and BF16 outputs against the unfused expression for 1, 2, 9 and 32 rows of width 5120, including residual addition. Zero strength bypasses the correction. The v1 and v5 patches remain as historical implementations. Apply only the v8 patch to the pinned upstream revision.

The nonzero edited model also uses four acceptance observations to explore speculative depths and caps drafts at nine tokens. The ordinary cost/acceptance controller remains adaptive within that bound and still verifies every proposed token with the target model. Stock and zero edits retain their upstream ten-observation warm-up and original draft limit. This profile was chosen after measured rejection overhead; see the complete speed experiment history. The abandoned draft-seed correction is absent from version 8.
