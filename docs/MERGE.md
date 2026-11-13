# Bounded NVFP4 merge

`mlx_abliterate.merge` implements a matrix-at-a-time writer for an already validated no-op export. It is tested on small matrices and synthetic checkpoints. It has **not** merged or validated a full edited 27B checkpoint. Coding/tool retention, effective refusal behavior and serving speed remain separate gates.

## Input contract

`merge_export(source_export, destination, edit_spec, scratch_limit_bytes=1024**3)` requires the exported Safetensors directory and its `.mlx-abliterate/` receipts. The output must be a new directory outside the input export. An edit specification has this structure:

```json
{
  "schema_version": 1,
  "source_manifest_digest": "sha256:<digest from the source receipt>",
  "scale_policy": "absmax",
  "force_requantize": false,
  "projections": {
    "model.language_model.layers.0.mlp.down_proj": {
      "direction": [0.6, 0.8],
      "strength": 0.5
    }
  }
}
```

The two-element vector illustrates syntax only. Each real direction must match that projection's output width (5,120 for the intended model). The implementation normalizes finite nonzero vectors. Strength is bounded to `[0, 2]`. Only explicitly listed language-tower `mlp.down_proj`, `self_attn.o_proj` and `linear_attn.out_proj` modules in layers 0–63 are eligible, and all three NVFP4 companion tensors must exist. Vision and MTP projections are excluded. Already modified exports are rejected to avoid silently chaining edits against the original base fingerprint.

Zero strength copies original bytes and requires no MLX import, unless `force_requantize` is explicitly true. That option forces the decode/re-quantize path for a quantization-only control, including zero-strength listed projections. It never implicitly selects the other projections.

## Numerical semantics

Stored Ollama NVFP4 weights represent:

```text
W = decode_fp4_and_block_scales(packed, scales) * stored_global_factor
W_edited = W - strength * r * (r.T @ W)
```

Weight arithmetic and direction reductions use float32. Ordinary additive layer biases are not changed.

A crucial convention: **native MLX's `global_scale` argument is a tensor absmax, not Ollama's stored multiplicative factor**. The conversion is `native_absmax = stored_global_factor * 448 * 6`, or `2688 * stored_global_factor`. This matches the conversion used by the installed MLX-VLM scaled switch-linear implementation. Passing the stored scalar directly to native dequantization produces a factor-of-2,688 error.

The merge implementation decodes without a native global argument, then multiplies the stored scalar. It quantizes with the explicitly converted native absmax. Policies are:

- `absmax`: choose a new factor `max(abs(W_edited)) / 2688`. For an all-zero matrix use factor 1.
- `preserve`: retain the old factor. The report flags `scale_range_exceeded` if the edited matrix exceeds the representable global range. This policy can clip and requires evaluation.

Neither policy claims NVIDIA ModelOpt calibration equivalence. Re-quantization introduces error even if the directional formula is exact. Reports always retain `quality_validated: false`.

## Memory and I/O

The writer processes sorted tensor names and materializes at most one selected bundle at a time. It writes that small packed input bundle to scratch, loads it with MLX, computes and saves the modified bundle, copies its three payloads into the output, then deletes scratch and releases arrays/cache. Unchanged tensors stream through the same 4 MiB byte-copy helper as the no-op exporter. Each unchanged payload and each selected input payload is checked against the source receipt hash.

Before dense arithmetic, admission estimates two float32 weight matrices, two packed-weight/scale sets and 32 MiB overhead. The default scratch limit is 1 GiB. This estimate is a guard, **not a measured process-peak guarantee**; allocator/kernel behavior and actual export telemetry must still be measured. Large matrices fail admission rather than silently switching to an unbounded allocation. Input-column tiling is future work if a selected matrix cannot fit.

Output shard headers retain dtype and shape. Selected tensors are written in the same ModelOpt-compatible interchange naming as the no-op export. Config/tokenizer assets and runtime options are copied. New receipts contain payload hashes, the explicit edit specification, original model fingerprint, source-receipt hash, per-projection quantization information and unvalidated status. No model is imported automatically.

Publication uses an exclusive atomic directory rename: macOS `renamex_np(RENAME_EXCL)`, Linux `renameat2(RENAME_NOREPLACE)`, or Windows' non-overwriting rename. This prevents a concurrent empty destination from being replaced. Platforms without the required primitive fail explicitly. A failed merge removes only its own staging directory.

## Tests

Nine focused tests currently cover:

- Exact zero-strength handle identity and byte-preserving zero-edit export.
- Nonzero merge matching an independently written explicit matrix update followed by native quantization.
- Dynamic output correction versus the unquantized matrix-update reference, including ordinary bias; the strict mathematical check uses CPU float32 matmul to avoid conflating GPU matmul precision with the edit formula.
- Measured quantized-output drift on a deterministic small example (not a whole-model quality claim).
- Forced quantization-only control and the 2,688 global-scale convention.
- Scratch admission, invalid block scales and invalid global factors.
- Source fingerprint mismatch, ineligible projection rejection, corrupt-source detection and cleanup.
- An untouched norm payload remaining byte-identical while the selected projection changes.

The broader no-op exporter tests include exclusive publication under a concurrent destination. A real edited 27B merge must follow the dynamic-candidate quality gate, then use `verify_import` and the separate evaluation protocol after import.

## Real-matrix component measurement

One local component run used the stock layer-0 MLP down projection (5,120 × 17,408), a deterministic synthetic direction and strength 0.01. It completed decoding, editing and native re-quantization in 0.118 seconds after loading the packed bundle. MLX peak active allocation was 763,346,952 bytes, with a 50,135,044-byte starting allocation; the peak increase was 713,211,908 bytes, within the 1 GiB admission limit. The estimate was 846,856,192 bytes. The allocator cache cap was 256 MiB.

This is a single numerical/memory component measurement, excluding disk loading, serialization and whole-model inference. It is not an optimization-run ETA or evidence of useful behavioral changes. No modified weights were retained. Raw local evidence is in the ignored `runs/merge-component.json`; CPU RSS and MLX counters describe overlapping/different allocations and must not be added.
