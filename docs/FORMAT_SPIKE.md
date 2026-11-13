# Unchanged-model format spike

The first implementation provides read-only inventory and a bounded-memory **no-op interchange exporter**. It does not edit model weights or claim that an exported checkpoint retains performance until an actual import and benchmark pass.

## Observed installed source

The inspected `qwen3.8:27b-mlx` root manifest contains 1,199 tensor layers: 409 NVFP4 bundles and 790 ordinary floating-point tensors. Bundles expand to 2,017 individual payloads. Eight JSON assets carry config, tokenizer and vision metadata. License and parameter layers and the model config carry runtime settings. The legacy manifest has a nested parent/provenance reference. The current Ollama v2 tag points directly at that content-addressed manifest. The resolver now follows v2 first, matching the pinned runtime, and validates the symlink target digest; legacy lookup is a fallback. Neither reference is treated as a separate drafter. MTP tensors are preserved alongside all other tensors.

The dry-run plan produces 17 shards and approximately 18.174 GB of payload and assets. The nominal shard target is 1 GiB; an individual tensor larger than that remains intact in an oversized shard. Memory used to copy payloads is bounded to a 4 MiB buffer, independently of shard size. Metadata remains resident and scales with the tensor count. Disk planning adds a small metadata allowance and rejects insufficient free space.

## Mapping supported by the pinned importer

Source: Ollama [prequant.go at d63b702](https://github.com/ollama/ollama/blob/d63b702609c4300a2ca07be2bdb0adb9834725e5/create/prequant.go). This project contains an original implementation, not copied upstream code.

| Installed bundle tensor | Exported ModelOpt-style name | Payload treatment |
|---|---|---|
| `projection.weight`, U32 | `projection.weight` | Exact byte copy; packed shape unchanged |
| `projection.weight.scale`, U8 | `projection.weight_scale` | Exact byte copy; scale-byte dtype unchanged |
| `projection.weight.global_scale`, scalar F32 | `projection.weight_scale_2` | Exact byte copy; no reciprocal |
| Ordinary BF16/F16/F32 tensor | Original name | Exact byte copy |

The importer recognizes the ModelOpt naming with U32 packed weights. It fuses scale companions into its internal bundle names. Its scalar-F32 transformation preserves the scalar type; it does not invert ModelOpt global scales. This is distinct from the compressed-tensors path, which applies a reciprocal. Unsupported layouts fail before export instead of being guessed.

Config and tokenizer JSON bytes are copied unchanged, including existing quantization metadata. The emitted Modelfile uses `FROM ./` and explicitly carries renderer, parser, version requirement, capabilities, license, template/system where present, and model parameters. No `-q` option should be supplied during import. The exporter does not replace the stock tag or edit Ollama's blob store.

Receipts live under `.mlx-abliterate/` within the export directory so Ollama's top-level JSON discovery cannot accidentally ingest them as model assets. The Safetensors index is generated; the pinned importer excludes it from model JSON assets.

## Verification sequence

1. `plan_export(model)` reads metadata and reports size without writing weight files.
2. `export_model(model, destination)` verifies all source blob hashes before streaming. It writes a temporary sibling directory, records every tensor payload's SHA256, fsyncs shards, and renames only after completion. It refuses an existing destination. Failed writes remove the incomplete staging directory.
3. Import with `ollama create NEW_TAG -f /absolute/export/Modelfile` through the supported importer. Use a separate new tag.
4. `verify_import(NEW_TAG, export_directory)` reads every imported tensor payload and compares its hash, dtype, and shape against the receipt. It also compares JSON asset hashes and runtime metadata. Whole tensor-blob hashes may differ because bundle headers change; that alone does not imply changed weights.
5. Run the separate serving and behavioral benchmark. Payload fidelity does not establish speed, actual MTP use, or decoding behavior.

The exported receipt says `validated_import: false` deliberately. Verification returns a separate result rather than retroactively converting an export operation into a successful deployment claim.

## Tests and limits

Fixture tests cover exact NVFP4 bytes and names, scalar scale preservation, multiple shards, interruption cleanup, overwrite refusal, model-store destination refusal, and detection of a changed imported payload. Inventory regressions cover v2 precedence, digest verification, public host aliases, dangling-link fallback, and rejecting manifest symlinks outside the blob directory. The real-source dry run passes. A local import into `mlx-abliterate-noop:27b` has also passed complete verification of all 2,017 tensor payload hashes, dtypes and shapes, all JSON asset hashes, and runtime metadata. Serving and behavioral gates remain separate. An initial import exposed Ollama’s inclusion of all top-level JSON files; the receipts were moved into a subdirectory and the disposable tag was reimported before this verification.

The exporter is currently deliberately narrow: NVFP4 group 16 with scalar global scale, ordinary floating-point tensors, and the observed manifest metadata. Affine MLX quantization, other packed layouts, external draft layers, and arbitrary model-family conversion are rejected. It never dequantizes or calibrates weights. This is the no-op gate, not the edited NVFP4 writer or a complete abliteration engine.
