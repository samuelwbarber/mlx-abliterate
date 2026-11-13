"""Matrix-at-a-time directional NVFP4 merge for validated interchange exports.

Native MLX's global_scale parameter is an absmax, while Ollama's stored scalar
is a multiplicative factor. The conversion is explicitly 448 * 6 = 2688.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile

from .export import ExportError, _copy_range, _dump, _shards, _publish_directory, verify_export
from .inventory import read_safetensors_header

GIB = 1024 ** 3
NVFP4_MAX = 2688.0
MAX_EDIT_SPEC_BYTES = 16 * 1024**2
_ELIGIBLE = re.compile(r'model\.language_model\.layers\.(\d+)\.(mlp\.down_proj|self_attn\.o_proj|linear_attn\.out_proj)\Z')


def read_edit_spec(path):
    """Bounded JSON admission sized for all 128 width-5120 projections."""
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_EDIT_SPEC_BYTES + 1)
    if len(raw) > MAX_EDIT_SPEC_BYTES:
        raise ValueError('Edit specification exceeds 16 MiB')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _mlx():
    try:
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        raise RuntimeError('MLX is required for NVFP4 merge on Apple Silicon') from exc
    return mx


def _validate_strength(strength):
    if isinstance(strength, bool) or not isinstance(strength, (float, int)) or not math.isfinite(strength) or not 0 <= strength <= 2:
        raise ExportError('Strength must be finite and in [0, 2]')
    return float(strength)


def _validate_scratch_budget(value):
    if type(value) is not int or value <= 0:
        raise ExportError('Matrix scratch budget must be a positive integer byte count')


def merge_nvfp4(weight, scales, global_scale, direction, strength: float, *,
                scale_policy: str = 'absmax', force_requantize: bool = False,
                scratch_limit_bytes: int = GIB):
    """Return packed weight, block scales, scalar factor and a small report.

    Zero strength preserves original handles unless force_requantize=True.
    Nonzero edits use float32 weight arithmetic and normalize the direction.
    The scratch admission estimate covers two dense FP32 matrices, packed
    arrays and 32 MiB of overhead; it is not a measured process-memory bound.
    """
    mx = _mlx()
    from .edits import normalize_direction
    _validate_scratch_budget(scratch_limit_bytes)
    strength = _validate_strength(strength)
    if scale_policy not in {'preserve', 'absmax'}:
        raise ExportError('Scale policy must be preserve or absmax')
    if type(force_requantize) is not bool:
        raise ExportError('force_requantize must be boolean')
    if weight.dtype != mx.uint32 or weight.ndim != 2 or weight.shape[1] % 2:
        raise ExportError('Expected NVFP4 U32 matrix with groups of 16')
    if scales.dtype != mx.uint8 or scales.shape != (weight.shape[0], weight.shape[1] // 2):
        raise ExportError('Unexpected NVFP4 block-scale layout')
    if global_scale.dtype != mx.float32 or global_scale.shape != ():
        raise ExportError('Expected scalar float32 global factor')
    factor = global_scale.item()
    if not math.isfinite(factor) or factor <= 0:
        raise ExportError('Global factor must be finite and positive')
    # E4M3FN max exponent/fraction is a NaN; negative scale bytes are invalid.
    if not bool(mx.all(scales < 127).item()):
        raise ExportError('Invalid NVFP4 block-scale bytes')
    if strength == 0 and not force_requantize:
        return weight, scales, global_scale, {'changed': False, 'requantized': False,
                                             'scale_policy': scale_policy, 'strength': 0.0}
    elements = weight.shape[0] * weight.shape[1] * 8
    estimate = 2 * elements * 4 + 2 * (weight.nbytes + scales.nbytes) + 32 * 1024 ** 2
    if estimate > scratch_limit_bytes:
        raise ExportError(f'Matrix scratch estimate {estimate} exceeds budget {scratch_limit_bytes}')
    direction = normalize_direction(direction) if strength else None
    if direction is not None and direction.shape != (weight.shape[0],):
        raise ExportError('Direction width must match projection output dimension')
    # Deliberately multiply after decoding: the stored factor is NOT the native
    # global_scale argument. This also avoids a factor*2688/2688 round trip.
    decoded = mx.dequantize(weight, scales, group_size=16, bits=4, mode='nvfp4', dtype=mx.float32) * global_scale
    mx.eval(decoded)
    if strength:
        edited = decoded - strength * direction[:, None] * (direction @ decoded)[None, :]
        mx.eval(edited)
        del decoded
    else:
        edited = decoded
    if not bool(mx.all(mx.isfinite(edited)).item()):
        raise ExportError('Nonfinite edited weight')
    maximum = mx.max(mx.abs(edited)).item()
    if scale_policy == 'absmax':
        new_global = mx.array(maximum / NVFP4_MAX if maximum else 1.0, dtype=mx.float32)
    else:
        new_global = global_scale
    native_absmax = new_global * NVFP4_MAX
    if not math.isfinite(native_absmax.item()) or native_absmax.item() <= 0:
        raise ExportError('Native NVFP4 scale underflow or overflow')
    packed, blocks = mx.quantize(edited, group_size=16, bits=4, mode='nvfp4', global_scale=native_absmax)
    mx.eval(packed, blocks, new_global)
    report = {'changed': True, 'requantized': True, 'quantization_only': strength == 0,
              'strength': strength, 'scale_policy': scale_policy, 'old_global_factor': factor,
              'new_global_factor': new_global.item(), 'estimated_scratch_bytes': estimate,
              'edited_max_abs': maximum, 'scale_range_exceeded': maximum > native_absmax.item(),
              'quality_validated': False}
    return packed, blocks, new_global, report


def _source_entries(source):
    receipt_path = source / '.mlx-abliterate/export-report.json'
    receipt = json.loads(receipt_path.read_text())
    index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    records = {r['name']: r for r in receipt['tensors']}
    if set(index) != set(records):
        raise ExportError('Source receipt and shard index disagree')
    headers, bases = {}, {}
    for filename in set(index.values()):
        path = (source / filename).resolve()
        if path.parent != source or not path.is_file():
            raise ExportError('Unsafe or missing source shard')
        headers[filename] = read_safetensors_header(path)
        with path.open('rb') as stream:
            bases[filename] = 8 + struct.unpack('<Q', stream.read(8))[0]
    if {k for h in headers.values() for k in h if k != '__metadata__'} != set(index):
        raise ExportError('Source has unindexed tensors')
    entries = []
    for name in sorted(index):
        filename = index[name]
        tensor = headers[filename][name]
        old = records[name]
        lo, hi = tensor['data_offsets']
        if any(tensor[key] != old[key] for key in ('dtype', 'shape')) or hi - lo != old['size']:
            raise ExportError('Source tensor metadata differs from receipt')
        entries.append({**old, 'path': source / filename, 'offset': bases[filename] + lo})
    return receipt, entries


def _write_bundle(path, entries):
    header, offset = {}, 0
    for entry in entries:
        header[entry['name']] = {'dtype': entry['dtype'], 'shape': entry['shape'], 'data_offsets': [offset, offset + entry['size']]}
        offset += entry['size']
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    with path.open('xb') as target:
        target.write(struct.pack('<Q', len(encoded)))
        target.write(encoded)
        for entry in entries:
            with entry['path'].open('rb') as source:
                sha = _copy_range(source, target, entry['offset'], entry['size'])
            if sha != entry['payload_sha256']:
                raise ExportError('Source tensor hash mismatch')


def _validate_spec(spec, receipt, entries):
    if not isinstance(spec, dict):
        raise ExportError('Edit specification must be an object')
    if spec.get('schema_version') != 1 or spec.get('source_manifest_digest') != receipt['source_manifest_digest']:
        raise ExportError('Edit schema or base-model fingerprint mismatch')
    if receipt.get('modified'):
        raise ExportError('Chained edited sources are not supported; use the original no-op export')
    policy = spec.get('scale_policy')
    if policy not in {'preserve', 'absmax'}:
        raise ExportError('Edit specification must explicitly choose scale_policy')
    force = spec.get('force_requantize', False)
    if type(force) is not bool:
        raise ExportError('force_requantize must be boolean')
    edits = spec.get('projections')
    if not isinstance(edits, dict):
        raise ExportError('Explicit projection map is required')
    by_name = {entry['name']: entry for entry in entries}
    for prefix, edit in edits.items():
        match = _ELIGIBLE.fullmatch(prefix)
        if not match or not 0 <= int(match.group(1)) < 64:
            raise ExportError('Only explicit language-tower output projections are editable')
        if not isinstance(edit, dict) or 'strength' not in edit:
            raise ExportError('Projection requires a strength')
        strength = _validate_strength(edit['strength'])
        keys = [prefix + suffix for suffix in ('.weight', '.weight_scale', '.weight_scale_2')]
        if any(key not in by_name for key in keys):
            raise ExportError('Projection is not a complete NVFP4 bundle')
        if strength:
            values = edit.get('direction')
            if not isinstance(values, list) or len(values) != by_name[keys[0]]['shape'][0] or any(type(v) not in {float, int} or not math.isfinite(v) for v in values) or not any(values):
                raise ExportError('Direction must be a finite nonzero vector matching output width')
    return edits, policy, force


def merge_export(source_export: Path, destination: Path, edit_spec: dict, *,
                 scratch_limit_bytes: int = GIB) -> dict:
    """Write a new export, modifying only explicitly selected NVFP4 projections.

    Only one selected bundle is materialized at a time. All unchanged tensors
    are copied as raw byte ranges and checked against the source receipt.
    This function does not import, evaluate, or approve the resulting model.
    """
    source = Path(source_export).resolve()
    _validate_scratch_budget(scratch_limit_bytes)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if source == destination.resolve() or source in destination.resolve().parents:
        raise ExportError('Destination must be outside source export')
    # Match inference admission: copying just the declared asset prefix could
    # otherwise silently accept appended bytes or undeclared loader assets.
    # Keep the source immutable for the duration of the merge.
    source_verification = verify_export(source, require_unmodified=True)
    receipt, entries = _source_entries(source)
    edits, policy, force = _validate_spec(edit_spec, receipt, entries)
    by_name = {entry['name']: entry for entry in entries}
    total = sum(entry['size'] for entry in entries)
    largest = max((sum(by_name[p + s]['size'] for s in ('.weight', '.weight_scale', '.weight_scale_2')) for p in edits), default=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < total + 2 * largest + 32 * 1024 ** 2:
        raise ExportError('Insufficient disk space for output and one scratch bundle')
    staging = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    active_prefix, active_entries, remaining = None, None, 0
    reports, records, weight_map = {}, [], {}
    try:
        scratch = staging / '.merge-scratch'
        scratch.mkdir()
        shards = _shards(entries)
        for number, shard in enumerate(shards, 1):
            filename = f'model-{number:05d}-of-{len(shards):05d}.safetensors'
            header, offset = {}, 0
            for entry in shard:
                header[entry['name']] = {'dtype': entry['dtype'], 'shape': entry['shape'], 'data_offsets': [offset, offset + entry['size']]}
                offset += entry['size']
                weight_map[entry['name']] = filename
            encoded = json.dumps(header, separators=(',', ':')).encode()
            encoded += b' ' * (-len(encoded) % 8)
            with (staging / filename).open('xb') as target:
                target.write(struct.pack('<Q', len(encoded)))
                target.write(encoded)
                for entry in shard:
                    name = entry['name']
                    prefix = next((name[:-len(s)] for s in ('.weight_scale_2', '.weight_scale', '.weight') if name.endswith(s)), '')
                    modified = prefix in edits and (edits[prefix]['strength'] != 0 or force)
                    source_entry = entry
                    if modified:
                        if active_prefix is None:
                            keys = [prefix + suffix for suffix in ('.weight', '.weight_scale', '.weight_scale_2')]
                            _write_bundle(scratch / 'input.safetensors', [by_name[key] for key in keys])
                            mx = _mlx()
                            arrays = mx.load(str(scratch / 'input.safetensors'))
                            packed, blocks, factor, report = merge_nvfp4(*(arrays[key] for key in keys), edits[prefix].get('direction'), edits[prefix]['strength'], scale_policy=policy, force_requantize=force, scratch_limit_bytes=scratch_limit_bytes)
                            mx.save_safetensors(str(scratch / 'output.safetensors'), dict(zip(keys, (packed, blocks, factor))), metadata={'format': 'mlx-abliterate'})
                            del arrays, packed, blocks, factor
                            mx.clear_cache()
                            reports[prefix] = report
                            output_header = read_safetensors_header(scratch / 'output.safetensors')
                            for key in keys:
                                if any(output_header[key][field] != by_name[key][field] for field in ('shape', 'dtype')):
                                    raise ExportError('Merged tensor dtype or shape changed unexpectedly')
                            with (scratch / 'output.safetensors').open('rb') as stream:
                                base = 8 + struct.unpack('<Q', stream.read(8))[0]
                            active_entries = {key: {**by_name[key], 'path': scratch / 'output.safetensors', 'offset': base + output_header[key]['data_offsets'][0]} for key in keys}
                            active_prefix, remaining = prefix, 3
                        if active_prefix != prefix:
                            raise ExportError('Selected bundle tensors are not contiguous')
                        source_entry = active_entries[name]
                    with source_entry['path'].open('rb') as origin:
                        sha = _copy_range(origin, target, source_entry['offset'], source_entry['size'])
                    if not modified and sha != entry['payload_sha256']:
                        raise ExportError('Unchanged source tensor hash mismatch')
                    records.append({k: v for k, v in {**entry, 'payload_sha256': sha, 'shard': filename}.items() if k != 'path'})
                    if modified:
                        remaining -= 1
                        if remaining == 0:
                            for file in scratch.iterdir():
                                file.unlink()
                            active_prefix, active_entries = None, None
                target.flush()
                os.fsync(target.fileno())
        scratch.rmdir()
        for asset in receipt['assets']:
            name = asset['name']
            if Path(name).name != name or name.startswith('.'):
                raise ExportError('Unsafe asset path')
            with (source / name).open('rb') as origin, (staging / name).open('xb') as target:
                sha = _copy_range(origin, target, 0, asset['size'])
            if 'sha256:' + sha != asset['digest']:
                raise ExportError('Source JSON asset hash mismatch')
        _dump(staging / 'model.safetensors.index.json', {'metadata': {'total_size': total}, 'weight_map': weight_map})
        receipts = staging / '.mlx-abliterate'
        receipts.mkdir()
        shutil.copyfile(source / '.mlx-abliterate/create-options.json', receipts / 'create-options.json')
        shutil.copyfile(source / 'Modelfile', staging / 'Modelfile')
        _dump(receipts / 'edit-spec.json', edit_spec)
        result = {'schema_version': 1, 'edited_projection_count': len(reports), 'projection_reports': reports,
                  'source_content_fingerprint': source_verification['content_fingerprint'],
                  'source_manifest_digest': receipt['source_manifest_digest'], 'validated_import': False,
                  'quality_validated': False, 'source_receipt_sha256': hashlib.sha256((source / '.mlx-abliterate/export-report.json').read_bytes()).hexdigest()}
        _dump(receipts / 'merge-report.json', result)
        _dump(receipts / 'export-report.json', {**receipt, 'tensors': records, 'validated_import': False, 'modified': bool(reports)})
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        _publish_directory(staging, destination)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
