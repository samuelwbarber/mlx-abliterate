"""Bounded, byte-preserving Ollama-to-Safetensors interchange spike.

This is an original implementation of an interchange mapping, not a model editor.
NVFP4 payloads use the ModelOpt naming accepted by Ollama v0.40.16.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import sys

from .inventory import blob_path, inspect_model, read_safetensors_header, resolve_model

CHUNK_BYTES = 4 * 1024 * 1024
SHARD_BYTES = 1024 * 1024 * 1024
TENSOR_TYPE = 'application/vnd.ollama.image.tensor'
JSON_TYPE = 'application/vnd.ollama.image.json'
MANIFEST_TYPE = 'application/vnd.docker.distribution.manifest.v2+json'


class ExportError(ValueError):
    """The source cannot be exported without an unvalidated transformation."""


def _json(path):
    return json.loads(path.read_text())


def _prepare(model, models_dir=None, *, verify_payloads=False):
    inventory = inspect_model(model, models_dir, verify_payloads=verify_payloads)
    root, manifest_path = resolve_model(model, models_dir)
    manifest_bytes = manifest_path.read_bytes()
    if 'sha256:' + hashlib.sha256(manifest_bytes).hexdigest() != inventory['manifest_digest']:
        raise ExportError('Source manifest changed during inspection')
    manifest = json.loads(manifest_bytes)
    config = _json(blob_path(root, manifest['config']['digest']))
    tensors, assets, options, seen = [], [], {}, set()
    for key in ('renderer', 'parser', 'requires', 'capabilities'):
        if config.get(key):
            options[key] = config[key]
    for layer in manifest['layers']:
        kind, digest = layer['mediaType'], layer['digest']
        path = blob_path(root, digest)
        if kind == TENSOR_TYPE:
            header = read_safetensors_header(path)
            with path.open('rb') as source:
                start = 8 + struct.unpack('<Q', source.read(8))[0]
            metadata = header.get('__metadata__', {})
            name = layer['name']
            names = set(header) - {'__metadata__'}
            if metadata.get('quant_type') == 'nvfp4':
                if names != {name, name + '.scale', name + '.global_scale'}:
                    raise ExportError(f'Unexpected NVFP4 companions for {name}')
                w, s, g = (header[name + suffix] for suffix in ('', '.scale', '.global_scale'))
                if (w['dtype'] != 'U32' or len(w['shape']) != 2 or
                    s['dtype'] != 'U8' or s['shape'] != [w['shape'][0], w['shape'][1] // 2] or
                    w['shape'][1] % 2 or g['dtype'] != 'F32' or g['shape'] != []):
                    raise ExportError(f'Unsupported NVFP4 shape/dtype for {name}')
                if set(metadata) - {'quant_type', 'group_size'} or metadata.get('group_size', '16') != '16':
                    raise ExportError(f'Unsupported NVFP4 metadata for {name}')
                mapping = {name: name, name + '.scale': name + '_scale', name + '.global_scale': name + '_scale_2'}
            elif not metadata and names == {name} and header[name]['dtype'] in {'BF16', 'F16', 'F32'}:
                mapping = {name: name}
            else:
                raise ExportError(f'Unsupported tensor layout for {name}')
            for original, output in mapping.items():
                if output in seen:
                    raise ExportError(f'Duplicate tensor {output}')
                seen.add(output)
                entry = header[original]
                lo, hi = entry['data_offsets']
                tensors.append({'name': output, 'original_name': original, 'dtype': entry['dtype'],
                                'shape': entry['shape'], 'digest': digest, 'offset': start + lo, 'size': hi - lo})
        elif kind == JSON_TYPE:
            name = layer['name']
            if Path(name).name != name or not name.endswith('.json') or name.startswith('.'):
                raise ExportError(f'Unsafe asset filename: {name}')
            if name in {a['name'] for a in assets}:
                raise ExportError(f'Duplicate asset filename: {name}')
            assets.append({'name': name, 'digest': digest, 'size': layer['size']})
        elif kind == MANIFEST_TYPE:
            # Historical parent/provenance, not a separate draft model.
            continue
        elif kind == 'application/vnd.ollama.image.params':
            options['parameters'] = _json(path)
        elif kind == 'application/vnd.ollama.image.license':
            options.setdefault('license', []).append(path.read_text())
        elif kind in {'application/vnd.ollama.image.template', 'application/vnd.ollama.image.system'}:
            options[kind.rsplit('.', 1)[1]] = path.read_text()
        else:
            raise ExportError(f'Unsupported manifest layer: {kind}')
    if not tensors or 'config.json' not in {a['name'] for a in assets}:
        raise ExportError('Tensor layers and config.json are required')
    reserved = {'model.safetensors.index.json', 'export-report.json', 'create-options.json'}
    if reserved & {a['name'] for a in assets}:
        raise ExportError('Source contains an export-reserved filename')
    return root, inventory, tensors, assets, options


def _shards(tensors):
    shards, current, size = [], [], 0
    for tensor in tensors:
        if current and size + tensor['size'] > SHARD_BYTES:
            shards.append(current)
            current, size = [], 0
        current.append(tensor)
        size += tensor['size']
    if current:
        shards.append(current)
    return shards


def plan_export(model: str, models_dir: Path | None = None) -> dict:
    """Inspect without reading weight payloads or writing model files."""
    _, inventory, tensors, assets, options = _prepare(model, models_dir)
    total = sum(t['size'] for t in tensors) + sum(a['size'] for a in assets)
    return {'schema_version': 1, 'source_model': model,
            'source_manifest_digest': inventory['manifest_digest'],
            'format': 'modelopt-nvfp4-safetensors', 'tensor_count': len(tensors),
            'shard_count': len(_shards(tensors)), 'payload_bytes': total,
            'estimated_disk_bytes': total + 4 * 1024 * 1024,
            'copy_buffer_bytes': CHUNK_BYTES, 'runtime_metadata_keys': sorted(options),
            'validated_import': False,
            'notes': ['No weight decoding or quantization.', 'Requires separate Ollama import and payload verification.']}


def _copy_range(source, destination, offset, size):
    source.seek(offset)
    remaining, digest = size, hashlib.sha256()
    while remaining:
        data = source.read(min(remaining, CHUNK_BYTES))
        if not data:
            raise ExportError('Source truncated during export')
        destination.write(data)
        digest.update(data)
        remaining -= len(data)
    return digest.hexdigest()


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _quote(value):
    text = str(value)
    if '"""' in text:
        raise ExportError('Metadata contains unsupported triple-quote delimiter')
    return '"""' + text + '"""'


def _modelfile(options):
    lines = ['FROM ./']
    for key in ('renderer', 'parser', 'requires', 'template', 'system'):
        if key in options:
            lines.append(key.upper() + ' ' + _quote(options[key]))
    for value in options.get('capabilities', []):
        lines.append('CAPABILITY ' + _quote(value))
    for value in options.get('license', []):
        lines.append('LICENSE ' + _quote(value))
    for key, value in options.get('parameters', {}).items():
        if not key.replace('_', '').isalnum():
            raise ExportError('Invalid parameter name')
        for item in value if isinstance(value, list) else [value]:
            encoded = _quote(item) if isinstance(item, str) else json.dumps(item)
            lines.append('PARAMETER ' + key + ' ' + encoded)
    return '\n'.join(lines) + '\n'


def _publish_directory(staging: Path, destination: Path) -> None:
    """Atomically rename without replacing even an empty concurrent directory."""
    if sys.platform == 'win32':
        os.rename(staging, destination)  # Windows rename rejects existing dst.
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin' and hasattr(library, 'renamex_np'):
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(os.fsencode(staging), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith('linux') and hasattr(library, 'renameat2'):
        function = library.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100, os.fsencode(staging), -100, os.fsencode(destination), 1)  # NOREPLACE
    else:
        raise ExportError('Atomic exclusive directory rename is unavailable on this platform')
    if result:
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(number, os.strerror(number), str(destination))
        raise OSError(number, os.strerror(number), str(destination))


def export_model(model: str, destination: Path, models_dir: Path | None = None) -> dict:
    """Atomically write a new interchange directory, using a fixed copy buffer.

    Does not import into Ollama or claim runtime fidelity. Never overwrites a
    destination. Tensor payload SHA256s enable checking the imported artifact.
    """
    root, inventory, tensors, assets, options = _prepare(model, models_dir, verify_payloads=True)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if root.resolve() == destination.resolve() or root.resolve() in destination.resolve().parents:
        raise ExportError('Export destination must be outside the Ollama model store')
    destination.parent.mkdir(parents=True, exist_ok=True)
    needed = sum(t['size'] for t in tensors) + sum(a['size'] for a in assets) + 4 * 1024 * 1024
    if shutil.disk_usage(destination.parent).free < needed:
        raise ExportError(f'Insufficient disk space: need at least {needed} bytes')
    modelfile = _modelfile(options)
    source_digests = {entry['digest'] for entry in tensors + assets}
    source_stats = {digest: blob_path(root, digest).stat() for digest in source_digests}
    staging = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    try:
        weight_map, records = {}, []
        shards = _shards(tensors)
        for index, entries in enumerate(shards, 1):
            filename = f'model-{index:05d}-of-{len(shards):05d}.safetensors'
            header, offset = {}, 0
            for entry in entries:
                header[entry['name']] = {'dtype': entry['dtype'], 'shape': entry['shape'], 'data_offsets': [offset, offset + entry['size']]}
                offset += entry['size']
                weight_map[entry['name']] = filename
            encoded = json.dumps(header, separators=(',', ':')).encode()
            encoded += b' ' * (-len(encoded) % 8)
            with (staging / filename).open('xb') as target:
                target.write(struct.pack('<Q', len(encoded)))
                target.write(encoded)
                for entry in entries:
                    with blob_path(root, entry['digest']).open('rb') as source:
                        sha = _copy_range(source, target, entry['offset'], entry['size'])
                    records.append({**entry, 'payload_sha256': sha, 'shard': filename})
                target.flush()
                os.fsync(target.fileno())
        for asset in assets:
            with blob_path(root, asset['digest']).open('rb') as source, (staging / asset['name']).open('xb') as target:
                sha = _copy_range(source, target, 0, asset['size'])
                if 'sha256:' + sha != asset['digest']:
                    raise ExportError(f'Asset digest mismatch: {asset["name"]}')
        _dump(staging / 'model.safetensors.index.json', {'metadata': {'total_size': sum(t['size'] for t in tensors)}, 'weight_map': weight_map})
        receipt_dir = staging / '.mlx-abliterate'
        receipt_dir.mkdir()
        _dump(receipt_dir / 'create-options.json', options)
        (staging / 'Modelfile').write_text(modelfile)
        report = {'schema_version': 1, 'source_model': model, 'source_manifest_digest': inventory['manifest_digest'],
                  'format': 'modelopt-nvfp4-safetensors', 'validated_import': False, 'tensors': records, 'assets': assets}
        _dump(receipt_dir / 'export-report.json', report)
        for digest, original in source_stats.items():
            current = blob_path(root, digest).stat()
            if (current.st_ino, current.st_size, current.st_mtime_ns) != (original.st_ino, original.st_size, original.st_mtime_ns):
                raise ExportError('Source blob changed during export')
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        _publish_directory(staging, destination)
        return {'source_manifest_digest': inventory['manifest_digest'], 'tensor_count': len(records),
                'shard_count': len(shards), 'validated_import': False, 'destination': str(destination)}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_import(model: str, export_directory: Path, models_dir: Path | None = None) -> dict:
    """Read and hash every imported tensor payload against an export receipt."""
    report = _json(Path(export_directory) / '.mlx-abliterate' / 'export-report.json')
    options = _json(Path(export_directory) / '.mlx-abliterate' / 'create-options.json')
    root, _, tensors, assets, actual_options = _prepare(model, models_dir)
    expected = {t['original_name']: t for t in report['tensors']}
    actual = {t['original_name']: t for t in tensors}
    if expected.keys() != actual.keys():
        raise ExportError('Imported tensor names differ from exported source')
    for name, tensor in actual.items():
        wanted = expected[name]
        if any(tensor[key] != wanted[key] for key in ('shape', 'dtype', 'size')):
            raise ExportError(f'Imported tensor metadata differs: {name}')
        with blob_path(root, tensor['digest']).open('rb') as source, open(os.devnull, 'wb') as sink:
            sha = _copy_range(source, sink, tensor['offset'], tensor['size'])
        if sha != wanted['payload_sha256']:
            raise ExportError(f'Imported tensor payload differs: {name}')
    expected_assets = {a['name']: a['digest'] for a in report['assets']}
    actual_assets = {a['name']: a['digest'] for a in assets}
    if actual_assets != expected_assets:
        raise ExportError('Imported JSON asset hashes differ')
    if actual_options != options:
        raise ExportError('Imported runtime metadata differs')
    return {'model': model, 'tensor_count': len(tensors), 'payloads_verified': True,
            'assets_verified': True, 'runtime_metadata_verified': True,
            'serving_speed_verified': False, 'model_behavior_verified': False}


def verify_export(export_directory: Path, *, require_unmodified: bool = False) -> dict:
    """Verify local export bytes and return a content-based experiment identity.

    This scans tensor payloads with a fixed 4 MiB buffer. The original manifest
    digest alone is not an identity for a merged or locally modified checkpoint.
    Every tensor, asset, runtime option, and Modelfile contributes to this digest.
    Callers must keep the verified directory immutable until inference finishes.
    """
    root = Path(export_directory).resolve()

    def local_file(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ExportError('Missing or escaping export file')
        return path

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ExportError('Duplicate JSON key in export metadata')
            result[key] = value
        return result

    def bounded_json(relative):
        path = local_file(relative)
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ExportError('Export metadata exceeds bounded size')
        value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
        if not isinstance(value, dict):
            raise ExportError('Expected export metadata object')
        return value

    def hash_range(path, offset=0, size=None):
        size = path.stat().st_size if size is None else size
        digest, remaining = hashlib.sha256(), size
        with path.open('rb') as source:
            source.seek(offset)
            while remaining:
                chunk = source.read(min(remaining, CHUNK_BYTES))
                if not chunk:
                    raise ExportError('Export file truncated during verification')
                digest.update(chunk)
                remaining -= len(chunk)
        return digest.hexdigest()

    receipt = bounded_json('.mlx-abliterate/export-report.json')
    if receipt.get('schema_version') != 1 or not isinstance(receipt.get('source_manifest_digest'), str):
        raise ExportError('Unsupported export receipt schema')
    if require_unmodified and receipt.get('modified'):
        raise ExportError('Collection and pilots require an unmodified no-op export')
    index = bounded_json('model.safetensors.index.json').get('weight_map')
    if not isinstance(index, dict) or not index or any(not isinstance(k, str) or not isinstance(v, str) for k, v in index.items()):
        raise ExportError('Invalid export tensor index')
    records = receipt.get('tensors')
    if not isinstance(records, list) or any(not isinstance(r, dict) or not isinstance(r.get('name'), str) for r in records):
        raise ExportError('Invalid export tensor receipts')
    by_name = {record['name']: record for record in records}
    if len(by_name) != len(records) or set(index) != set(by_name):
        raise ExportError('Export receipt and tensor index disagree')
    headers, bases, seen = {}, {}, set()
    for shard in sorted(set(index.values())):
        if Path(shard).name != shard or not shard.endswith('.safetensors'):
            raise ExportError('Unsafe shard filename')
        path = local_file(shard)
        header = read_safetensors_header(path)
        names = set(header) - {'__metadata__'}
        if names & seen:
            raise ExportError('Duplicate tensor across export shards')
        seen.update(names)
        headers[shard] = header
        with path.open('rb') as source:
            bases[shard] = 8 + struct.unpack('<Q', source.read(8))[0]
    if seen != set(index):
        raise ExportError('Export contains unindexed or missing tensors')
    verified_tensors = []
    for name in sorted(index):
        shard = index[name]
        descriptor = headers[shard].get(name)
        record = by_name[name]
        if descriptor is None:
            raise ExportError('Tensor index references the wrong shard')
        lo, hi = descriptor['data_offsets']
        size = hi - lo
        if descriptor['dtype'] != record.get('dtype') or descriptor['shape'] != record.get('shape') or size != record.get('size'):
            raise ExportError('Export tensor shape/dtype/size differs from receipt')
        actual = hash_range(local_file(shard), bases[shard] + lo, size)
        if actual != record.get('payload_sha256'):
            raise ExportError('Export tensor payload hash mismatch')
        verified_tensors.append({'name': name, 'dtype': descriptor['dtype'], 'shape': descriptor['shape'],
                                 'size': size, 'payload_sha256': actual})
    assets = receipt.get('assets')
    if not isinstance(assets, list):
        raise ExportError('Invalid export asset receipts')
    verified_assets, asset_names = [], set()
    for asset in assets:
        name = asset.get('name') if isinstance(asset, dict) else None
        if not isinstance(name, str) or Path(name).name != name or name.startswith('.') or name in asset_names:
            raise ExportError('Unsafe or duplicate export asset')
        asset_names.add(name)
        path = local_file(name)
        size = path.stat().st_size
        if size != asset.get('size'):
            raise ExportError('Export asset size differs from receipt')
        actual = hash_range(path)
        if 'sha256:' + actual != asset.get('digest'):
            raise ExportError('Export asset hash mismatch')
        verified_assets.append({'name': name, 'size': size, 'sha256': actual})
    allowed = asset_names | set(index.values()) | {'model.safetensors.index.json', 'Modelfile', '.mlx-abliterate'}
    if {p.name for p in root.iterdir()} != allowed:
        raise ExportError('Export contains undeclared files that could affect loading')
    options = bounded_json('.mlx-abliterate/create-options.json')
    modelfile = local_file('Modelfile')
    if modelfile.stat().st_size > 32 * 1024 * 1024:
        raise ExportError('Modelfile exceeds bounded size')
    identity = {'schema_version': 1, 'source_manifest_digest': receipt['source_manifest_digest'],
                'modified': bool(receipt.get('modified')), 'tensors': verified_tensors,
                'assets': sorted(verified_assets, key=lambda item: item['name']),
                'create_options_sha256': hash_range(local_file('.mlx-abliterate/create-options.json')),
                'modelfile_sha256': hash_range(modelfile)}
    # Parse options even though identity uses exact bytes, detecting malformed
    # runtime metadata before consumers use it. No runtime fields are ignored.
    del options
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return {'schema_version': 1, 'source_manifest_digest': receipt['source_manifest_digest'],
            'content_fingerprint': fingerprint, 'modified': bool(receipt.get('modified')),
            'tensor_count': len(verified_tensors), 'asset_count': len(verified_assets),
            'payload_bytes_verified': sum(r['size'] for r in verified_tensors),
            'copy_buffer_bytes': CHUNK_BYTES, 'payloads_verified': True}
