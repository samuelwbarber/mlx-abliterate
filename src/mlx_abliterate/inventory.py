"""Bounded, read-only inspection of Ollama's Safetensors model store.

Tensor payloads are never loaded. Their content-addressed digests are declared,
not verified; small metadata blobs and manifests are hashed before parsing.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from collections import Counter
from typing import Any

MANIFEST_TYPE = 'application/vnd.docker.distribution.manifest.v2+json'
TENSOR_TYPE = 'application/vnd.ollama.image.tensor'
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
_DIGEST = re.compile(r'sha256:[0-9a-f]{64}\Z')
_COMPONENT = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.:-]*\Z')
_DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'U16': 2, 'I16': 2,
                'F16': 2, 'BF16': 2, 'U32': 4, 'I32': 4, 'F32': 4,
                'U64': 8, 'I64': 8, 'F64': 8, 'F8_E4M3': 1, 'F8_E5M2': 1}


class InventoryError(ValueError):
    """Malformed, unsupported, or inconsistent local model artifact."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InventoryError('Duplicate JSON object key')
        result[key] = value
    return result


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise InventoryError('Model path escapes its store')
    return resolved


def resolve_model(model: str, models_dir: Path | None = None) -> tuple[Path, Path]:
    """Resolve a model tag without network access or accepting filesystem paths."""
    root = Path(models_dir or os.environ.get('OLLAMA_MODELS', Path.home() / '.ollama/models')).expanduser().resolve()
    if not model or '@' in model or '\\' in model:
        raise InventoryError('Expected an Ollama model name, optionally with a tag')
    components = model.split('/')
    if any(not _COMPONENT.fullmatch(x) or x in {'.', '..'} for x in components):
        raise InventoryError('Invalid model name')
    name, sep, tag = components[-1].partition(':')
    if not name or (sep and (not tag or ':' in tag)):
        raise InventoryError('Invalid model tag')
    components[-1] = name
    if len(components) == 1:
        components = ['registry.ollama.ai', 'library', *components]
    elif len(components) == 2:
        components = ['registry.ollama.ai', *components]
    elif len(components) != 3:
        raise InventoryError('Expected [registry/][namespace/]model[:tag]')
    host, namespace, name = components
    public = host.lower() in {'registry.ollama.ai', 'ollama.com'}
    v2_host = 'ollama.com' if public else host.replace(':', '%3A')
    candidates = [root / 'manifests-v2' / v2_host / namespace / name / (tag or 'latest'),
                  root / 'manifests' / host / namespace / name / (tag or 'latest')]
    if public:
        alternate = 'ollama.com' if host.lower() == 'registry.ollama.ai' else 'registry.ollama.ai'
        candidates.append(root / 'manifests' / alternate / namespace / name / (tag or 'latest'))
    for candidate in candidates:
        path = _inside(root, candidate)
        if candidate.is_symlink():
            # v2 tags point at immutable, content-addressed manifest blobs.
            # Resolve symlinks only inside the blob directory and verify the
            # filename digest before accepting their JSON as a manifest.
            if path.parent != (root / 'blobs').resolve():
                raise InventoryError('Manifest symlink must target the blob directory')
            digest = path.name.replace('sha256-', 'sha256:', 1)
            if not _DIGEST.fullmatch(digest):
                raise InventoryError('Manifest symlink target is not a sha256 blob')
            if not path.exists():
                continue  # Match Ollama's fallback for dangling v2 links.
            _json_bytes(path, digest)
        if path.is_file():
            return root, path
    raise InventoryError(f'Model manifest not found for {model}')


def blob_path(root: Path, digest: str) -> Path:
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise InventoryError('Invalid sha256 blob digest')
    path = _inside(root, root / 'blobs' / digest.replace(':', '-'))
    if not path.is_file():
        raise InventoryError(f'Missing blob {digest}')
    return path


def _json_bytes(path: Path, digest: str | None = None) -> tuple[Any, str]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise InventoryError('JSON metadata exceeds bounded reader limit')
    data = path.read_bytes()
    actual = 'sha256:' + hashlib.sha256(data).hexdigest()
    if digest is not None and digest != actual:
        raise InventoryError(f'Metadata content hash mismatch: {digest}')
    try:
        return json.loads(data, object_pairs_hook=_unique_object), actual
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InventoryError('Invalid JSON metadata') from exc


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Validate offsets/dtypes using file size, reading only the bounded header."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise InventoryError('Truncated Safetensors prefix')
        length = struct.unpack('<Q', prefix)[0]
        if length < 2 or length > MAX_HEADER_BYTES or length > size - 8:
            raise InventoryError('Invalid or excessive Safetensors header length')
        try:
            header = json.loads(stream.read(length), object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InventoryError('Invalid Safetensors JSON header') from exc
    if not isinstance(header, dict) or not header:
        raise InventoryError('Expected Safetensors object header')
    intervals = []
    for name, tensor in header.items():
        if name == '__metadata__':
            if not isinstance(tensor, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in tensor.items()):
                raise InventoryError('Invalid Safetensors metadata')
            continue
        if not isinstance(tensor, dict):
            raise InventoryError('Invalid tensor descriptor')
        shape, dtype, offsets = tensor.get('shape'), tensor.get('dtype'), tensor.get('data_offsets')
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise InventoryError('Invalid tensor shape')
        if dtype not in _DTYPE_BYTES:
            raise InventoryError(f'Unsupported tensor dtype: {dtype}')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise InventoryError('Invalid tensor offsets')
        start, end = offsets
        if start < 0 or end < start or end > size - 8 - length:
            raise InventoryError('Tensor offsets exceed payload')
        if end - start != math.prod(shape) * _DTYPE_BYTES[dtype]:
            raise InventoryError('Tensor byte length disagrees with shape/dtype')
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise InventoryError('Tensor payload has overlapping offsets or gaps')
        cursor = end
    if cursor != size - 8 - length:
        raise InventoryError('Tensor payload contains trailing bytes')
    return header


def tensor_layout(header: dict, name: str) -> tuple[str, list[int]]:
    quant = header.get('__metadata__', {}).get('quant_type')
    tensor = header.get(name)
    if not isinstance(tensor, dict):
        raise InventoryError(f'Manifest tensor name missing from header: {name}')
    keys = set(header) - {'__metadata__'}
    if quant is None:
        if keys != {name} or tensor['dtype'] not in {'BF16', 'F16', 'F32', 'F64'}:
            raise InventoryError(f'Unsupported unquantized tensor layout: {name}')
        return 'none', tensor['shape']
    if quant != 'nvfp4':
        raise InventoryError(f'Unsupported quantization layout: {quant}')
    metadata = header.get('__metadata__', {})
    if metadata.get('group_size', '16') != '16' or set(metadata) - {'quant_type', 'group_size'}:
        raise InventoryError('Unsupported NVFP4 quantization metadata')
    expected = {name, name + '.scale', name + '.global_scale'}
    if keys != expected or tensor['dtype'] != 'U32' or len(tensor['shape']) != 2:
        raise InventoryError(f'Unsupported NVFP4 packed weight layout: {name}')
    rows, packed_cols = tensor['shape']
    logical = [rows, packed_cols * 8]
    scale, global_scale = header[name + '.scale'], header[name + '.global_scale']
    if logical[1] % 16 or scale['dtype'] != 'U8' or scale['shape'] != [rows, logical[1] // 16]:
        raise InventoryError('Unsupported NVFP4 block-scale layout')
    if global_scale['dtype'] != 'F32' or global_scale['shape'] != []:
        raise InventoryError('Unsupported NVFP4 global-scale layout')
    return 'nvfp4', logical


def inspect_model(model: str, models_dir: Path | None = None, *,
                  verify_payloads: bool = False) -> dict[str, Any]:
    """Inventory manifests, metadata and tensor headers without loading weights.

    Set verify_payloads=True to stream SHA-256 over all tensor blobs with a
    fixed 1 MiB buffer. The default validates metadata and payload sizes only.

    Only the root manifest's tensors are active. Nested manifests are recorded as
    provenance, not silently merged into the current tensor set or MTP model.
    """
    root, manifest_path = resolve_model(model, models_dir)
    initial, initial_digest = _json_bytes(manifest_path)
    manifests: list[dict] = []
    blobs: dict[str, dict] = {}
    headers: dict[str, dict] = {}
    visited: set[str] = set()
    active: set[str] = set()

    def descriptor(value: Any) -> Path:
        if not isinstance(value, dict) or not isinstance(value.get('mediaType'), str):
            raise InventoryError('Invalid manifest descriptor')
        digest, size = value.get('digest'), value.get('size')
        if type(size) is not int or size < 0:
            raise InventoryError('Invalid descriptor size')
        path = blob_path(root, digest)
        if path.stat().st_size != size:
            raise InventoryError(f'Blob size mismatch: {digest}')
        if digest in blobs and blobs[digest]['size'] != size:
            raise InventoryError('Conflicting blob sizes')
        record = blobs.setdefault(digest, {'digest': digest, 'size': size, 'media_types': [], 'content_hash_verified': False})
        if value['mediaType'] not in record['media_types']:
            record['media_types'].append(value['mediaType'])
        return path

    def visit(manifest: Any, digest: str, depth: int = 0) -> None:
        if depth > 32 or len(visited) > 10000:
            raise InventoryError('Manifest graph exceeds traversal limit')
        if digest in active:
            raise InventoryError('Manifest graph contains a cycle')
        if digest in visited:
            return
        if not isinstance(manifest, dict) or manifest.get('schemaVersion') != 2 or not isinstance(manifest.get('layers'), list):
            raise InventoryError('Unsupported manifest schema')
        active.add(digest)
        config_descriptor = manifest.get('config')
        config_path = descriptor(config_descriptor)
        _json_bytes(config_path, config_descriptor['digest'])
        blobs[config_descriptor['digest']]['content_hash_verified'] = True
        manifests.append({'digest': digest, 'config': config_descriptor, 'layers': manifest['layers']})
        for layer in manifest['layers']:
            path = descriptor(layer)
            media = layer['mediaType']
            if media == MANIFEST_TYPE:
                child, child_digest = _json_bytes(path, layer['digest'])
                blobs[child_digest]['content_hash_verified'] = True
                visit(child, child_digest, depth + 1)
            elif media == TENSOR_TYPE:
                if layer['digest'] not in headers:
                    headers[layer['digest']] = read_safetensors_header(path)
                tensor_layout(headers[layer['digest']], layer.get('name', ''))
            elif media.startswith('application/vnd.ollama.image.'):
                # Hash bounded metadata, including templates/licenses, without
                # publishing its contents or accepting an unknown weight format.
                if media.rsplit('.', 1)[-1] not in {'json', 'license', 'params', 'template', 'system', 'messages'}:
                    raise InventoryError(f'Unsupported layer media type: {media}')
                if path.stat().st_size > MAX_JSON_BYTES:
                    raise InventoryError('Metadata blob exceeds reader limit')
                if 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest() != layer['digest']:
                    raise InventoryError('Metadata content hash mismatch')
                blobs[layer['digest']]['content_hash_verified'] = True
            else:
                raise InventoryError(f'Unsupported layer media type: {media}')
        active.remove(digest)
        visited.add(digest)

    visit(initial, initial_digest)
    if verify_payloads:
        for digest, record in blobs.items():
            if record['content_hash_verified']:
                continue
            hasher = hashlib.sha256()
            with blob_path(root, digest).open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    hasher.update(chunk)
            if 'sha256:' + hasher.hexdigest() != digest:
                raise InventoryError(f'Tensor payload content hash mismatch: {digest}')
            record['content_hash_verified'] = True
    config, _ = _json_bytes(blob_path(root, initial['config']['digest']))
    layers = []
    model_config: dict = {}
    seen_names: set[str] = set()
    for layer in initial['layers']:
        enriched = dict(layer)
        name = layer.get('name')
        if name:
            if name in seen_names:
                raise InventoryError(f'Duplicate active layer name: {name}')
            seen_names.add(name)
        if layer['mediaType'] == TENSOR_TYPE:
            header = headers[layer['digest']]
            quant, logical = tensor_layout(header, name)
            enriched.update(tensor_header=header, quantization=quant, logical_shape=logical)
        if name == 'config.json':
            model_config, _ = _json_bytes(blob_path(root, layer['digest']))
        layers.append(enriched)
    tensors = [x for x in layers if x['mediaType'] == TENSOR_TYPE]
    projections = [x for x in tensors if x['name'].startswith('model.language_model.layers.') and x['name'].endswith(('.self_attn.o_proj.weight', '.linear_attn.out_proj.weight', '.mlp.down_proj.weight'))]
    return {
        'schema_version': 1, 'model': model, 'manifest_digest': initial_digest,
        'manifest_relative_path': str(manifest_path.relative_to(root)),
        'config': config, 'model_config': model_config, 'manifests': manifests,
        'layers': layers, 'blobs': sorted(blobs.values(), key=lambda x: x['digest']),
        'summary': {
            'manifest_count': len(manifests), 'unique_blob_count': len(blobs),
            'unique_blob_bytes': sum(x['size'] for x in blobs.values()),
            'active_tensor_count': len(tensors),
            'active_tensor_bytes': sum(x['size'] for x in {x['digest']: x for x in tensors}.values()),
            'quantization_counts': dict(Counter(x['quantization'] for x in tensors)),
            'mtp_tensor_count': sum(x['name'].startswith('mtp.') for x in tensors),
            'editable_projection_count': len(projections),
            'largest_editable_bf16_bytes': max((math.prod(x['logical_shape']) * 2 for x in projections), default=0),
            'tensor_payload_hashes_verified': verify_payloads,
        },
    }
