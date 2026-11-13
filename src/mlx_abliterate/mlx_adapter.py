"""Narrow Qwen hybrid adapter using the installed MLX-VLM forward path.

MLX imports are delayed: inventory/preflight remain usable without Apple Silicon.
No model weights, model forward implementation, or upstream code are vendored.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import struct
import sys
from typing import Any

from .inventory import read_safetensors_header, tensor_layout

GIB = 1024 ** 3


class FidelityError(ValueError):
    """The exact-source fidelity or memory admission gate failed."""


def decode_e4m3(byte: int) -> float:
    """Decode an IEEE-like E4M3FN byte independently of MLX."""
    if not 0 <= byte <= 255:
        raise FidelityError('FP8 scale must be a byte')
    magnitude = byte & 127
    exponent, fraction = divmod(magnitude, 8)
    if magnitude == 127:
        raise FidelityError('Non-finite FP8 block scale')
    value = fraction / 512 if exponent == 0 else (1 + fraction / 8) * 2.0 ** (exponent - 7)
    return -value if byte & 128 else value


def decode_nvfp4_group(packed: bytes, scale: int, global_scale: float) -> list[float]:
    """Decode one group of 16 values; low nibble is the first value."""
    if len(packed) != 8 or not math.isfinite(global_scale) or global_scale <= 0:
        raise FidelityError('Expected eight packed bytes and a positive finite global scale')
    magnitudes = (0., .5, 1., 1.5, 2., 3., 4., 6.)
    factor = decode_e4m3(scale) * global_scale
    if factor < 0:
        raise FidelityError('Negative block scale')
    return [magnitudes[code & 7] * (-factor if code & 8 else factor)
            for byte in packed for code in (byte & 15, byte >> 4)]


def probe_nvfp4_blob(path: Path, name: str, *, row: int = 0, group: int = 0) -> dict:
    """Compare one stock group with MLX dequantization and scaled projection.

    Reads at most the tensor header plus 13 payload bytes, never the full matrix.
    This checks numeric layout, not cross-engine whole-model equivalence.
    """
    header = read_safetensors_header(path)
    quant, shape = tensor_layout(header, name)
    if quant != 'nvfp4' or not 0 <= row < shape[0] or not 0 <= group < shape[1] // 16:
        raise FidelityError('Expected an in-range NVFP4 group')
    with path.open('rb') as stream:
        base = 8 + struct.unpack('<Q', stream.read(8))[0]
        stream.seek(base + header[name]['data_offsets'][0] + row * shape[1] // 2 + group * 8)
        packed = stream.read(8)
        stream.seek(base + header[name + '.scale']['data_offsets'][0] + row * (shape[1] // 16) + group)
        scale = stream.read(1)[0]
        stream.seek(base + header[name + '.global_scale']['data_offsets'][0])
        factor = struct.unpack('<f', stream.read(4))[0]
    expected = decode_nvfp4_group(packed, scale, factor)
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.quantization.nvfp4 import ScaledQuantizedLinear
    weight = mx.array([list(struct.unpack('<II', packed))], dtype=mx.uint32)
    scales = mx.array([[scale]], dtype=mx.uint8)
    # A deterministic vector gives signed nonuniform accumulation, unlike ones.
    inputs = mx.array([[((i * 7) % 11 - 5) / 5 for i in range(16)]], dtype=mx.float32)
    linear = nn.QuantizedLinear(16, 1, bias=False, group_size=16, bits=4, mode='nvfp4')
    linear.weight, linear.scales = weight, scales
    scaled = ScaledQuantizedLinear(linear, mx.array(factor, mx.float32), 'weight_scale_2')
    actual = scaled(inputs)
    reference = inputs @ mx.array([expected], dtype=mx.float32).T
    decoded = mx.dequantize(weight, scales, group_size=16, bits=4, mode='nvfp4').astype(mx.float32) * factor
    mx.eval(actual, reference, decoded)
    decode_error = max(abs(a - b) for a, b in zip(decoded.tolist()[0], expected))
    projection_error = abs(actual.item() - reference.item())
    tolerance = 2e-6 * max(1., max(map(abs, expected)), abs(reference.item()))
    if decode_error > tolerance or projection_error > tolerance:
        raise FidelityError('NVFP4 decoding/projection disagrees with scalar reference')
    return {'row': row, 'group': group, 'decode_max_abs_error': decode_error,
            'projection_abs_error': projection_error, 'tolerance': tolerance,
            'global_scale': factor, 'block_scale_byte': scale, 'passed': True}


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise FidelityError('Missing or excessive local metadata file')
    return json.loads(path.read_text())


def preflight_export(path: Path) -> dict:
    """Validate local export topology and layouts without loading MLX weights."""
    path = Path(path).resolve()
    config = _read_json(path / 'config.json')
    text = config.get('text_config', {})
    if config.get('model_type') != 'qwen3_5' or text.get('num_hidden_layers') != 64 or text.get('hidden_size') != 5120:
        raise FidelityError('This adapter supports only the inspected dense 64-layer/5120-width Qwen hybrid')
    layer_types = text.get('layer_types')
    expected_types = ['full_attention' if (i + 1) % 4 == 0 else 'linear_attention' for i in range(64)]
    if layer_types != expected_types or text.get('full_attention_interval') != 4:
        raise FidelityError('Layer types disagree with the supported runtime attention schedule')
    if text.get('num_experts', 0):
        raise FidelityError('MoE is outside this adapter')
    index = _read_json(path / 'model.safetensors.index.json').get('weight_map')
    if not isinstance(index, dict) or not index:
        raise FidelityError('Missing tensor index')
    headers, total = {}, 0
    for shard in sorted(set(index.values())):
        candidate = (path / shard).resolve()
        if not candidate.is_relative_to(path) or candidate.parent != path:
            raise FidelityError('Shard path escapes export directory')
        headers[shard] = read_safetensors_header(candidate)
        total += candidate.stat().st_size
    for name, shard in index.items():
        if name not in headers[shard]:
            raise FidelityError('Index references missing tensor')
    actual_keys = {k for h in headers.values() for k in h if k != '__metadata__'}
    if actual_keys != set(index):
        raise FidelityError('Shards contain unindexed tensors')
    projections = []
    for i, kind in enumerate(layer_types):
        branch = 'self_attn.o_proj' if kind == 'full_attention' else 'linear_attn.out_proj'
        for suffix in (branch, 'mlp.down_proj'):
            prefix = f'model.language_model.layers.{i}.{suffix}'
            if prefix + '.weight' not in index:
                raise FidelityError('Missing expected editable projection')
            projections.append(prefix)
    quant_count = 0
    for name in index:
        if not name.endswith('.weight_scale_2'):
            continue
        prefix = name[:-len('.weight_scale_2')]
        keys = [prefix + suffix for suffix in ('.weight', '.weight_scale', '.weight_scale_2')]
        if any(key not in index for key in keys):
            raise FidelityError('Incomplete NVFP4 tensor group')
        weight, scale, global_scale = [headers[index[key]][key] for key in keys]
        if weight['dtype'] != 'U32' or len(weight['shape']) != 2:
            raise FidelityError('Expected unchanged Ollama U32 packing')
        rows, packed = weight['shape']
        if packed % 2 or scale['dtype'] != 'U8' or scale['shape'] != [rows, packed // 2]:
            raise FidelityError('Unexpected NVFP4 block scales')
        if global_scale['dtype'] != 'F32' or global_scale['shape'] != []:
            raise FidelityError('Unexpected NVFP4 global scale')
        quant_count += 1
    if any(prefix + '.weight_scale_2' not in index for prefix in projections):
        raise FidelityError('Editable projection is not NVFP4')
    return {'config': config, 'weight_map': index, 'shard_bytes': total,
            'quantized_modules': quant_count, 'projection_paths': projections,
            'mtp_serialized_tensor_count': sum(k.startswith('mtp.') for k in index),
            'mtp_logical_tensor_count': sum(k.startswith('mtp.') and not k.endswith(('.weight_scale', '.weight_scale_2')) for k in index),
            'expected_projection_count': len(projections)}


def _resident_bytes() -> int:
    # ru_maxrss is bytes on macOS; this is a conservative high-water mark,
    # not an instantaneous footprint or a disjoint addition to MLX counters.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == 'darwin' else rss * 1024)


class QwenHybridAdapter:
    def __init__(self, model: Any, preflight: dict, memory_limit: int):
        self.model = model
        self.preflight = preflight
        self.memory_limit = memory_limit
        self.projections = {}
        layers = model.language_model.layers
        if len(layers) != 64:
            raise FidelityError('Loaded layer count mismatch')
        for path in preflight['projection_paths']:
            parts = path.split('.')
            layer = layers[int(parts[3])]
            module = getattr(getattr(layer, parts[4]), parts[5])
            if getattr(module, 'mode', None) != 'nvfp4' or not hasattr(module, 'global_scale'):
                raise FidelityError('Projection lost NVFP4 global-scale semantics')
            if module.weight.dtype.__str__() != 'mlx.core.uint32':
                raise FidelityError('Projection no longer has packed U32 weights')
            self.projections[path] = module
        if len(self.projections) != 128:
            raise FidelityError('Expected exactly 128 editable projections')

    def _projection_slot(self, path):
        parts = path.split('.')
        layer = self.model.language_model.layers[int(parts[3])]
        return getattr(layer, parts[4]), parts[5]

    def set_edits(self, directions: dict, strengths: float | dict = 0.) -> None:
        """Atomically validate wrappers, then attach them to original base handles."""
        from .edits import DirectionalProjection
        if not directions or set(directions) - set(self.projections):
            raise FidelityError('Directions must name known output projections')
        if isinstance(strengths, dict) and set(strengths) != set(directions):
            raise FidelityError('Per-projection strengths must match directions')
        wrappers = {}
        for path, direction in directions.items():
            strength = strengths[path] if isinstance(strengths, dict) else strengths
            wrapper = DirectionalProjection(self.projections[path], direction, strength)
            if wrapper.direction.shape != (5120,):
                raise FidelityError('Projection directions must have width 5120')
            wrappers[path] = wrapper
        self.reset_edits()
        for path, wrapper in wrappers.items():
            parent, attribute = self._projection_slot(path)
            setattr(parent, attribute, wrapper)

    def reset_edits(self) -> None:
        """Restore exact immutable module handles; no reloading or weight copies."""
        for path, base in self.projections.items():
            parent, attribute = self._projection_slot(path)
            setattr(parent, attribute, base)

    def collect_last_hidden(self, token_ids: list[int]) -> dict[int, Any]:
        """Collect 65 last-token residual vectors with per-layer eval boundaries.

        Position 0 is the input embedding; positions 1..64 are unnormalized
        decoder-layer outputs. Only one unpadded prompt is supported. Temporary
        wrappers are removed even on an exception, and base forward code runs.
        """
        import mlx.core as mx
        import mlx.nn as nn
        values = {}
        class Observer(nn.Module):
            def __init__(self, base, position):
                super().__init__()
                self.base = base
                self.position = position
                self.is_linear = base.is_linear
                if base.is_linear:
                    self.linear_attn = base.linear_attn
                else:
                    self.self_attn = base.self_attn
            def __call__(self, x, *args, **kwargs):
                if self.position == 1:
                    values[0] = mx.mean(x[:, -1, :].astype(mx.float32), axis=0)
                    mx.eval(values[0])
                result = self.base(x, *args, **kwargs)
                values[self.position] = mx.mean(result[:, -1, :].astype(mx.float32), axis=0)
                mx.eval(values[self.position])
                if _resident_bytes() > self_limit or mx.get_active_memory() > self_limit:
                    raise FidelityError('Memory budget exceeded during residual collection')
                return result
        self_limit = self.memory_limit
        layers = self.model.language_model.model.layers
        originals = list(layers)
        try:
            for i, base in enumerate(originals):
                layers[i] = Observer(base, i + 1)
            self.last_logits(token_ids)
        finally:
            layers[:] = originals
        if set(values) != set(range(65)):
            raise FidelityError('Residual collection did not visit all layers')
        return values

    def last_logits(self, token_ids: list[int]):
        """Single unpadded prompt; fresh recurrent/KV and positional state every call."""
        if not token_ids or len(token_ids) > 4096:
            raise FidelityError('Pilot prompts must have 1–4096 tokens')
        import mlx.core as mx
        language = self.model.language_model
        for field in ('_rope_deltas', '_position_ids'):
            if hasattr(language, field):
                setattr(language, field, None)
        cache = language.make_cache()
        output = language(mx.array([token_ids], dtype=mx.int32), cache=cache, logits_to_keep=1)
        logits = output.logits[:, -1, :]
        mx.eval(logits)
        if _resident_bytes() > self.memory_limit or mx.get_active_memory() > self.memory_limit:
            raise FidelityError('Process/MLX memory budget exceeded')
        return logits


def load_export(path: Path, *, memory_limit: int = 34 * GIB,
                cache_limit: int = GIB) -> QwenHybridAdapter:
    """Load the validated exact export; call only after the no-op serving gate.

    Uses upstream sanitization and strict loading, but maps packed tensors
    directly to native names so ModelOpt's U8 importer cannot reinterpret U32.
    MTP stays in the export and is intentionally absent from the search runtime.
    """
    path = Path(path).resolve()
    report = preflight_export(path)
    if report['shard_bytes'] + 4 * GIB > memory_limit:
        raise FidelityError('Insufficient configured headroom for packed model and runtime')
    import mlx.core as mx
    from mlx_vlm.models.qwen3_5.config import ModelConfig
    from mlx_vlm.models.qwen3_5.qwen3_5 import Model
    from mlx_vlm.utils import load_model_weights
    if not mx.metal.is_available():
        raise FidelityError('Metal GPU is required; CPU fallback is unsupported')
    recommended = mx.device_info()['max_recommended_working_set_size']
    if memory_limit > recommended:
        raise FidelityError('Requested budget exceeds recommended GPU working set')
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(memory_limit)  # MLX documents this as guidance, not a hard cap.
    mx.set_cache_limit(cache_limit)
    config = copy.deepcopy(report['config'])
    config.pop('quantization_config', None)
    config['text_config'].pop('quantization_config', None)
    config['quantization'] = {'mode': 'nvfp4', 'bits': 4, 'group_size': 16}
    weights = {}
    for shard in sorted(set(report['weight_map'].values())):
        values = mx.load(str(path / shard))
        for name, value in values.items():
            if name.startswith('mtp.'):
                continue
            destination = name[:-len('.weight_scale')] + '.scales' if name.endswith('.weight_scale') else name
            weights[destination] = value
        mx.eval(list(weights.values()))
        if _resident_bytes() > memory_limit or mx.get_active_memory() > memory_limit:
            raise FidelityError('Memory budget exceeded while reading packed shards')
    model = Model(ModelConfig.from_dict(config))
    weights = model.sanitize(weights)
    model = load_model_weights(model, config, weights, path, strict=True, lazy=False)
    model.freeze()
    if _resident_bytes() > memory_limit or mx.get_active_memory() > memory_limit:
        raise FidelityError('Memory budget exceeded after strict model load')
    return QwenHybridAdapter(model, report, memory_limit)


def runtime_versions() -> dict[str, str]:
    result = {}
    for package in ('mlx', 'mlx-vlm'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = 'not installed'
    return result
