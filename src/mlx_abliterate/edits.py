"""Original rank-one projection edits; MLX is an optional dependency.

These edits leave the wrapped weight representation untouched. They implement
weight-only directional ablation, not row normalization or Heretic's ARA method.
"""
from __future__ import annotations

import math

try:
    import mlx.core as mx
    import mlx.nn as nn
except ModuleNotFoundError:
    mx = None
    nn = None


def _require_mlx():
    if mx is None:
        raise RuntimeError('MLX is required for edits; install the mlx extra on Apple Silicon')


def _strength(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('strength must be a finite number')
    if not 0 <= value <= 2:
        raise ValueError('strength must be within the MVP interval [0, 2]')
    return float(value)


def normalize_direction(direction):
    """Return a finite, nonzero, one-dimensional unit direction in float32."""
    _require_mlx()
    direction = mx.array(direction, dtype=mx.float32)
    if direction.ndim != 1 or direction.size == 0:
        raise ValueError('direction must be a nonempty vector')
    if not bool(mx.all(mx.isfinite(direction)).item()):
        raise ValueError('direction must be finite')
    # Scale first to avoid overflow for finite large input vectors.
    largest = mx.max(mx.abs(direction))
    if largest.item() == 0:
        raise ValueError('direction must be nonzero')
    scaled = direction / largest
    normalized = scaled / mx.sqrt(mx.sum(scaled * scaled))
    mx.eval(normalized)
    return normalized


def _correct(output, direction, strength: float, bias=None):
    if output.ndim == 0 or output.shape[-1] != direction.shape[0]:
        raise ValueError('output width must match direction')
    if not mx.issubdtype(output.dtype, mx.floating):
        raise ValueError('output must be floating point')
    value = output.astype(mx.float32)
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != output.shape[-1]:
            raise ValueError('ordinary layer bias must match output width')
        value = value - bias.astype(mx.float32)
    coefficient = mx.sum(value * direction, axis=-1, keepdims=True)
    correction = (strength * coefficient * direction).astype(output.dtype)
    return output - correction


def directional_output(output, direction, strength: float, bias=None):
    """Evaluate y - a*r*(r.T*(y-b)); quantization offsets are NOT bias.

    Zero strength returns the original output handle without extra computation.
    The direction is normalized for nonzero edits and reductions use float32.
    """
    _require_mlx()
    strength = _strength(strength)
    if strength == 0:
        return output
    return _correct(output, normalize_direction(direction), strength, bias)


class DirectionalProjection(nn.Module if nn is not None else object):
    """Wrap one projection without copying/dequantizing its packed weights.

    `base.bias`, when present, is the ordinary additive layer bias. MLX affine
    quantization's `base.biases` is intentionally not subtracted. Attach this
    wrapper at the projection output before the parent module's residual add.
    Changing strength invalidates all downstream model caches; the caller must
    discard them before another evaluation. This wrapper owns no KV cache.
    """
    def __init__(self, base, direction, strength: float = 0):
        _require_mlx()
        super().__init__()
        if not callable(base):
            raise ValueError('base must be a callable projection')
        self.base = base
        self.direction = normalize_direction(direction)
        self.strength = _strength(strength)

    def set_strength(self, strength: float) -> None:
        self.strength = _strength(strength)

    def __call__(self, inputs):
        output = self.base(inputs)
        if self.strength == 0:
            return output
        return _correct(output, self.direction, self.strength, getattr(self.base, 'bias', None))
