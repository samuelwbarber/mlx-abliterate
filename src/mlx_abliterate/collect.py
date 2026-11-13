"""Bounded online last-nonpadding-token means; no prompt activations retained."""
from __future__ import annotations

from .edits import _require_mlx, mx


class OnlineLastTokenMean:
    """One float32 accumulator of `width` elements, independent of sequence count.

    Input is [batch, sequence, width]. `lengths` refers to right-padded sequences;
    left-padding requires explicit `positions` instead. Without either, the last
    sequence position is used. Call separately at each intended collection site.
    """
    def __init__(self, width: int):
        _require_mlx()
        if type(width) is not int or width < 1:
            raise ValueError('width must be a positive integer')
        self.width = width
        self.count = 0
        self._sum = mx.zeros((width,), dtype=mx.float32)
        mx.eval(self._sum)

    @property
    def accumulator_bytes(self) -> int:
        return self.width * 4

    def update(self, hidden, *, lengths=None, positions=None) -> None:
        if hidden.ndim != 3 or hidden.shape[-1] != self.width:
            raise ValueError('hidden must have shape [batch, sequence, width]')
        batch, sequence, _ = hidden.shape
        if batch < 1 or sequence < 1 or not mx.issubdtype(hidden.dtype, mx.floating):
            raise ValueError('hidden must be a nonempty floating tensor')
        if lengths is not None and positions is not None:
            raise ValueError('provide lengths or positions, not both')
        if lengths is not None:
            lengths = list(lengths)
            if len(lengths) != batch or any(type(n) is not int or not 1 <= n <= sequence for n in lengths):
                raise ValueError('lengths must contain one valid positive length per row')
            indices = [n - 1 for n in lengths]
        elif positions is not None:
            indices = list(positions)
            if len(indices) != batch or any(type(n) is not int or not 0 <= n < sequence for n in indices):
                raise ValueError('positions must contain one valid index per row')
        else:
            indices = [sequence - 1] * batch
        selected = hidden[mx.arange(batch), mx.array(indices), :].astype(mx.float32)
        if not bool(mx.all(mx.isfinite(selected)).item()):
            raise ValueError('selected hidden states must be finite')
        updated = self._sum + mx.sum(selected, axis=0)
        mx.eval(updated)
        if not bool(mx.all(mx.isfinite(updated)).item()):
            raise ValueError('float32 accumulator overflow')
        self._sum = updated
        self.count += batch

    def mean(self):
        if self.count == 0:
            raise ValueError('cannot compute mean before collecting samples')
        result = self._sum / self.count
        mx.eval(result)
        return result

    def reset(self) -> None:
        self._sum = mx.zeros((self.width,), dtype=mx.float32)
        mx.eval(self._sum)
        self.count = 0
