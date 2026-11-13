"""Process and MLX memory limits without adding overlapping memory counters."""
from __future__ import annotations

from dataclasses import dataclass
import math
import resource
import sys
import time

GIB = 1024 ** 3


class MemoryBudgetExceeded(RuntimeError):
    pass


def snapshot(mx=None) -> dict:
    # Darwin reports bytes; Linux reports KiB. This is a high-water RSS, not
    # Apple physical footprint and not an estimate of current available RAM.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"monotonic_seconds": time.monotonic(),
              "process_peak_rss_bytes": int(peak * (1 if sys.platform == "darwin" else 1024))}
    if mx is not None:
        report.update(mlx_active_bytes=mx.get_active_memory(),
                      mlx_cache_bytes=mx.get_cache_memory(),
                      mlx_peak_bytes=mx.get_peak_memory())
    return report


@dataclass
class MemoryGuard:
    """Phase-boundary admission guard, not an OOM-proof external watchdog.

    MLX's allocator limit and process high-water RSS are independent checks.
    This guard cannot observe system compression/pressure or other processes.
    """
    process_limit_bytes: int = 34 * GIB
    allocator_limit_bytes: int = 32 * GIB
    cache_limit_bytes: int = GIB

    def __post_init__(self):
        for value in (self.process_limit_bytes, self.allocator_limit_bytes, self.cache_limit_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("Memory limits must be positive integer byte counts")
        if not self.cache_limit_bytes < self.allocator_limit_bytes <= self.process_limit_bytes:
            raise ValueError("Require cache < allocator <= process budget")

    def configure(self, mx):
        if not mx.metal.is_available():
            raise RuntimeError("Metal GPU required; CPU fallback is not supported")
        info = mx.metal.device_info()
        recommended = info.get("max_recommended_working_set_size")
        effective = min(self.allocator_limit_bytes, recommended) if recommended else self.allocator_limit_bytes
        mx.set_memory_limit(effective)
        mx.set_cache_limit(min(self.cache_limit_bytes, effective // 8))
        return {"allocator_limit_bytes": effective, "device": info}

    def check(self, mx=None) -> dict:
        observed = snapshot(mx)
        if observed["process_peak_rss_bytes"] > self.process_limit_bytes:
            raise MemoryBudgetExceeded("Process peak RSS exceeded configured budget")
        if observed.get("mlx_active_bytes", 0) > self.allocator_limit_bytes:
            raise MemoryBudgetExceeded("MLX active allocations exceeded configured budget")
        return observed


def estimate_remaining(trial_seconds: list[float], remaining_trials: int) -> dict:
    """Measured trial-only forecast; excludes unmeasured setup/export phases."""
    import statistics
    if type(remaining_trials) is not int or remaining_trials < 0:
        raise ValueError("remaining_trials must be nonnegative")
    if len(trial_seconds) < 3 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                    or not math.isfinite(x) or x <= 0 for x in trial_seconds):
        raise ValueError("At least three finite positive trial timings are required")
    return {"remaining_trials": remaining_trials,
            "estimated_trial_seconds": statistics.median(trial_seconds) * remaining_trials,
            "observed_range_seconds": [min(trial_seconds) * remaining_trials,
                                       max(trial_seconds) * remaining_trials],
            "scope": "Trial execution only; observed range is not a confidence interval."}
