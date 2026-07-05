"""Timing utilities for benchmarking simulation throughput."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


class Timer:
    """
    Wall-clock timer with optional CUDA event synchronization.

    On GPU: uses CUDA events (accurate GPU time, excludes CPU overhead).
    On CPU: uses time.perf_counter.
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and HAS_CUPY
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if self.use_gpu:
            self._start = cp.cuda.Event()
            self._end = cp.cuda.Event()
            self._start.record()
        else:
            self._cpu_start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.use_gpu:
            self._end.record()
            self._end.synchronize()
            self.elapsed_ms = cp.cuda.get_elapsed_time(self._start, self._end)
        else:
            self.elapsed_ms = (time.perf_counter() - self._cpu_start) * 1000.0

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ms / 1000.0


@contextmanager
def timed(use_gpu: bool = True):
    """Context manager returning a Timer after the block exits."""
    t = Timer(use_gpu=use_gpu)
    with t:
        yield t
