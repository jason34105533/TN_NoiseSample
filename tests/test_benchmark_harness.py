"""Tests for the benchmarking harness's GPU device metadata reporting."""
import pytest

from benchmarks.timing import gpu_device_info
from tests.conftest import requires_gpu


def test_gpu_device_info_none_without_gpu(monkeypatch):
    import benchmarks.timing as timing_mod

    monkeypatch.setattr(timing_mod, "HAS_CUPY", False)
    assert gpu_device_info() is None


@requires_gpu
def test_gpu_device_info_reports_real_device():
    info = gpu_device_info()
    assert info is not None
    assert isinstance(info["gpu_device_name"], str) and info["gpu_device_name"]
    assert isinstance(info["gpu_memory_total_bytes"], int)
    assert info["gpu_memory_total_bytes"] > 0


def test_run_benchmark_record_has_gpu_fields_when_no_gpu():
    from benchmarks.run_benchmark import run_benchmark

    results = run_benchmark(
        n=3, g=3, num_instances=1, num_shots=2, num_error_sets=2,
        batch_size=2, final_batch_size=3, fast=True, use_gpu=False,
    )
    assert len(results) == 1
    assert results[0]["gpu_device_name"] is None
    assert results[0]["gpu_memory_total_bytes"] is None
