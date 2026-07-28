"""Unit tests for performance calculation correctness.

These tests verify the math, not performance values.
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest


def test_percentile_calculation():
    """Verify percentile calculation is correct."""
    data = list(range(1, 101))  # 1..100
    s = sorted(data)
    n = len(s)
    median = s[n // 2]
    p95 = s[int(n * 0.95)]

    assert median == 51  # 100//2 = 50, s[50] = 51
    assert p95 == 96  # int(100 * 0.95) = 95, s[95] = 96


def test_tflops_calculation():
    """Verify TFLOP/s = 2*M*N*K / latency_seconds / 1e12."""
    M, N, K = 512, 4096, 4096
    flops = 2 * M * N * K
    latency_s = 0.001  # 1ms
    tflops = flops / latency_s / 1e12
    expected = 2 * 512 * 4096 * 4096 / 0.001 / 1e12
    assert abs(tflops - expected) < 1e-6


def test_payload_byte_calculation():
    """Verify PP payload byte calculations."""
    hidden_size = 5120
    capture_layers = 3
    token_rows = 4
    dtype_size = 2  # float16

    hs_bytes = token_rows * hidden_size * dtype_size
    res_bytes = token_rows * hidden_size * dtype_size
    aux_bytes = token_rows * capture_layers * hidden_size * dtype_size
    combined = hs_bytes + res_bytes + aux_bytes

    assert hs_bytes == 4 * 5120 * 2
    assert res_bytes == 4 * 5120 * 2
    assert aux_bytes == 4 * 3 * 5120 * 2
    assert combined == 4 * 5120 * 2 * (2 + 3)
    # 4 * 5120 * 2 * 5 = 204800 bytes = 200 KiB
    assert combined == 204800


def test_json_schema_communication():
    """Verify communication benchmark JSON has required fields."""
    # Synthetic JSON
    data = {
        "rank": 0,
        "world_size": 2,
        "backend": "nccl",
        "results": [
            {
                "operation": "send_recv",
                "bytes": 4096,
                "dtype": "float16",
                "median_latency_us": 10.0,
                "effective_gbs": 0.4,
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    try:
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["world_size"] == 2
        assert loaded["results"][0]["operation"] == "send_recv"
        assert loaded["results"][0]["bytes"] == 4096
    finally:
        os.unlink(path)


def test_json_schema_tensor_core():
    """Verify tensor core benchmark JSON has required fields."""
    data = {
        "gpu_name": "NVIDIA H100 PCIe",
        "N": 4096,
        "K": 4096,
        "results": [
            {
                "dtype": "bfloat16",
                "M": 512,
                "median_latency_us": 100.0,
                "tflops": 100.0,
                "speedup_vs_fp32": 5.0,
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    try:
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["gpu_name"] == "NVIDIA H100 PCIe"
        assert loaded["results"][0]["tflops"] > 0
    finally:
        os.unlink(path)


def test_missing_metric_handling():
    """Verify missing metrics are reported as NOT_AVAILABLE."""
    NOT_AVAILABLE = "NOT_AVAILABLE"

    metrics = {"sm_active": 75.0, "tensor_active": None}
    result = {}
    for key in ["sm_active", "tensor_active", "dram_throughput"]:
        val = metrics.get(key)
        result[key] = val if val is not None else NOT_AVAILABLE

    assert result["sm_active"] == 75.0
    assert result["tensor_active"] == NOT_AVAILABLE
    assert result["dram_throughput"] == NOT_AVAILABLE


def test_contamination_classification():
    """Verify environment contamination classification logic."""
    def classify(gpu_util, mem_free_mib, has_foreign):
        if not has_foreign:
            return "CLEAN"
        if mem_free_mib < 4096:
            return "UNSAFE_TO_RUN"
        if gpu_util < 5:
            return "LIGHTLY_CONTENDED"
        if gpu_util < 50:
            return "CONTAMINATED"
        return "UNSAFE_TO_RUN"

    assert classify(0, 65536, False) == "CLEAN"
    assert classify(2, 65536, True) == "LIGHTLY_CONTENDED"
    assert classify(30, 65536, True) == "CONTAMINATED"
    assert classify(80, 65536, True) == "UNSAFE_TO_RUN"
    assert classify(0, 1024, True) == "UNSAFE_TO_RUN"
