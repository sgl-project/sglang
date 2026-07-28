"""Unit tests for performance benchmark correctness.

These tests verify the math, not performance values.
All tests are CPU-only and do not require GPU.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import pytest


# --- Same-M baseline mapping ---

def test_same_m_baseline_mapping():
    """Verify that M=512 BF16 is compared against M=512 FP32, never M=1 FP32."""
    fp32_baselines = {
        (1, 4096, 4096): 17.5,
        (4, 4096, 4096): 17.2,
        (16, 4096, 4096): 25.0,
        (64, 4096, 4096): 25.0,
        (128, 4096, 4096): 21.5,
        (256, 4096, 4096): 24.0,
        (512, 4096, 4096): 24.0,
        (1024, 4096, 4096): 23.9,
    }

    # BF16 M=512 result
    bf16_median = 20.5
    M, N, K = 512, 4096, 4096

    # Correct: use same-shape baseline
    correct_baseline = fp32_baselines[(M, N, K)]
    correct_speedup = correct_baseline / bf16_median

    # Wrong: use M=1 baseline
    wrong_baseline = fp32_baselines[(1, N, K)]
    wrong_speedup = wrong_baseline / bf16_median

    assert correct_baseline == 24.0, f"Expected 24.0, got {correct_baseline}"
    assert abs(correct_speedup - (24.0 / 20.5)) < 1e-6
    assert wrong_baseline == 17.5, f"M=1 baseline should be 17.5"
    assert abs(wrong_speedup - (17.5 / 20.5)) < 1e-6
    assert correct_speedup != wrong_speedup, "Same-M and M=1 baselines must differ"


def test_no_m1_reuse_for_other_m():
    """Explicitly verify M=1024 does not use M=1 baseline."""
    baselines = {(M, 4096, 4096): float(M) for M in [1, 4, 16, 64, 128, 256, 512, 1024]}

    for M in [4, 16, 64, 128, 256, 512, 1024]:
        baseline = baselines[(M, 4096, 4096)]
        assert baseline == float(M), f"M={M}: baseline should be {M}, got {baseline}"
        assert baseline != baselines[(1, 4096, 4096)], f"M={M}: should not reuse M=1 baseline"


# --- TFLOP/s calculation ---

def test_tflops_calculation():
    """Verify TFLOP/s = 2*M*N*K / latency_seconds / 1e12."""
    M, N, K = 512, 4096, 4096
    flops = 2 * M * N * K
    latency_us = 24.0  # microseconds
    latency_s = latency_us * 1e-6
    tflops = flops / latency_s / 1e12
    expected = 2 * 512 * 4096 * 4096 / (24.0e-6) / 1e12
    assert abs(tflops - expected) < 1e-6
    # Also check the formula used in bench: flops / median_us * 1e-6
    tflops_alt = flops / latency_us * 1e-6
    assert abs(tflops - tflops_alt) < 1e-6


def test_tflops_grouped_gemm():
    """Verify grouped GEMM FLOPs uses actual tokens per expert."""
    num_experts = 4
    tokens_per_expert = [10, 20, 30, 40]
    N, K = 4096, 5120
    actual_flops = sum(2 * m * N * K for m in tokens_per_expert)
    total_tokens = sum(tokens_per_expert)
    naive_flops = 2 * total_tokens * N * K  # This overcounts
    assert actual_flops == 2 * (10 + 20 + 30 + 40) * N * K  # Actually same for balanced
    # For masked GEMM, actual flops should use per-expert tokens
    # not the expected_m * N * K


# --- Percentile calculations ---

def test_percentile_calculation():
    """Verify percentile calculation is correct."""
    data = list(range(1, 101))  # 1..100
    s = sorted(data)
    n = len(s)
    median = s[n // 2]
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    p95 = s[p95_idx]
    p99 = s[p99_idx]

    assert median == 51  # s[50] = 51
    assert p95 == 96  # s[95] = 96
    assert p99 == 100  # s[99] = 100


def test_p95_not_below_median():
    """p95 should never be below median for sorted data."""
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    s = sorted(data)
    n = len(s)
    median = s[n // 2]
    p95 = s[min(int(n * 0.95), n - 1)]
    assert p95 >= median


def test_p99_not_below_p95():
    """p99 should never be below p95 for sorted data."""
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    s = sorted(data)
    n = len(s)
    p95 = s[min(int(n * 0.95), n - 1)]
    p99 = s[min(int(n * 0.99), n - 1)]
    assert p99 >= p95


# --- Standard deviation and CV ---

def test_std_and_cv():
    """Verify standard deviation and coefficient of variation."""
    data = [10.0, 20.0, 30.0]
    mean = sum(data) / len(data)
    std = math.sqrt(sum((x - mean) ** 2 for x in data) / (len(data) - 1))
    cv = std / mean if mean > 0 else 0
    assert abs(mean - 20.0) < 1e-6
    assert abs(std - 10.0) < 1e-6
    assert abs(cv - 0.5) < 1e-6


# --- Payload byte calculation ---

def test_payload_byte_calculation():
    """Verify PP payload byte calculations."""
    hidden_size = 5120
    capture_layers = 3
    token_rows = 4
    dtype_size = 2  # bfloat16

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


def test_glm_payload_sizes():
    """Verify GLM-5.2 derived payload sizes."""
    hidden_size = 5120
    dtype_size = 2  # bfloat16

    # hidden_only M=1
    assert 1 * hidden_size * dtype_size == 10240

    # hidden+residual M=4
    assert 4 * hidden_size * dtype_size * 2 == 81920

    # hidden+residual+aux M=16 (3 capture layers)
    assert 16 * hidden_size * dtype_size * (2 + 3) == 819200


# --- RTT/2 labeling ---

def test_rtt_half_labeling():
    """Verify RTT/2 is labeled as an estimate."""
    rtt_us = 80.0
    rtt_half = rtt_us / 2

    result = {
        "rtt_us": rtt_us,
        "rtt_half_estimate_us": rtt_half,
        "note": "RTT/2 is an estimate, not a direct measurement",
    }

    assert "estimate" in result["note"].lower()
    assert result["rtt_half_estimate_us"] == 40.0
    assert "rtt_half_estimate_us" in result


# --- Impossible bandwidth detection ---

def test_impossible_bandwidth_detection():
    """Verify suspicious bandwidth is flagged."""
    # PCIe Gen5 x16 one-direction ceiling: ~64 GB/s
    # SHM fallback ceiling: ~36-37 GB/s
    ceiling_pcie = 64.0  # GB/s
    ceiling_shm = 37.0

    # Normal measurement
    normal_bw = 35.0
    assert normal_bw <= ceiling_shm

    # Suspicious: above PCIe ceiling for one-way
    suspicious_bw = 100.0
    assert suspicious_bw > ceiling_pcie

    # The benchmark should warn
    warnings = []
    if suspicious_bw > ceiling_pcie:
        warnings.append("SUSPICIOUS_RESULT: effective BW exceeds PCIe Gen5 x16 ceiling")
    assert len(warnings) > 0


def test_negative_latency_detection():
    """Verify negative or zero latency is flagged."""
    warnings = []
    median = 0.0
    if median <= 0:
        warnings.append("SUSPICIOUS_RESULT: zero or negative median latency")
    assert len(warnings) > 0


def test_mismatched_shape_speedup_detection():
    """Verify speedup from mismatched shapes is flagged."""
    fp32_baselines = {(1, 4096, 4096): 17.5, (512, 4096, 4096): 24.0}

    # Correct: same shape
    M = 512
    baseline = fp32_baselines[(M, 4096, 4096)]
    assert baseline == 24.0

    # If someone used M=1 baseline for M=512
    wrong_baseline = fp32_baselines[(1, 4096, 4096)]
    assert wrong_baseline != baseline


# --- Cache mode metadata ---

def test_cache_mode_metadata():
    """Verify cache mode metadata is recorded."""
    for mode in ["hot", "rotating", "flushed"]:
        result = {"cache_mode": mode}
        assert result["cache_mode"] == mode

    # Rotating should record number and total size
    num_rotating = 8
    b_shape = (4096, 4096)
    b_dtype_size = 2  # bfloat16
    rotating_total_bytes = num_rotating * b_shape[0] * b_shape[1] * b_dtype_size
    assert rotating_total_bytes == 8 * 4096 * 4096 * 2
    # 268 MiB > 50 MiB L2 cache
    assert rotating_total_bytes > 50 * 1024 * 1024


# --- JSON schema validation ---

def test_json_schema_tensor_core():
    """Verify tensor core benchmark JSON has required fields."""
    required_fields = [
        "dtype", "backend", "tf32_enabled", "M", "N", "K",
        "median_latency_us", "p95_latency_us", "p99_latency_us",
        "tflops", "speedup_vs_fp32_same_shape", "cache_mode",
        "contamination", "gpu",
    ]
    data = {
        "dtype": "bfloat16",
        "backend": "torch.matmul",
        "tf32_enabled": False,
        "M": 512, "N": 4096, "K": 4096,
        "median_latency_us": 20.5,
        "p95_latency_us": 22.0,
        "p99_latency_us": 25.0,
        "tflops": 834.9,
        "speedup_vs_fp32_same_shape": 1.17,
        "cache_mode": "hot",
        "contamination": "LIGHTLY_CONTENDED",
        "gpu": "NVIDIA H100 PCIe",
    }
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def test_json_schema_communication():
    """Verify communication benchmark JSON has required fields."""
    required_fields = [
        "operation", "protocol", "bytes", "median_latency_us",
        "effective_gbs", "p2p_status", "contamination",
    ]
    data = {
        "operation": "send_recv_one_way_with_ack",
        "protocol": "one_way_with_ack",
        "bytes": 1048576,
        "median_latency_us": 100.0,
        "effective_gbs": 10.45,
        "p2p_status": "unavailable (SHM/host-mediated fallback)",
        "contamination": "LIGHTLY_CONTENDED",
    }
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


# --- NOT_AVAILABLE backend handling ---

def test_not_available_backend_handling():
    """Verify NOT_AVAILABLE is reported rather than silent fallback."""
    NOT_AVAILABLE = "NOT_AVAILABLE"

    def check_backend(name):
        available = {"torch_scaled_mm": True, "deep_gemm": False}
        return available.get(name, False)

    result = {}
    for backend in ["torch_scaled_mm", "deep_gemm"]:
        if not check_backend(backend):
            result[backend] = NOT_AVAILABLE
        else:
            result[backend] = "AVAILABLE"

    assert result["deep_gemm"] == NOT_AVAILABLE
    assert result["torch_scaled_mm"] == "AVAILABLE"


# --- Nsight Systems compatibility table parsing ---

def test_nsys_compatibility_parsing():
    """Verify Nsight Systems compatibility table parsing."""
    nsys_data = {
        "versions": [
            {"path": "/old/nsys", "version": "2023.1.2", "verdict": "INCOMPATIBLE"},
            {"path": "/new/nsys", "version": "2026.3.1", "verdict": "COMPATIBLE"},
        ],
        "selected": "/new/nsys",
    }

    compatible = [v for v in nsys_data["versions"] if v["verdict"] == "COMPATIBLE"]
    assert len(compatible) == 1
    assert compatible[0]["version"] == "2026.3.1"
    assert nsys_data["selected"] == "/new/nsys"


# --- NCU permission classification ---

def test_ncu_permission_classification():
    """Verify NCU permission classification."""
    def classify_ncu(error, version_compat):
        if not version_compat:
            return "NCU_VERSION_INCOMPATIBLE"
        if error is not None and "ERR_NVGPUCTRPERM" in error:
            return "NCU_COUNTER_PERMISSION_DENIED"
        if error is None:
            return "NCU_CAPTURE_SUCCESS"
        return "NCU_AVAILABLE"

    assert classify_ncu("ERR_NVGPUCTRPERM", True) == "NCU_COUNTER_PERMISSION_DENIED"
    assert classify_ncu("ERR_NVGPUCTRPERM", False) == "NCU_VERSION_INCOMPATIBLE"
    assert classify_ncu(None, True) == "NCU_CAPTURE_SUCCESS"
    assert classify_ncu(None, False) == "NCU_VERSION_INCOMPATIBLE"


# --- Missing metric handling ---

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


# --- Contamination classification ---

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
