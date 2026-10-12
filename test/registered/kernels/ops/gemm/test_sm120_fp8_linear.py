"""Correctness and dispatch tests for SM12x small-M FP8 linear."""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.gemm import try_sm120_fp8_linear
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear_bmm_flashinfer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)):
    pytest.skip(
        "SM120 FP8 linear dispatch requires CUDA SM120", allow_module_level=True
    )


ALL_M = (1, 2, 4, 8, 9)

# M=1 uses the streaming GEMV whenever its broad shape gate accepts the call;
# qualified M>=2 cases and the oversized M=1 gate/up projection use KDA.
SUPPORTED_CONFIGS = [
    ("Qwen3.8-27B", "attn-qkv", 8192, 5120, ALL_M),
    ("Qwen3.8-27B", "gdn-in", 16384, 5120, (1, 2, 4, 8)),
    ("Qwen3.8-27B", "out", 5120, 6144, ALL_M),
    ("Qwen3-8B/Llama-3.1-8B", "qkv", 6144, 4096, (1,)),
    ("Qwen3-8B/Llama-3.1-8B", "o", 4096, 4096, ALL_M),
    ("Qwen3-14B", "qkv", 7168, 5120, ALL_M),
    ("Qwen3-14B", "o", 5120, 5120, ALL_M),
    ("Qwen3-14B", "gate-up", 34816, 5120, ALL_M),
    ("Nemotron-3-Super", "shared-up", 5376, 4096, (1,)),
]
SUPPORTED_SHAPES = [
    pytest.param(m, n, k, id=f"{model}-{op}-m{m}".lower())
    for model, op, n, k, supported_m in SUPPORTED_CONFIGS
    for m in supported_m
]

# These M>1 shapes are numerically valid but missed either the KDA accuracy or
# performance gate. Their M=1 variants may still use the streaming GEMV.
UNSUPPORTED_CONFIGS = [
    pytest.param((3,), 8192, 5120, id="unsupported-m"),
    pytest.param((9,), 16384, 5120, id="qwen38-gdn-in"),
    pytest.param((2, 4, 8, 9), 5376, 4096, id="nemotron-shared-up"),
    pytest.param((2, 4, 8, 9), 4096, 5376, id="nemotron-shared-down"),
    pytest.param((8,), 6144, 4096, id="qwen3-8b-qkv"),
]


def _make_inputs(m: int, n: int, k: int, seed: int = 0):
    torch.manual_seed(seed)
    input_scale = torch.tensor(0.025, dtype=torch.float32, device="cuda")
    weight_scale = torch.tensor(0.02, dtype=torch.float32, device="cuda")
    input = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = (
        torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        .mul_(32)
        .to(torch.float8_e4m3fn)
        .t()
    )
    return input, weight, weight_scale, input_scale


def _reference(args):
    return apply_fp8_linear_bmm_flashinfer(*args)


def _assert_matches_flashinfer(actual, expected):
    # CUTLASS and cuBLAS may accumulate in a different order. On SM120 the
    # observed differences are sparse and stay within standard BF16 tolerance.
    torch.testing.assert_close(actual, expected)


def _run_sm120(args, *, m=None, vector_scales=False, bias=None):
    input, weight, weight_scale, input_scale = args
    if m is not None:
        input = input[:m]
    output_scale = input_scale * weight_scale
    if vector_scales:
        input_scale = input_scale.reshape(1)
        output_scale = output_scale.reshape(1)
    return try_sm120_fp8_linear(
        input,
        weight,
        input_scale,
        output_scale,
        bias,
    )


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("m,n,k", SUPPORTED_SHAPES)
def test_supported_shapes_match_flashinfer(m: int, n: int, k: int, seed: int):
    args = _make_inputs(m, n, k, seed)
    expected = _reference(args)
    actual = _run_sm120(args)
    assert actual is not None
    _assert_matches_flashinfer(actual, expected)


@pytest.mark.parametrize("m", ALL_M)
def test_cuda_graph_replay_uses_current_input(m: int):
    args = _make_inputs(m, 8192, 5120)
    input, weight, weight_scale, input_scale = args
    output_scale = input_scale * weight_scale

    # Compile the selected provider before capture; JIT compilation is not
    # CUDA Graph safe and this test must also work when selected in isolation.
    warmup = _run_sm120(args)
    assert warmup is not None
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = try_sm120_fp8_linear(input, weight, input_scale, output_scale)
        assert actual is not None

    # A replay must consume the new activation rather than the values present
    # during capture. This also exercises the quantize-to-GEMM dependency.
    torch.manual_seed(17)
    input.copy_(torch.randn_like(input).mul_(8))
    graph.replay()
    expected = _reference(args)
    _assert_matches_flashinfer(actual, expected)


@pytest.mark.parametrize("m", (2, 4, 8))
def test_saturated_real_activation_range_matches_flashinfer(m: int):
    args = _make_inputs(m, 16384, 5120, seed=11)
    input = args[0]
    # Static ModelOpt scales can expose both saturation and FP8 rounding
    # boundaries in live GDN activations. This distribution reproduces the
    # class of mismatch that the former fused quantizer caused in E2E decode.
    input.mul_(32)
    input[:, :8] = torch.tensor(
        [-32.0, -11.25, -11.0, -0.013, 0.013, 11.0, 11.25, 32.0],
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = _reference(args)
    actual = _run_sm120(args)
    assert actual is not None
    _assert_matches_flashinfer(actual, expected)


def test_scalar_and_vector_scale_layouts_dispatch():
    args = _make_inputs(8, 8192, 5120)
    expected = _run_sm120(args)
    assert expected is not None

    # Some callers retain per-tensor scales as one-element vectors. The Python
    # facade normalizes both layouts to the scalar TVM-FFI contract.
    vector_scale_output = _run_sm120(args, vector_scales=True)
    torch.testing.assert_close(vector_scale_output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m_values,n,k", UNSUPPORTED_CONFIGS)
def test_unsupported_shapes_fall_back(m_values, n: int, k: int):
    args = _make_inputs(max(m_values), n, k)
    for m in m_values:
        assert _run_sm120(args, m=m) is None


def test_bias_falls_back():
    args = _make_inputs(8, 8192, 5120)
    bias = torch.zeros(args[1].shape[1], dtype=torch.bfloat16, device="cuda")
    assert _run_sm120(args, bias=bias) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
