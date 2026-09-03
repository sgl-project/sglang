"""Correctness and dispatch tests for the KDA SM120 FP8 skinny GEMM."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.kda_kernels.sm120_fp8_skinny_gemm import (
    try_sm120_fp8_skinny_gemm,
)
from sglang.kernels.kda_kernels.sm120_fp8_skinny_gemm_sm120 import (
    _run_sm120_fp8_skinny_gemm,
)
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear_bmm_flashinfer
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)):
    pytest.skip("KDA FP8 skinny GEMM requires CUDA SM120", allow_module_level=True)


ALL_M = (1, 2, 4, 8, 9)

# Every tuple below passed three-seed bitwise comparison and cold-L2
# performance gating against FlashInfer. Shapes reflect SGLang's packed QKV and
# gate/up weights, not the individual checkpoint tensors.
QUALIFIED_CONFIGS = [
    ("Qwen3.8-27B", "attn-qkv", 8192, 5120, ALL_M),
    ("Qwen3.8-27B", "gdn-in", 16384, 5120, (2, 4, 8)),
    ("Qwen3.8-27B", "out", 5120, 6144, ALL_M),
    ("Qwen3-8B/Llama-3.1-8B", "o", 4096, 4096, ALL_M),
    ("Qwen3-14B", "qkv", 7168, 5120, ALL_M),
    ("Qwen3-14B", "o", 5120, 5120, (2, 4, 8, 9)),
    ("Qwen3-14B", "gate-up", 34816, 5120, ALL_M),
    ("Nemotron-3-Super", "shared-up", 5376, 4096, (1,)),
]
QUALIFIED_SHAPES = [
    pytest.param(m, n, k, id=f"{model}-{op}-m{m}".lower())
    for model, op, n, k, supported_m in QUALIFIED_CONFIGS
    for m in supported_m
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


def _reference(input, weight, weight_scale, input_scale):
    return apply_fp8_linear_bmm_flashinfer(
        input,
        weight,
        weight_scale,
        input_scale,
    )


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("m,n,k", QUALIFIED_SHAPES)
def test_qualified_shapes_are_bitwise_equal(m: int, n: int, k: int, seed: int):
    input, weight, weight_scale, input_scale = _make_inputs(m, n, k, seed)
    expected = _reference(input, weight, weight_scale, input_scale)
    actual = try_sm120_fp8_skinny_gemm(
        input,
        weight,
        input_scale,
        input_scale * weight_scale,
        bias=None,
    )
    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", ALL_M)
def test_cuda_graph_replay_uses_current_input(m: int):
    input, weight, weight_scale, input_scale = _make_inputs(m, 8192, 5120)
    output_scale = input_scale * weight_scale

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _run_sm120_fp8_skinny_gemm(input, weight, input_scale, output_scale)

    # A replay must consume the new activation rather than the values present
    # during capture. This also exercises the quantize-to-GEMM dependency.
    torch.manual_seed(17)
    input.copy_(torch.randn_like(input).mul_(8))
    graph.replay()
    expected = _reference(input, weight, weight_scale, input_scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", (2, 4, 8))
def test_saturated_real_activation_range_is_bitwise_equal(m: int):
    input, weight, weight_scale, input_scale = _make_inputs(m, 16384, 5120, seed=11)
    # Static ModelOpt scales can expose both saturation and FP8 rounding
    # boundaries in live GDN activations. This distribution reproduces the
    # class of mismatch that the former fused quantizer caused in E2E decode.
    input.mul_(32)
    input[:, :8] = torch.tensor(
        [-32.0, -11.25, -11.0, -0.013, 0.013, 11.0, 11.25, 32.0],
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = _reference(input, weight, weight_scale, input_scale)
    actual = try_sm120_fp8_skinny_gemm(
        input,
        weight,
        input_scale,
        input_scale * weight_scale,
        bias=None,
    )
    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_production_dispatch_and_fallback():
    input, weight, weight_scale, input_scale = _make_inputs(8, 8192, 5120)
    output_scale = input_scale * weight_scale
    expected = try_sm120_fp8_skinny_gemm(
        input, weight, input_scale, output_scale, bias=None
    )
    assert expected is not None

    # Some callers retain per-tensor scales as one-element vectors. The Python
    # facade normalizes both layouts to the scalar TVM-FFI contract.
    vector_scale_output = try_sm120_fp8_skinny_gemm(
        input,
        weight,
        input_scale.reshape(1),
        output_scale.reshape(1),
        bias=None,
    )
    torch.testing.assert_close(vector_scale_output, expected, rtol=0, atol=0)

    unsupported_input = input[:3]
    assert (
        try_sm120_fp8_skinny_gemm(
            unsupported_input, weight, input_scale, output_scale, bias=None
        )
        is None
    )

    # The GDN projection is qualified only for M=2/4/8. Keep M=1/9 on the
    # reference path because real-activation E2E qualification exposed drift.
    for slow_m in (1, 9):
        slow_input, slow_weight, slow_weight_scale, slow_input_scale = _make_inputs(
            slow_m, 16384, 5120
        )
        assert (
            try_sm120_fp8_skinny_gemm(
                slow_input,
                slow_weight,
                slow_input_scale,
                slow_input_scale * slow_weight_scale,
                bias=None,
            )
            is None
        )

    # Nemotron E2E qualification keeps shared-up only at M=1. Shared-up at
    # larger batches and shared-down regressed in both GPU-lane orientations.
    nemotron_up_input, nemotron_up_weight, weight_scale, input_scale = _make_inputs(
        9, 5376, 4096
    )
    for slow_m in (2, 4, 8, 9):
        assert (
            try_sm120_fp8_skinny_gemm(
                nemotron_up_input[:slow_m],
                nemotron_up_weight,
                input_scale,
                input_scale * weight_scale,
                bias=None,
            )
            is None
        )

    nemotron_down_input, nemotron_down_weight, weight_scale, input_scale = _make_inputs(
        9, 4096, 5376
    )
    for slow_m in ALL_M:
        assert (
            try_sm120_fp8_skinny_gemm(
                nemotron_down_input[:slow_m],
                nemotron_down_weight,
                input_scale,
                input_scale * weight_scale,
                bias=None,
            )
            is None
        )

    # This QKV shape is bitwise-correct but slower than FlashInfer, so it must
    # stay on the fallback path.
    slow_input, slow_weight, slow_weight_scale, slow_input_scale = _make_inputs(
        8, 6144, 4096
    )
    assert (
        try_sm120_fp8_skinny_gemm(
            slow_input,
            slow_weight,
            slow_input_scale,
            slow_input_scale * slow_weight_scale,
            bias=None,
        )
        is None
    )
    bias = torch.zeros(weight.shape[1], dtype=torch.bfloat16, device="cuda")
    assert (
        try_sm120_fp8_skinny_gemm(input, weight, input_scale, output_scale, bias=bias)
        is None
    )


def _make_modelopt_dispatch_fixture():
    method = object.__new__(ModelOptFp8LinearMethod)
    method.use_marlin = False
    method.use_sm120_gemv = True
    method.use_kda_fp8_skinny = True
    weight_storage = torch.empty((256, 512), dtype=torch.uint8, device="cuda")
    layer = SimpleNamespace(
        weight=weight_storage.t(),
        input_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        sm120_gemv_alpha=torch.ones(1, dtype=torch.float32, device="cuda"),
    )
    input = torch.empty((1, 512), dtype=torch.bfloat16, device="cuda")
    return method, layer, input


def test_modelopt_prefers_native_m1_gemv(monkeypatch):
    import sglang.kernels.ops.gemm as gemm_ops
    import sglang.kernels.ops.gemm.sm120_fp8_gemv as native_gemv
    import sglang.kernels.ops.quantization.fp8_kernel as fp8_kernel

    method, layer, input = _make_modelopt_dispatch_fixture()
    expected = torch.empty((1, 256), dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(native_gemv, "use_sm120_fp8_gemv", lambda *args: True)
    monkeypatch.setattr(native_gemv, "sm120_fp8_gemv", lambda *args: expected)
    monkeypatch.setattr(
        fp8_kernel,
        "static_quant_fp8",
        lambda *args, **kwargs: (input.to(torch.float8_e4m3fn), None),
    )
    monkeypatch.setattr(
        gemm_ops,
        "try_sm120_fp8_skinny_gemm",
        lambda *args, **kwargs: pytest.fail("KDA must not preempt native M=1 GEMV"),
    )

    assert method.apply(layer, input) is expected


def test_modelopt_uses_kda_after_native_rejects(monkeypatch):
    import sglang.kernels.ops.gemm as gemm_ops
    import sglang.kernels.ops.gemm.sm120_fp8_gemv as native_gemv

    method, layer, input = _make_modelopt_dispatch_fixture()
    expected = torch.empty((1, 256), dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(native_gemv, "use_sm120_fp8_gemv", lambda *args: False)
    monkeypatch.setattr(
        gemm_ops, "try_sm120_fp8_skinny_gemm", lambda *args, **kwargs: expected
    )

    assert method.apply(layer, input) is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
