# SPDX-License-Identifier: Apache-2.0
"""Numeric tests for the weight-only FP4 compressed-tensors schemes.

Each test fills checkpoint-format tensors, runs process_weights_after_loading and
apply_weights, and compares against weights dequantized in PyTorch. That covers
the checkpoint-to-kernel translation only the schemes do -- the inverted NVFP4
global scale, raw E8M0 scale bytes, the [gate; up] order of w13, non-gated
experts -- plus the kernel forward. The kernels alone are covered by
test_nvfp4_marlin.py.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A16Fp4,
    CompressedTensorsW4A16Nvfp4MoE,
)
from sglang.srt.utils.common import (
    is_sm80_supported,
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import (
    make_nvfp4_weight_and_ref,
)

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

requires_fp4_marlin = pytest.mark.skipif(
    not (
        is_sm80_supported()
        or is_sm90_supported()
        or is_sm100_supported()
        or is_sm120_supported()
    ),
    reason="Weight-only FP4 Marlin requires CUDA SM8X/SM9X/SM100/SM120",
)

SIZE_M = 17
SIZE_K = 256
SIZE_N = 192


def _rel_err(actual, expected):
    # Relative, not absolute: the error scales with the output magnitude, which
    # depends on K and on the random inputs. One bf16 eps is 2**-7 = 0.0078.
    return (actual.float() - expected.float()).norm() / expected.float().norm()


def _make_weight(fmt, rows, cols, dtype):
    """Returns (packed fp4, scales, global-scale divisor or None, reference)."""
    if fmt == "nvfp4":
        fp4, scales, global_scale, ref = make_nvfp4_weight_and_ref(
            rows, cols, dtype, group_size=16
        )
        # compressed-tensors stores the global scale as a divisor.
        return fp4, scales, (1 / global_scale).item(), ref
    raise ValueError(fmt)


def _build_linear(fmt, dtype, w4a4):
    """A w4a4 checkpoint is served weight-only; its activation quantization is
    dropped, not applied, so it must match the a16 numbers."""
    if fmt == "nvfp4":
        scheme = CompressedTensorsW4A16Fp4(has_input_global_scale=w4a4)
    layer = torch.nn.Module()
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[SIZE_N],
        input_size_per_partition=SIZE_K,
        params_dtype=dtype,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer.to("cuda")
    fp4, scales, divisor, ref = _make_weight(fmt, SIZE_N, SIZE_K, dtype)
    layer.weight_packed.data.copy_(fp4)
    layer.weight_scale.data.copy_(scales)
    if divisor is not None:
        layer.weight_global_scale.data.fill_(divisor)
    if w4a4 and fmt == "nvfp4":
        layer.input_global_scale.data.fill_(1 / 448.0)
    return scheme, layer, ref


LINEAR_CASES = [
    ("nvfp4", torch.float16, False),
    ("nvfp4", torch.float16, True),
    ("nvfp4", torch.bfloat16, False),
    ("nvfp4", torch.bfloat16, True),
]


@requires_fp4_marlin
@pytest.mark.parametrize("fmt,dtype,w4a4", LINEAR_CASES)
def test_linear_matches_dequant_reference(fmt, dtype, w4a4):
    torch.manual_seed(0)
    scheme, layer, weight_ref = _build_linear(fmt, dtype, w4a4)
    scheme.process_weights_after_loading(layer)

    x = torch.randn((SIZE_M, SIZE_K), dtype=dtype, device="cuda") / 10
    out = scheme.apply_weights(layer, x)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    rel = _rel_err(out, x @ weight_ref.T)
    assert rel < 0.02, f"relative error {rel:.4f} too large"


@pytest.mark.skipif(
    not is_sm100_supported(), reason="FlashInfer mm_bf16_fp4 requires SM100"
)
@pytest.mark.parametrize(
    "backend",
    [
        fp4_utils.Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
        fp4_utils.Fp4GemmRunnerBackend.FLASHINFER_CUDNN,
    ],
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_flashinfer_linear_matches_dequant_reference(monkeypatch, backend, has_bias):
    """The FlashInfer prep: 128x4 block-scale swizzle and the global scale as
    alpha. A 3D input covers the reshape around the 2D GEMM."""
    if backend.is_flashinfer_cudnn():
        cudnn = pytest.importorskip("cudnn")
        # FlashInfer's minimum for the cuDNN bf16 x fp4 GEMM (9.23.1).
        if cudnn.backend_version() < 92301:
            pytest.skip(f"cuDNN {cudnn.backend_version()} lacks bf16 x fp4 GEMM")
    monkeypatch.setattr(fp4_utils, "FP4_GEMM_RUNNER_BACKEND", backend)
    torch.manual_seed(0)
    dtype = torch.bfloat16
    scheme, layer, weight_ref = _build_linear("nvfp4", dtype, False)
    scheme.process_weights_after_loading(layer)
    assert layer.use_flashinfer_bf16_fp4

    x = torch.randn((2, SIZE_M, SIZE_K), dtype=dtype, device="cuda") / 10
    bias = torch.randn(SIZE_N, dtype=dtype, device="cuda") if has_bias else None
    out = scheme.apply_weights(layer, x, bias)
    expected = x @ weight_ref.T + (bias if has_bias else 0)
    torch.cuda.synchronize()
    assert out.shape == (2, SIZE_M, SIZE_N)
    rel = _rel_err(out, expected)
    assert rel < 0.02, f"relative error {rel:.4f} too large"


NUM_EXPERTS = 4
TOP_K = 2
NUM_TOKENS = 17
# FP4 Marlin MoE requires hidden_size % 128 == 0 and
# intermediate_size_per_partition % 64 == 0.
HIDDEN = 256
INTERMEDIATE = 128


def _build_moe(fmt, is_gated, activation, dtype):
    """Returns the scheme, the loaded layer and per-expert reference weights.
    Each expert gets its own scales, so a scale applied to the wrong expert
    shows up."""
    config = MoeRunnerConfig(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        top_k=TOP_K,
        params_dtype=dtype,
        activation=activation,
        is_gated=is_gated,
    )
    layer = torch.nn.Module()
    layer.moe_runner_config = config
    scheme = CompressedTensorsW4A16Nvfp4MoE()
    scheme.create_weights(
        layer=layer,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        params_dtype=dtype,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer.to("cuda")

    w13_rows = (2 if is_gated else 1) * INTERMEDIATE
    refs = {}
    for name, rows, cols in (("w13", w13_rows, HIDDEN), ("w2", HIDDEN, INTERMEDIATE)):
        refs[name] = torch.empty(NUM_EXPERTS, rows, cols, dtype=dtype, device="cuda")
        for e in range(NUM_EXPERTS):
            # One matrix for gate and up together, so for NVFP4 they share a
            # global scale, as llm-compressor exports them.
            fp4, scales, divisor, refs[name][e] = _make_weight(fmt, rows, cols, dtype)
            getattr(layer, f"{name}_weight_packed").data[e].copy_(fp4)
            getattr(layer, f"{name}_weight_scale").data[e].copy_(scales)
            if divisor is not None:
                getattr(layer, f"{name}_weight_global_scale").data[e] = divisor

    layer.dispatcher = SimpleNamespace(
        local_expert_mapping=None, num_experts=NUM_EXPERTS
    )
    scheme.create_moe_runner(layer, config)
    scheme.process_weights_after_loading(layer)
    return scheme, layer, refs["w13"], refs["w2"]


def _moe_reference(x, w13, w2, topk_weights, topk_ids, is_gated, activation):
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = topk_ids[t, j].item()
            h = x[t].float() @ w13[e].float().T
            if is_gated:
                gate, up = h.chunk(2)
                act = F.silu(gate) * up
            elif activation == "relu2":
                act = torch.square(F.relu(h))
            else:
                act = F.silu(h)
            out[t] += topk_weights[t, j].float() * (act @ w2[e].float().T)
    return out


@requires_fp4_marlin
@pytest.mark.parametrize("fmt", ["nvfp4"])
@pytest.mark.parametrize(
    "is_gated,activation", [(True, "silu"), (False, "relu2"), (False, "silu")]
)
def test_moe_matches_dequantized_experts(fmt, is_gated, activation):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    scheme, layer, w13, w2 = _build_moe(fmt, is_gated, activation, dtype)

    x = torch.randn(NUM_TOKENS, HIDDEN, dtype=dtype, device="cuda") / 10
    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32, device="cuda")
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), TOP_K, dim=-1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = topk_ids.to(torch.int32)
    dispatch_output = StandardDispatchOutput(
        x, None, StandardTopKOutput(topk_weights, topk_ids, logits)
    )

    out = scheme.apply_weights(layer, dispatch_output).hidden_states
    expected = _moe_reference(x, w13, w2, topk_weights, topk_ids, is_gated, activation)
    torch.cuda.synchronize()
    # The kernel keeps the intermediate activation in bf16, hence 0.03.
    rel = _rel_err(out, expected)
    assert rel < 0.03, f"relative error {rel:.4f} too large"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
