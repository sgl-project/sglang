# SPDX-License-Identifier: Apache-2.0
"""Preserve the BF16 activation boundary when fusing DeepGEMM's down quantizer."""

import pytest
import torch

from sglang.kernels.ops.activation.silu_and_mul_clamp import (
    silu_and_mul_clamp,
    silu_and_mul_contig_post_quant,
)
from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale
from sglang.kernels.ops.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    sglang_per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _reference(x):
    activated = torch.empty(
        (x.shape[0], x.shape[1] // 2), device=x.device, dtype=x.dtype
    )
    silu_and_mul_clamp(x, activated, 10.0)
    q, scale = sglang_per_token_group_quant_fp8(activated, 128)
    return q, tma_align_input_scale(scale)


def _fused(x, *, tma_aligned=True, rows_per_cta=None):
    rows, width = x.shape
    hidden = width // 2
    if rows_per_cta is None:
        rows_per_cta = 4 if hidden <= 2048 else 2 if hidden <= 4096 else 1
    q = torch.empty((rows, hidden), device=x.device, dtype=torch.float8_e4m3fn)
    scale = create_per_token_group_quant_fp8_output_scale(
        q.shape, x.device, 128, tma_aligned, tma_aligned, False
    )
    silu_and_mul_contig_post_quant(
        x,
        q,
        scale,
        128,
        swiglu_limit=10.0,
        round_to_bf16=True,
        rows_per_cta=rows_per_cta,
    )
    return q, scale


@pytest.mark.parametrize("rows", [1, 17, 1024, 4237])
@pytest.mark.parametrize("hidden", [256, 512, 1024, 1536, 2048, 3072, 4096, 8192])
@pytest.mark.parametrize("tma_aligned", [False, True])
def test_bitwise_matches_deepgemm_reference(rows, hidden, tma_aligned):
    # FP32-only fusion changes the scale and FP8 bins near BF16 round boundaries.
    torch.manual_seed(42)
    x = (
        torch.randn(rows, 2 * hidden, device="cuda", dtype=torch.bfloat16) * 12
    ).contiguous()
    expected_q, expected_s = _reference(x)
    actual_q, actual_s = _fused(x, tma_aligned=tma_aligned)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_s, expected_s)
    if tma_aligned:
        assert actual_s.stride(0) == 1
        assert actual_s.stride(1) % 4 == 0


@pytest.mark.parametrize("rows_per_cta", [1, 2, 4, 8])
def test_tail_rows_are_not_dropped(rows_per_cta):
    torch.manual_seed(7)
    x = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)
    expected_q, expected_s = _reference(x)
    actual_q, actual_s = _fused(x, rows_per_cta=rows_per_cta)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_s, expected_s)


def test_zero_and_nan_padding_matches_separate_kernels():
    x = torch.zeros(17, 1024, device="cuda", dtype=torch.bfloat16)
    x[1].fill_(float("nan"))
    expected_q, expected_s = _reference(x)
    actual_q, actual_s = _fused(x)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_s, expected_s)


def test_empty_input_does_not_launch_an_empty_grid():
    q, scale = _fused(torch.empty(0, 1024, device="cuda", dtype=torch.bfloat16))
    assert q.shape == (0, 512) and scale.shape == (0, 4)


def test_graph_replay_uses_new_input():
    torch.manual_seed(8)
    x = torch.randn(1025, 1024, device="cuda", dtype=torch.bfloat16)
    _fused(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_q, actual_s = _fused(x)
    x.normal_()
    graph.replay()
    expected_q, expected_s = _reference(x)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_s, expected_s)


def test_existing_fp32_precision_default_is_unchanged():
    torch.manual_seed(19)
    x = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(17, 512, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty(17, 4, device="cuda", dtype=torch.float32)
    silu_and_mul_contig_post_quant(x, out, scale, 128, swiglu_limit=10.0)
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=10).float()
    up = up.clamp(-10, 10).float()
    activated = gate * torch.sigmoid(gate) * up
    expected_s = activated.reshape(17, 4, 128).abs().amax(-1).clamp_min(1e-10) / 448
    # Rounding the activation to BF16 before absmax violates this FP32 contract.
    torch.testing.assert_close(scale, expected_s, rtol=2e-6, atol=1e-10)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
