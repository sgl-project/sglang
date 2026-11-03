# SPDX-License-Identifier: Apache-2.0
"""Match clamp -> BF16 SwiGLU -> group FP8 quant without the intermediate."""

import sys

import pytest
import torch

from sglang.kernels.ops.activation.silu_and_mul_clamp import (
    silu_and_mul_clamp,
    silu_and_mul_contig_post_quant,
)
from sglang.kernels.ops.quantization.per_token_group_quant import per_token_group_quant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _fused(x, *, round_to_bf16=True, rows_per_cta=4):
    rows, width = x.shape
    out = torch.empty((rows, width // 2), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows, width // 256), device=x.device, dtype=torch.float32)
    silu_and_mul_contig_post_quant(
        x,
        out,
        scale,
        128,
        swiglu_limit=10,
        round_to_bf16=round_to_bf16,
        rows_per_cta=rows_per_cta,
    )
    return out, scale


def _reference(x):
    activated = torch.empty(
        (x.shape[0], x.shape[1] // 2), device=x.device, dtype=x.dtype
    )
    silu_and_mul_clamp(x, activated, 10)
    return per_token_group_quant(activated, group_size=128)


@pytest.mark.parametrize("rows", [1, 17, 1024, 4237])
@pytest.mark.parametrize("rows_per_cta", [1, 4])
def test_bitwise_matches_separate_kernels(rows, rows_per_cta):
    torch.manual_seed(42)
    x = (torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16) * 12).contiguous()
    expected_q, expected_s = _reference(x)
    actual_q, actual_s = _fused(x, rows_per_cta=rows_per_cta)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_s, expected_s)


def test_zero_and_uninitialized_padding_sanitization():
    x = torch.zeros(17, 1024, device="cuda", dtype=torch.bfloat16)
    x[1].fill_(float("nan"))
    expected_q, expected_s = _reference(x)
    actual_q, actual_s = _fused(x)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_s, expected_s)


def test_graph_replay_uses_updated_input():
    torch.manual_seed(7)
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


def test_existing_precision_default_is_unchanged():
    torch.manual_seed(8)
    x = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)
    actual_q, actual_s = _fused(x, round_to_bf16=False, rows_per_cta=1)
    expected_q = torch.empty_like(actual_q)
    expected_s = torch.empty_like(actual_s)
    silu_and_mul_contig_post_quant(x, expected_q, expected_s, 128, swiglu_limit=10)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_s, expected_s)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
