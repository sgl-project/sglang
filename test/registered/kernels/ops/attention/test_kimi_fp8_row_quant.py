"""Kimi packed-scale row schedule preserves the original quantization bytes."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
    pytest.skip("Blackwell CUDA is required", allow_module_level=True)


@pytest.mark.parametrize("m", [256, 384])
@pytest.mark.parametrize("k", [1536, 7168])
def test_row_schedule_matches_flat(m, k):
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )
    from sglang.srt.layers.quantization.kimi_fp8_dense import quantize_rows

    torch.manual_seed(42)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    x[1, 0] = float("nan")
    x[2, 0] = float("inf")
    x[3] *= 1e-20
    reference = sglang_per_token_group_quant_fp8(
        x, 128, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True
    )
    actual = quantize_rows(x)
    assert torch.equal(actual[0].view(torch.uint8), reference[0].view(torch.uint8))
    assert torch.equal(actual[1], reference[1])
    assert actual[1].stride() == reference[1].stride()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = quantize_rows(x)
    x.mul_(0.5)
    graph.replay()
    expected = quantize_rows(x)
    assert torch.equal(captured[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(captured[1], expected[1])
