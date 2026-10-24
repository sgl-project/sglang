import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_contiguous_deep_gemm_respects_oai_swiglu():
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        _contiguous_deep_gemm_silu_mul_quant,
    )

    torch.manual_seed(0)
    gate_up = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")
    output, scale = _contiguous_deep_gemm_silu_mul_quant(
        gate_up,
        group_size=128,
        topk=4,
        gemm1_alpha=1.702,
        gemm1_clamp_limit=7.0,
    )

    actual = output.float() * scale.repeat_interleave(128, dim=-1)
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate = gate.clamp(max=7.0)
    expected = gate * torch.sigmoid(1.702 * gate)
    expected *= up.clamp(-7.0, 7.0) + 1.0

    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    normalized_mae = (actual - expected).abs().mean() / expected.abs().mean()
    assert cosine > 0.999
    assert normalized_mae < 0.03
