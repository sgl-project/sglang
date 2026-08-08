import sys

import pytest
import torch

from sglang.kernels.ops.attention.fla.layernorm_gated import (
    rms_norm_gated,
    rms_norm_gated_fp8_quant,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 128
NUM_HEADS = 16
EPS = 1e-6


@pytest.mark.parametrize("scale_ue8m0", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 16, 257, 16384])
def test_rms_norm_gated_fp8_quant(num_tokens, scale_ue8m0):
    torch.manual_seed(num_tokens)
    shape = (num_tokens * NUM_HEADS, HEAD_DIM)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    normalized = rms_norm_gated(
        x=x,
        weight=weight,
        bias=None,
        z=z,
        eps=EPS,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="swish",
    ).reshape(num_tokens, NUM_HEADS * HEAD_DIM)
    expected_q, expected_scale = sglang_per_token_group_quant_fp8(
        normalized,
        HEAD_DIM,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )

    actual_q, actual_scale = rms_norm_gated_fp8_quant(
        x=x,
        weight=weight,
        z=z,
        eps=EPS,
        num_groups=NUM_HEADS,
        activation="swish",
        scale_ue8m0=scale_ue8m0,
    )
    actual_q = actual_q.reshape(num_tokens, NUM_HEADS * HEAD_DIM)

    assert actual_q.dtype == expected_q.dtype
    assert actual_scale.shape == expected_scale.shape
    assert actual_scale.stride() == expected_scale.stride()
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    mismatch_rate = (actual_q != expected_q).float().mean().item()
    assert mismatch_rate <= (3e-6 if scale_ue8m0 else 2e-3)
    torch.testing.assert_close(
        actual_q.float(), expected_q.float(), rtol=0.13, atol=0.002
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
