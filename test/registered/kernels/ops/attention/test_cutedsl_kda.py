"""Tests for CuTe DSL fused sigmoid gating KDA decode kernel."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.cutedsl_kda import (
    cutedsl_fused_sigmoid_gating_kda_update,
)
from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("B", [16, 128])
def test_cutedsl_kda_matches_triton(B: int, varlen: bool):
    """CuTe DSL decode must read dt_bias in the caller's dtype; KDA models keep it in fp32.

    bf16 then fp32 at the same shape, so the second call cannot reuse the first's kernel.
    """
    torch.manual_seed(2025)
    H, HV, K, V = 32, 32, 128, 128
    lead = (1, B) if varlen else (B, 1)

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    pool_size = 2 * B + 1
    indices = torch.randperm(pool_size, device="cuda")[:B].to(torch.int32)
    indices[B // 2] = -1
    valid = indices >= 0
    cu_seqlens = (
        torch.arange(B + 1, dtype=torch.int32, device="cuda") if varlen else None
    )
    initial_state = torch.randn(pool_size, HV, V, K, dtype=torch.float32, device="cuda")

    for dt_bias_dtype in (torch.bfloat16, torch.float32):
        dt_bias = torch.randn(HV * K, dtype=dt_bias_dtype, device="cuda")
        state_cutedsl = initial_state.clone()
        state_triton = initial_state.clone()

        for _ in range(3):
            q = torch.randn(*lead, H, K, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(*lead, H, K, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(*lead, HV, V, dtype=torch.bfloat16, device="cuda")
            a = torch.randn(B, HV * K, dtype=torch.bfloat16, device="cuda")
            b = torch.randn(B, HV, dtype=torch.bfloat16, device="cuda")

            out_cutedsl = cutedsl_fused_sigmoid_gating_kda_update(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                initial_state_source=state_cutedsl,
                initial_state_indices=indices,
                cu_seqlens=cu_seqlens,
            )
            out_triton = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=state_triton,
                initial_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                is_kda=True,
            )

            torch.testing.assert_close(
                out_cutedsl.reshape(B, HV, V)[valid].float(),
                out_triton.reshape(B, HV, V)[valid].float(),
                atol=1e-3,
                rtol=1e-2,
            )
            torch.testing.assert_close(
                state_cutedsl, state_triton, atol=1e-4, rtol=1e-4
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
