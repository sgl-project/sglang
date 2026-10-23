"""Tests for CuTe DSL fused sigmoid gating delta rule kernel (GDN)."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

try:
    import cutlass  # noqa: F401

    from sglang.kernels.ops.attention import cutedsl_gdn

    CUTEDSL_AVAILABLE = True
except ImportError:
    CUTEDSL_AVAILABLE = False
    cutedsl_gdn = None

try:
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def run_triton_kernel(
    A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale, cu_seqlens=None
):
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
@pytest.mark.parametrize("pool_view", ["4d", "1d"])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("B", [16, 128])
def test_cutedsl_gdn_precision(B: int, varlen: bool, pool_view: str):
    """CuTe DSL decode must read and update the Triton [pool, HV, V, K] state pool."""
    torch.manual_seed(2025)
    T, H, K, V, HV = 1, 16, 128, 128, 32
    scale = K**-0.5
    lead = (1, B) if varlen else (B, T)
    gate_shape = (B, HV) if varlen else (B, T, HV)

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    pool_size = 2 * B + 1
    indices = torch.randperm(pool_size, device="cuda")[:B].to(torch.int32)
    indices[B // 2] = -1
    valid = indices >= 0
    cu_seqlens = (
        torch.arange(B + 1, dtype=torch.int32, device="cuda") if varlen else None
    )
    state_cutedsl = torch.randn(pool_size, HV, V, K, dtype=torch.float32, device="cuda")
    state_triton = state_cutedsl.clone()
    pool = state_cutedsl.view(-1) if pool_view == "1d" else state_cutedsl

    for _ in range(3):
        a = torch.randn(*gate_shape, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(*gate_shape, dtype=torch.bfloat16, device="cuda")
        q = torch.randn(*lead, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(*lead, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(*lead, HV, V, dtype=torch.bfloat16, device="cuda")

        out_cutedsl = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log, dt_bias, q, k, v, a, b, pool, indices, cu_seqlens, scale=scale
        )
        out_triton = run_triton_kernel(
            A_log, dt_bias, q, k, v, a, b, state_triton, indices, scale, cu_seqlens
        )

        torch.testing.assert_close(
            out_cutedsl.reshape(B, HV, V)[valid].float(),
            out_triton.reshape(B, HV, V)[valid].float(),
            atol=1e-3,
            rtol=1e-2,
        )
        torch.testing.assert_close(state_cutedsl, state_triton, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
