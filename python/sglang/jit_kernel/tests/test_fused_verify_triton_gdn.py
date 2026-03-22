"""Tests for fused sigmoid gating delta rule MTP kernel (GDN target_verify).

Compares the fused kernel `fused_sigmoid_gating_delta_rule_update` against
the reference two-step implementation:
    1. g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    2. o = fused_recurrent_gated_delta_rule_update(q, k, v, g, beta, ...)
"""

import pytest
import torch

try:
    from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update,
    )
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False


def _make_tensors(N, T, H, HV, K, V, device="cuda", seed=2025):
    """Create input tensors for GDN target_verify."""
    torch.manual_seed(seed)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device=device)
    a = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device=device)
    q = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, N * T, HV, V, dtype=torch.bfloat16, device=device)
    indices = torch.arange(N, dtype=torch.int32, device=device)
    initial_state = torch.randn(N, HV, K, V, dtype=torch.float, device=device)
    cu_seqlens = torch.arange(0, N * T + 1, T, dtype=torch.int32, device=device)
    return A_log, dt_bias, a, b, q, k, v, initial_state, indices, cu_seqlens


def run_reference(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    disable_state_update=True,
    intermediate_states_buffer=None,
    intermediate_state_indices=None,
    cache_steps=None,
    retrieve_parent_token=None,
):
    """Reference: fused_gdn_gating + fused_recurrent_gated_delta_rule_update."""
    # fused_gdn_gating expects 2D [seq_len, HV]
    a_2d = a.view(-1, a.shape[-1])
    b_2d = b.view(-1, b.shape[-1])
    g, beta = fused_gdn_gating(A_log, a_2d, b_2d, dt_bias)
    # fused_recurrent expects 3D [B, T, HV]
    g = g.view(a.shape)
    beta = beta.view(b.shape)

    # fused_recurrent requires intermediate_state_indices when cu_seqlens is used
    if cu_seqlens is not None and intermediate_state_indices is None:
        N = len(cu_seqlens) - 1
        intermediate_state_indices = torch.arange(N, dtype=torch.int32, device=q.device)

    return fused_recurrent_gated_delta_rule_update(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
    )


def run_fused_mtp(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    disable_state_update=True,
    intermediate_states_buffer=None,
    intermediate_state_indices=None,
    cache_steps=None,
    retrieve_parent_token=None,
):
    """Fused: fused_sigmoid_gating_delta_rule_update."""
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=False,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
    )


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernel not available")
@pytest.mark.parametrize("N", [1, 8, 16])
@pytest.mark.parametrize("T", [1, 4, 8])
def test_fused_gdn_mtp_precision(N: int, T: int):
    """Compare fused MTP output against reference."""
    H, HV, K, V = 16, 32, 128, 128

    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        N, T, H, HV, K, V
    )

    state_ref = state.clone()
    state_fused = state.clone()

    out_ref = run_reference(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_ref,
        indices,
        cu_seqlens,
        disable_state_update=True,
    )
    out_fused = run_fused_mtp(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_fused,
        indices,
        cu_seqlens,
        disable_state_update=True,
    )

    torch.testing.assert_close(out_ref, out_fused, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
@pytest.mark.parametrize("N", [1, 16, 128])
def test_mtp_single_step_decode(N: int):
    """Verify MTP kernel matches reference for T=1 (decode scenario)."""
    T = 1
    H, HV, K, V = 16, 32, 128, 128

    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        N, T, H, HV, K, V
    )

    state_ref = state.clone()
    state_fused = state.clone()

    out_ref = run_reference(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_ref,
        indices,
        cu_seqlens,
        disable_state_update=False,
    )
    out_fused = run_fused_mtp(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_fused,
        indices,
        cu_seqlens,
        disable_state_update=False,
    )

    torch.testing.assert_close(out_ref, out_fused, rtol=1e-2, atol=1e-2)

    # Also verify states match after update
    state_diff = (state_ref.float() - state_fused.float()).abs()
    state_max_diff = state_diff.max().item()
    state_fail_rate = (state_diff > 0.1).float().mean().item() * 100
    print(
        f"  single_step state N={N}: max_diff={state_max_diff:.2e}, "
        f"fail_rate={state_fail_rate:.2f}%"
    )
    assert state_fail_rate < 0.01, f"State mismatch: fail_rate={state_fail_rate:.2f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
