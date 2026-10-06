"""Tests for fused sigmoid gating delta rule MTP kernel (GDN target_verify).

Compares the fused kernel `fused_sigmoid_gating_delta_rule_update` against
the reference two-step implementation:
    1. g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    2. o = fused_recurrent_gated_delta_rule_update(q, k, v, g, beta, ...)
"""

import sys

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_sm90_supported
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

try:
    from sglang.kernels.ops.attention.fla import fused_recurrent as packed_decode_module
    from sglang.kernels.ops.attention.fla import (
        fused_sigmoid_gating_recurrent as recurrent_module,
    )
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
        fused_recurrent_gated_delta_rule_update,
    )
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        _select_recurrent_launch_config,
        fused_sigmoid_gating_delta_rule_update,
    )

    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)


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


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernel not available")
@pytest.mark.parametrize("N", [1, 3, 16])
def test_qwen35_tp4_fused_gdn_mtp_precision(N: int):
    """Exercise the gfx950 TP4 launch shape against the reference path."""
    T, H, HV, K, V = 4, 4, 16, 128, 128
    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        N, T, H, HV, K, V
    )

    out_ref = run_reference(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state.clone(),
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
        state.clone(),
        indices,
        cu_seqlens,
        disable_state_update=True,
    )

    torch.testing.assert_close(out_ref, out_fused, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not (torch.version.hip and is_gfx95_supported()), reason="requires AMD gfx95"
)
def test_qwen35_tp4_launch_config_is_narrow():
    assert _select_recurrent_launch_config(1, 4, 16, 128, 128, False) == (8, 4)
    assert _select_recurrent_launch_config(3, 4, 16, 128, 128, False) == (16, 2)
    assert _select_recurrent_launch_config(32, 4, 16, 128, 128, False) == (16, 2)
    assert _select_recurrent_launch_config(33, 4, 16, 128, 128, False) == (32, 1)
    assert _select_recurrent_launch_config(3, 8, 32, 128, 128, False) == (32, 1)
    assert _select_recurrent_launch_config(3, 4, 16, 128, 128, True) == (32, 1)


_requires_sm90 = pytest.mark.skipif(
    torch.version.hip is not None or not is_sm90_supported(), reason="requires SM90"
)


@_requires_sm90
def test_sm90_verify_launch_config_is_narrow():
    assert _select_recurrent_launch_config(1, 16, 32, 128, 128, False, True) == (4, 1)
    assert _select_recurrent_launch_config(64, 4, 8, 128, 128, False, True) == (4, 1)
    assert _select_recurrent_launch_config(65, 16, 32, 128, 128, False, True) == (32, 1)
    assert _select_recurrent_launch_config(1, 16, 32, 128, 128, False, False) == (32, 1)
    assert _select_recurrent_launch_config(1, 16, 32, 128, 128, True, True) == (32, 1)
    assert _select_recurrent_launch_config(1, 16, 32, 64, 64, False, True) == (32, 1)


@_requires_sm90
@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
@pytest.mark.parametrize("N", [1, 3, 16, 64])
@pytest.mark.parametrize("T", [4, 16])
@pytest.mark.parametrize("H,HV", [(16, 32), (4, 16)])
@pytest.mark.parametrize("tree", [False, True])
def test_sm90_verify_launch_is_bit_exact(
    N: int, T: int, H: int, HV: int, tree: bool, monkeypatch
):
    """The tuned SM90 verify launch matches the BV=32 launch bit for bit."""
    K = V = 128
    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        N, T, H, HV, K, V
    )
    retrieve_parent_token = None
    if tree:
        # Random draft tree: token i > 0 reads the cached state of an earlier token.
        steps = torch.arange(T, device="cuda")
        retrieve_parent_token = (torch.rand(N, T, device="cuda") * steps).long()
        retrieve_parent_token[:, 0] = -1

    def run(launch_config):
        monkeypatch.setattr(
            recurrent_module, "_select_recurrent_launch_config", launch_config
        )
        buffer = torch.zeros(N, T, HV, V, K, dtype=torch.float32, device="cuda")
        out = run_fused_mtp(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            state.clone(),
            indices,
            cu_seqlens,
            disable_state_update=True,
            intermediate_states_buffer=buffer,
            intermediate_state_indices=indices,
            cache_steps=T,
            retrieve_parent_token=retrieve_parent_token,
        )
        return out, buffer

    out_tuned, states_tuned = run(_select_recurrent_launch_config)
    out_default, states_default = run(lambda *args, **kwargs: (32, 1))

    assert _select_recurrent_launch_config(N, H, HV, K, V, False, True) == (4, 1)
    assert torch.equal(out_tuned, out_default)
    assert torch.equal(states_tuned, states_default)


@_requires_sm90
def test_sm90_packed_decode_launch_config_is_narrow():
    def select(n: int, h: int, hv: int) -> tuple[int, int]:
        return _select_recurrent_launch_config(
            n, h, hv, 128, 128, False, packed_decode=True
        )

    assert select(1, 16, 48) == (4, 1)
    assert select(64, 4, 12) == (4, 1)
    assert select(65, 16, 48) == (32, 1)


@_requires_sm90
@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
@pytest.mark.parametrize("N", [1, 3, 16, 64])
@pytest.mark.parametrize("H,HV", [(16, 48), (4, 12)])
def test_sm90_packed_decode_launch_is_bit_exact(N: int, H: int, HV: int, monkeypatch):
    """The tuned SM90 packed decode launch matches the BV=32 launch bit for bit."""
    K = V = 128
    num_slots = N + 2
    torch.manual_seed(2025)
    mixed_qkv = torch.randn(N, 2 * H * K + HV * V, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(N, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, HV, dtype=torch.bfloat16, device="cuda")
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda")
    initial_state = torch.randn(num_slots, HV, V, K, dtype=torch.float32, device="cuda")
    state_indices = torch.randperm(num_slots, device="cuda")[:N].to(torch.int32)

    def run(launch_config):
        monkeypatch.setattr(
            packed_decode_module, "_select_recurrent_launch_config", launch_config
        )
        state = initial_state.clone()
        out = mixed_qkv.new_empty(N, 1, HV, V)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=K**-0.5,
            initial_state=state,
            out=out,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        return out, state

    out_tuned, state_tuned = run(_select_recurrent_launch_config)
    out_default, state_default = run(lambda *args, **kwargs: (32, 1))

    assert torch.equal(out_tuned, out_default)
    assert torch.equal(state_tuned, state_default)


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


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
def test_verify_scratch_pitch_uses_allocated_steps():
    # Gear below the allocated step dim must not spill into the neighbor block.
    N, T, ALLOCATED = 2, 4, 8
    H, HV, K, V = 16, 32, 128, 128

    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        N, T, H, HV, K, V
    )
    buffer = torch.full(
        (N + 1, ALLOCATED, HV, V, K), float("nan"), dtype=torch.float32, device="cuda"
    )

    run_fused_mtp(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state,
        indices,
        cu_seqlens,
        disable_state_update=True,
        intermediate_states_buffer=buffer,
        intermediate_state_indices=indices,
        cache_steps=T,
    )

    assert not torch.isnan(buffer[:N, :T]).any()
    assert torch.isnan(buffer[N:]).all()
    assert torch.isnan(buffer[:N, T:]).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
