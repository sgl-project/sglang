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
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update,
    )
    from sglang.kernels.ops.attention.linear import (
        fused_sigmoid_gating_recurrent as recurrent_module,
    )
    from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        _select_recurrent_launch_config,
        fused_sigmoid_gating_delta_rule_update,
    )

    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="base-a-jit-kernel-unit", runner_config="amd")
# backend-specific: the gfx95 TP4 launch config only applies on MI35x.
register_amd_ci(est_time=10, suite="nightly-test-1-gpu-amd-mi35x", nightly=True)


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


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernel not available")
def test_fused_recurrent_update_strided_state_pool():
    """FusedRecurrentUpdateFunction must update a strided unified-memory pool.

    Regression test for sgl-project/sglang#53615: with --enable-unified-memory,
    Jet-Nemotron/JetVLM pass temporal[layer] of the state view over the KV arena
    as initial_state_source. One slot entry holds every state layer's conv and
    temporal rows, so the pool is strided and a plain @input_guard used to hand
    the kernel a .contiguous() copy that was then discarded, leaving the pool
    untouched and every decode step running from a zeroed state. The fix keeps
    the pool's strides (keep_strides=("initial_state_source",)), as #50307 did
    for the GDN chunk prefill.

    Runs a prefill step and a decode step on a TensorEntryLayout strided pool
    and on a contiguous pool holding the same values, and requires the same
    outputs, the same updated states, and no writes outside the batch slots'
    rows of the tested layer.
    """
    from sglang.srt.mem_cache.kv.decl import TensorPartDecl
    from sglang.srt.mem_cache.kv.layout.state import state_entry_layout

    # Jet-Nemotron-2B jet geometry, as JetNemotronConfig.mamba2_cache_params
    # builds it: num_heads 12, head_dim 96, expand_v 2, conv_size 4.
    H, K = 12, 96
    V = 2 * K
    conv_shape = (H * V + 2 * H * K, 4 - 1)
    temporal_shape = (H, V, K)  # (num_heads, head_dim, state_size)
    num_layers, layer = 4, 2
    num_slots = 9  # slot 0 reserved
    slots = torch.tensor([3, 6, 8], dtype=torch.int32, device="cuda")

    layers = tuple(range(num_layers))
    layout = state_entry_layout(
        (
            TensorPartDecl(
                name="conv.0",
                shape=conv_shape,
                dtype=torch.bfloat16,
                layer_ids=layers,
                slice_axis=0,
            ),
            TensorPartDecl(
                name="temporal",
                shape=temporal_shape,
                dtype=torch.float32,
                layer_ids=layers,
                slice_axis=0,
            ),
        )
    )
    raw = torch.zeros(num_slots * layout.entry_bytes, dtype=torch.uint8, device="cuda")
    views = layout.views(raw, num_slots, entry_bytes=layout.entry_bytes)
    strided_pool = views["temporal"][layer]
    contiguous_pool = torch.zeros(
        num_slots, *temporal_shape, dtype=torch.float32, device="cuda"
    )

    # The pool under test really is a strided unified-memory view: a wider
    # slot pitch than one packed slot, with each slot's block contiguous.
    assert not strided_pool.is_contiguous()
    assert strided_pool.stride(0) > H * V * K
    assert strided_pool[0].is_contiguous()

    # Random (non-zero) initial state, so the kernel must read the strided
    # pool as well as write it back.
    torch.manual_seed(20261127)
    init_state = torch.randn(
        len(slots), *temporal_shape, dtype=torch.float32, device="cuda"
    )
    strided_pool[slots] = init_state
    contiguous_pool[slots] = init_state

    A_log = torch.rand(H, device="cuda") * 1.5
    dt_bias = torch.randn(H, device="cuda") * 0.5
    scale = K**-0.5

    def run_step(pool, lens):
        cu = torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()],
            dtype=torch.int64,
            device="cuda",
        )
        total = int(cu[-1])
        q = torch.nn.functional.silu(
            torch.randn(total, H, K, device="cuda", dtype=torch.bfloat16)
        )
        k = torch.nn.functional.silu(
            torch.randn(total, H, K, device="cuda", dtype=torch.bfloat16)
        )
        v = torch.randn(total, H, V, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(total, H, device="cuda", dtype=torch.bfloat16)
        g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias)
        beta = torch.sigmoid(torch.randn(total, H, device="cuda", dtype=torch.bfloat16))
        return fused_recurrent_gated_delta_rule_update(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            scale=scale,
            initial_state_source=pool,
            initial_state_indices=slots,
            # Jet passes int64 query_start_loc as cu_seqlens.
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
        )

    for step, lens in enumerate(((5, 3, 7), (1, 1, 1))):
        raw_before = raw.clone()
        strided_before = strided_pool.clone()
        contiguous_before = contiguous_pool.clone()
        other_layers_before = {
            other: views["temporal"][other].clone()
            for other in range(num_layers)
            if other != layer
        }
        other_slots = [s for s in range(num_slots) if s not in slots.tolist()]

        # Same inputs for both pools, so any difference is the pool layout.
        torch.manual_seed(100 + step)
        out_strided = run_step(strided_pool, lens)
        torch.manual_seed(100 + step)
        out_contiguous = run_step(contiguous_pool, lens)

        torch.testing.assert_close(out_strided, out_contiguous, rtol=1e-2, atol=1e-2)
        # The update must have landed in the strided pool, not in a copy.
        assert not torch.equal(strided_pool[slots], strided_before[slots])
        torch.testing.assert_close(
            strided_pool[slots],
            contiguous_pool[slots],
            rtol=1e-5,
            atol=1e-5,
        )
        # Nothing else may change: other slots, other layers, and every other
        # byte of the arena.
        assert torch.equal(strided_pool[other_slots], strided_before[other_slots])
        assert torch.equal(contiguous_pool[other_slots], contiguous_before[other_slots])
        for other, before in other_layers_before.items():
            assert torch.equal(views["temporal"][other], before)
        mask = torch.ones_like(raw, dtype=torch.bool)
        base = strided_pool.storage_offset() * 4
        for s in slots.tolist():
            start = base + s * strided_pool.stride(0) * 4
            mask[start : start + H * V * K * 4] = False
        assert torch.equal(raw[mask], raw_before[mask])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
