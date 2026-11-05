"""gfx950 KDA verify must preserve outputs, ring writes and snapshots within tolerance."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.linear.fused_kda_conv_recurrent_verify import (
    fused_kda_conv_gating_verify,
)
from sglang.kernels.ops.attention.linear.kda_verify_gfx950 import (
    kda_verify_gfx950,
    supports_kda_verify_gfx950,
)
from sglang.srt.layers.attention.linear.kda_backend_hip import (
    kda_chain_verify_gfx950,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=65, suite="stage-b-test-1-gpu-small-amd-mi35x")

_DEVICE = "cuda"
_T, _K, _W, _RING_LEN = 8, 128, 4, 16


def _inputs(B, H, seed, T=_T):
    torch.manual_seed(seed)
    dim = 3 * H * _K
    seq = B * T

    def rnd(*shape, dt=torch.bfloat16):
        return torch.randn(*shape, device=_DEVICE, dtype=dt)

    slots = B + 2
    inp = dict(
        mixed=rnd(seq, dim) * 0.5,
        w=rnd(dim, _W, dt=torch.float32) * 0.3,
        a=rnd(seq, H * _K) * 0.5,
        b=rnd(seq, H),
        A_log=rnd(H, dt=torch.float32) * 0.5,
        dt_bias=rnd(H * _K, dt=torch.float32) * 0.5,
        conv=rnd(slots, _W - 1, dim),
        ssm=rnd(slots, H, _K, _K, dt=torch.float32) * 0.2,
        win=rnd(slots, T, _W - 1, dim),
        rawv=rnd(slots, H, _RING_LEN, _K),
        rawk=rnd(slots, H, _RING_LEN, _K),
        g=rnd(slots, H, _RING_LEN, _K, dt=torch.float32),
        beta=rnd(slots, H, _RING_LEN, dt=torch.float32),
        # Row 0 is a padded graph slot (-1): neither kernel may touch its state.
        idx=torch.tensor([-1] + list(range(1, B)), device=_DEVICE, dtype=torch.int32)
        if B > 1
        else torch.tensor([1], device=_DEVICE, dtype=torch.int32),
    )
    # conv window / snapshot row per request: in production its batch row, not its slot
    inp["rows"] = inp["idx"]
    return inp


def _dispatchable(inp, H, snapshots=None, T=_T):
    return supports_kda_verify_gfx950(
        num_q_heads=H,
        num_v_heads=H,
        head_k_dim=_K,
        head_v_dim=_K,
        T=T,
        mixed_qkv=inp["mixed"],
        conv_weight=inp["w"],
        conv_state=inp["conv"],
        a=inp["a"],
        b=inp["b"],
        ssm_states=inp["ssm"],
        replayssm_rawv=inp["rawv"] if snapshots is None else None,
        intermediate_states_buffer=snapshots,
        intermediate_conv_window=inp["win"],
    )


def _assert_outputs_close(o_new, o_ref, idx, T=_T):
    rows = (idx >= 0).repeat_interleave(T)
    torch.testing.assert_close(
        o_new[0][rows].float(), o_ref[0][rows].float(), rtol=2e-2, atol=2e-3
    )


def _triton_verify(**kwargs):
    """The Triton chain verify, with the bias-free conv and shared slots the HIP kernel takes."""
    return fused_kda_conv_gating_verify(
        conv_bias=None, conv_state_indices=kwargs["cache_indices"], **kwargs
    )


def _run(fn, inp, H, lower_bound, snapshots=None, T=_T):
    ring = snapshots is None
    return fn(
        mixed_qkv=inp["mixed"],
        conv_weight=inp["w"],
        conv_state=inp["conv"].transpose(-1, -2),
        intermediate_conv_window=inp["win"].transpose(-1, -2),
        intermediate_state_indices=inp["rows"],
        a=inp["a"],
        b=inp["b"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        ssm_states=inp["ssm"],
        cache_indices=inp["idx"],
        intermediate_states_buffer=snapshots,
        scale=_K**-0.5,
        T=T,
        num_q_heads=H,
        num_v_heads=H,
        head_k_dim=_K,
        head_v_dim=_K,
        lower_bound=lower_bound,
        cache_ring=ring,
        replayssm_rawv=inp["rawv"] if ring else None,
        replayssm_rawk=inp["rawk"] if ring else None,
        replayssm_g=inp["g"] if ring else None,
        replayssm_beta=inp["beta"] if ring else None,
    )


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
@pytest.mark.parametrize("B", [1, 3, 8, 24])
@pytest.mark.parametrize("H", [12, 16])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
# 6: the AgentX recipe's verify block; tokens past T leave their lanes idle
@pytest.mark.parametrize("T", [8, 6, 3])
def test_matches_triton_chain_verify(B, H, lower_bound, T):
    ref, new = (_inputs(B, H, seed=B * 31 + H, T=T) for _ in range(2))
    assert _dispatchable(new, H, T=T)
    o_ref = _run(_triton_verify, ref, H, lower_bound, T=T)
    o_new = _run(kda_verify_gfx950, new, H, lower_bound, T=T)
    for name in ("rawv", "beta", "win", "conv", "ssm"):
        assert torch.equal(ref[name], new[name]), name
    torch.testing.assert_close(new["rawk"], ref["rawk"], rtol=8e-3, atol=1e-5)
    torch.testing.assert_close(new["g"], ref["g"], rtol=1e-5, atol=1e-6)
    _assert_outputs_close(o_new, o_ref, ref["idx"], T=T)


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
@pytest.mark.parametrize("B", [1, 3, 8, 24])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("T", [8, 6])
def test_snapshots_match_triton_chain_verify(B, lower_bound, T):
    """Without the ring (the monokernel configs), every step's state must land where the
    Triton kernel puts it, and a padded row's snapshots must stay untouched."""
    H = 12
    ref, new = (_inputs(B, H, seed=B * 17 + 5, T=T) for _ in range(2))
    rows = B + 2
    snaps = [
        torch.full((rows, T, H, _K, _K), 7.0, device=_DEVICE, dtype=torch.float32)
        for _ in range(2)
    ]
    assert _dispatchable(new, H, snaps[1], T=T)
    o_ref = _run(_triton_verify, ref, H, lower_bound, snaps[0], T=T)
    o_new = _run(kda_verify_gfx950, new, H, lower_bound, snaps[1], T=T)
    for name in ("win", "conv", "ssm"):
        assert torch.equal(ref[name], new[name]), name
    # The gate decays differ within an ulp and compound over T steps of O(1) states:
    # 4.8e-5 at most on MI350X (B=24).
    torch.testing.assert_close(snaps[1], snaps[0], rtol=1e-3, atol=1e-4)
    _assert_outputs_close(o_new, o_ref, ref["idx"], T=T)


@pytest.mark.parametrize("field", ["conv", "win", "w"])
def test_conv_inputs_the_kernel_cannot_read_fall_back(field):
    # The kernel reads the conv state and window as BF16 and a width-4 weight.
    inp = _inputs(1, 16, seed=0)
    assert _dispatchable(inp, 16)
    inp[field] = inp[field][:, :3] if field == "w" else inp[field].float()
    assert not _dispatchable(inp, 16)


# Mamba slots of six requests (row 1 is a padded graph row) beside their batch rows
# 0..5: no slot equals its row, so a kernel that swaps the two indexings shows.
_SLOTS = [7, -1, 4, 0, 6, 1]


def _production_indices(inp):
    inp["idx"] = torch.tensor(_SLOTS, device=_DEVICE, dtype=torch.int32)
    inp["rows"] = torch.arange(len(_SLOTS), device=_DEVICE, dtype=torch.int32)
    return inp


def _layer(inp, H, lower_bound):
    return SimpleNamespace(
        bias=None,
        conv_weights=inp["w"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        num_q_heads=H,
        num_v_heads=H,
        head_k_dim=_K,
        head_v_dim=_K,
        lower_bound=lower_bound,
    )


def _run_srt(inp, layer, snapshots=None, **overrides):
    """kda_chain_verify_gfx950 called as the KDA backend calls it for a chain verify."""
    ring_kwargs = (
        dict(
            cache_ring=True,
            replayssm_rawv=inp["rawv"],
            replayssm_rawk=inp["rawk"],
            replayssm_g=inp["g"],
            replayssm_beta=inp["beta"],
        )
        if snapshots is None
        else {}
    )
    kwargs = dict(
        mixed_qkv=inp["mixed"],
        a=inp["a"],
        b=inp["b"],
        draft_token_num=_T,
        conv_states=inp["conv"],
        conv_state_indices=inp["idx"],
        ssm_states=inp["ssm"],
        intermediate_state_cache=snapshots,
        intermediate_conv_window_cache=inp["win"],
        intermediate_state_indices=inp["rows"],
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        ring_kwargs=ring_kwargs,
    )
    kwargs.update(overrides)
    return kda_chain_verify_gfx950(layer, **kwargs)


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
@pytest.mark.parametrize("ring", [True, False], ids=["ring", "snapshots"])
def test_srt_verify_indexes_state_by_slot_and_window_by_batch_row(ring):
    """The KDA backend indexes the conv / SSM state and the ring by Mamba slot but the
    conv window and the snapshots by batch row: each buffer must follow its own index."""
    B, H, lower_bound = len(_SLOTS), 12, -5.0
    ref, new = (_production_indices(_inputs(B, H, seed=11)) for _ in range(2))
    snaps = [
        None
        if ring
        else torch.full(
            (B + 2, _T, H, _K, _K), 7.0, device=_DEVICE, dtype=torch.float32
        )
        for _ in range(2)
    ]
    o_ref = _run(_triton_verify, ref, H, lower_bound, snaps[0])
    o_new = _run_srt(new, _layer(new, H, lower_bound), snaps[1])
    assert o_new is not None
    for name in ("win", "conv", "ssm") + (("rawv", "beta") if ring else ()):
        assert torch.equal(ref[name], new[name]), name
    if ring:
        torch.testing.assert_close(new["rawk"], ref["rawk"], rtol=8e-3, atol=1e-5)
        torch.testing.assert_close(new["g"], ref["g"], rtol=1e-5, atol=1e-6)
    else:
        # a misindexed row is off by O(1); the gate's ulp drift reaches 3.5e-4 here
        torch.testing.assert_close(snaps[1], snaps[0], rtol=1e-3, atol=1e-3)
    _assert_outputs_close(o_new, o_ref, ref["idx"])


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
def test_reads_mixed_qkv_and_b_as_views_of_the_fused_projection():
    """Kimi-K3.1 passes mixed_qkv and b as torch.split views of one [q,k,v,g | f_a | b |
    pad] GEMM output: read at that row stride they must give what dense copies give."""
    B, H = 3, 12
    dense, views = (_inputs(B, H, seed=23) for _ in range(2))
    sizes = [3 * H * _K, H * _K, _K, H]
    sizes.append(-sum(sizes) % 8)  # the TP8 in-proj rows pad to 8 (6288 columns)
    front = torch.randn(B * _T, sum(sizes), device=_DEVICE, dtype=torch.bfloat16)
    views["mixed"], _, _, views["b"], _ = torch.split(front, sizes, dim=-1)
    views["mixed"].copy_(dense["mixed"])
    views["b"].copy_(dense["b"])
    assert _dispatchable(views, H)
    o_dense = _run(kda_verify_gfx950, dense, H, -5.0)
    o_views = _run(kda_verify_gfx950, views, H, -5.0)
    for name in ("rawv", "rawk", "g", "beta", "win", "conv", "ssm"):
        assert torch.equal(dense[name], views[name]), name
    rows = (dense["idx"] >= 0).repeat_interleave(_T)
    assert torch.equal(o_dense[0][rows], o_views[0][rows])


@pytest.mark.parametrize(
    "unsupported", ["tree", "conv_bias", "bf16_A_log", "ring_shorter_than_T"]
)
def test_srt_verify_declines_what_the_kernel_cannot_serve(unsupported):
    """These verifies must fall back to the Triton kernels (None), not run this one."""
    B, H = len(_SLOTS), 12
    inp = _production_indices(_inputs(B, H, seed=0))
    layer = _layer(inp, H, -5.0)
    overrides = {}
    if unsupported == "tree":
        tree = torch.full((B, _T), -1, device=_DEVICE, dtype=torch.int64)
        overrides = dict(
            retrieve_next_token=tree,
            retrieve_next_sibling=tree,
            retrieve_parent_token=tree,
        )
    elif unsupported == "conv_bias":
        layer.bias = torch.zeros(inp["w"].shape[0], device=_DEVICE)
    elif unsupported == "bf16_A_log":
        layer.A_log = layer.A_log.bfloat16()
    else:
        # packed rings of T - 1 steps: only their length is unsupported
        for name in ("rawv", "rawk", "g", "beta"):
            inp[name] = torch.zeros_like(inp[name][:, :, : _T - 1]).contiguous()
    assert _run_srt(inp, layer, **overrides) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
