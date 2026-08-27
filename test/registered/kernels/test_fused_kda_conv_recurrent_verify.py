import sys

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_kda_conv_recurrent_verify import (
    fused_kda_conv_gating_verify,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_update,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

_DEVICE = "cuda"

_CASES = [
    (1, 4, 4, 4, 128, 128, 4, False, None, False, 1),
    (1, 4, 4, 4, 128, 128, 4, True, None, False, 2),
    (1, 4, 4, 4, 128, 128, 4, True, 2.0, False, 3),
    (3, 4, 4, 4, 128, 128, 4, True, None, False, 4),
    (3, 4, 4, 4, 128, 128, 4, True, None, True, 5),
    (2, 3, 4, 4, 128, 128, 4, True, None, False, 6),
    (2, 8, 2, 2, 128, 128, 4, True, 1.5, False, 7),
    (1, 4, 8, 8, 64, 64, 4, True, None, False, 8),
]

# ReplaySSM ring-write cases: _CASES plus HV != H shapes, which exercise the
# per-k-head rawk vs per-v-head g/beta writer split (the GQA hazard: a wrong
# head index scribbles another head's ring silently).
_RING_CASES = _CASES + [
    (2, 4, 2, 4, 128, 128, 4, True, None, False, 20),
    (1, 5, 2, 8, 64, 64, 4, True, 1.5, False, 21),
    (3, 4, 4, 8, 128, 128, 4, True, None, True, 22),
]

# Power of two >= 2 * max draft T in _RING_CASES, matching the pool invariant
# (memory_pool.py: ring length must be a power of two >= 2 * draft tokens).
_RING_LEN = 16


def _make_ring_buffers(H, HV, K, V):
    # Garbage-filled so a full-tensor bitwise compare proves both that written
    # positions match and that neither kernel scribbles outside them.
    slots = 8
    return {
        "rawv": torch.randn(
            slots, HV, _RING_LEN, V, device=_DEVICE, dtype=torch.bfloat16
        ),
        "rawk": torch.randn(
            slots, H, _RING_LEN, K, device=_DEVICE, dtype=torch.bfloat16
        ),
        "g": torch.randn(slots, HV, _RING_LEN, K, device=_DEVICE, dtype=torch.float32),
        "beta": torch.randn(slots, HV, _RING_LEN, device=_DEVICE, dtype=torch.float32),
    }


def _make_inputs(B, T, H, HV, K, V, W, has_bias, neg_slot, seed):
    torch.manual_seed(seed)
    dim = 2 * H * K + HV * V
    seq_len = B * T
    lines = slots = 8

    inputs = {
        "mixed": torch.randn(seq_len, dim, device=_DEVICE, dtype=torch.bfloat16) * 0.5,
        "w": torch.randn(dim, W, device=_DEVICE, dtype=torch.bfloat16) * 0.3,
        "bias": (
            torch.randn(dim, device=_DEVICE, dtype=torch.bfloat16) * 0.1
            if has_bias
            else None
        ),
        "a": torch.randn(seq_len, HV * K, device=_DEVICE, dtype=torch.bfloat16) * 0.5,
        "b": torch.randn(seq_len, HV, device=_DEVICE, dtype=torch.bfloat16),
        "A_log": torch.randn(HV, device=_DEVICE, dtype=torch.float32) * 0.5,
        "dt_bias": torch.randn(HV * K, device=_DEVICE, dtype=torch.float32) * 0.5,
        # Pool layouts mirroring MambaPool: conv [lines, state_len, dim] (then
        # transposed), ssm [slots, HV, V, K] fp32, window [lines, T, W-1, dim],
        # intermediate ssm cache [lines, T, HV, V, K] fp32.
        "conv_pool": torch.randn(
            lines, W - 1, dim, device=_DEVICE, dtype=torch.bfloat16
        ),
        "ssm": torch.randn(slots, HV, V, K, device=_DEVICE, dtype=torch.float32) * 0.2,
        "win_pool": torch.zeros(
            lines, T, W - 1, dim, device=_DEVICE, dtype=torch.bfloat16
        ),
        "inter_ssm": torch.zeros(
            lines, T, HV, V, K, device=_DEVICE, dtype=torch.float32
        ),
    }
    idx_vals = list(range(2, 2 + B))
    if neg_slot and B >= 2:
        idx_vals[1] = -1
    inputs["idx_vals"] = idx_vals
    inputs["cache_indices"] = torch.tensor(idx_vals, device=_DEVICE, dtype=torch.int32)
    inputs["inter_indices"] = torch.arange(B, device=_DEVICE, dtype=torch.int32)
    return inputs


def _run_reference(inp, B, T, H, HV, K, V, lower_bound, rings=None):
    dim = 2 * H * K + HV * V
    seq_len = B * T
    conv = inp["conv_pool"].clone()
    ssm = inp["ssm"].clone()
    win = inp["win_pool"].clone()
    ic = inp["inter_ssm"].clone() if rings is None else None

    x3 = inp["mixed"].reshape(B, T, dim).transpose(1, 2)
    out3 = causal_conv1d_update(
        x3,
        conv.transpose(-1, -2),
        inp["w"],
        inp["bias"],
        activation="silu",
        conv_state_indices=inp["cache_indices"],
        intermediate_conv_window=win.transpose(-1, -2),
        intermediate_state_indices=inp["inter_indices"],
    )
    mixed_out = out3.transpose(1, 2).reshape(seq_len, dim)
    q, k, v = mixed_out.split([H * K, H * K, HV * V], dim=-1)
    q = q.unflatten(-1, (H, K)).unsqueeze(0)
    k = k.unflatten(-1, (H, K)).unsqueeze(0)
    v = v.unflatten(-1, (HV, V)).unsqueeze(0)
    cu = torch.arange(0, B + 1, device=_DEVICE, dtype=torch.int32) * T
    o = fused_sigmoid_gating_delta_rule_update(
        A_log=inp["A_log"],
        a=inp["a"],
        dt_bias=inp["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=inp["b"],
        initial_state_source=ssm,
        initial_state_indices=inp["cache_indices"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        is_kda=True,
        disable_state_update=True,
        # ReplaySSM mode (rings set) drops the per-step snapshots, exactly as
        # the production GDN/KDA backends pass None + cache_ring.
        intermediate_states_buffer=ic,
        intermediate_state_indices=inp["inter_indices"] if rings is None else None,
        cache_steps=T,
        retrieve_parent_token=None,
        lower_bound=lower_bound,
        cache_ring=rings is not None,
        replayssm_rawv=rings["rawv"] if rings is not None else None,
        replayssm_rawk=rings["rawk"] if rings is not None else None,
        replayssm_g=rings["g"] if rings is not None else None,
        replayssm_beta=rings["beta"] if rings is not None else None,
    )
    return o, conv, win, ic


def _run_fused(inp, B, T, H, HV, K, V, lower_bound, num_warps, rings=None):
    conv = inp["conv_pool"].clone()
    ssm = inp["ssm"].clone()
    win = inp["win_pool"].clone()
    ic = inp["inter_ssm"].clone() if rings is None else None

    o = fused_kda_conv_gating_verify(
        mixed_qkv=inp["mixed"],
        conv_weight=inp["w"],
        conv_bias=inp["bias"],
        conv_state=conv.transpose(-1, -2),
        conv_state_indices=inp["cache_indices"],
        intermediate_conv_window=win.transpose(-1, -2),
        intermediate_state_indices=inp["inter_indices"],
        a=inp["a"],
        b=inp["b"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        ssm_states=ssm,
        cache_indices=inp["cache_indices"],
        intermediate_states_buffer=ic,
        scale=K**-0.5,
        T=T,
        num_q_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        lower_bound=lower_bound,
        num_warps=num_warps,
        cache_ring=rings is not None,
        replayssm_rawv=rings["rawv"] if rings is not None else None,
        replayssm_rawk=rings["rawk"] if rings is not None else None,
        replayssm_g=rings["g"] if rings is not None else None,
        replayssm_beta=rings["beta"] if rings is not None else None,
    )
    return o, conv, win, ic


def _compare_case(case, num_warps, use_ring=False):
    B, T, H, HV, K, V, W, has_bias, lower_bound, neg_slot, seed = case
    inp = _make_inputs(B, T, H, HV, K, V, W, has_bias, neg_slot, seed)
    if use_ring:
        template = _make_ring_buffers(H, HV, K, V)
        rings_ref = {name: buf.clone() for name, buf in template.items()}
        rings_fus = {name: buf.clone() for name, buf in template.items()}
    else:
        rings_ref = rings_fus = None
    o_ref, conv_ref, win_ref, ic_ref = _run_reference(
        inp, B, T, H, HV, K, V, lower_bound, rings=rings_ref
    )
    o_fus, conv_fus, win_fus, ic_fus = _run_fused(
        inp, B, T, H, HV, K, V, lower_bound, num_warps, rings=rings_fus
    )

    idx_vals = inp["idx_vals"]
    valid_rows = [i for i, slot in enumerate(idx_vals) if slot >= 0]
    touched_slots = [slot for slot in idx_vals if slot >= 0]

    o_ref_v = o_ref.reshape(B, T, HV, V)[valid_rows]
    o_fus_v = o_fus.reshape(B, T, HV, V)[valid_rows]
    assert torch.equal(o_ref_v, o_fus_v)
    assert torch.equal(conv_ref[touched_slots], conv_fus[touched_slots])
    assert torch.equal(win_ref[valid_rows], win_fus[valid_rows])
    if use_ring:
        # Full-tensor bitwise: ring values are elementwise (conv FMA chain,
        # gate, sigmoid), upstream of every tl.sum, so they are exact at any
        # num_warps; the shared garbage init makes any out-of-slot or
        # negative-slot scribble a mismatch.
        for name in ("rawv", "rawk", "g", "beta"):
            assert torch.equal(rings_ref[name], rings_fus[name]), name
    else:
        torch.testing.assert_close(
            ic_ref[valid_rows], ic_fus[valid_rows], atol=4e-3, rtol=0
        )


@pytest.mark.parametrize("case", _CASES)
def test_matches_unfused_reference(case):
    _compare_case(case, num_warps=4)


@pytest.mark.parametrize("case", _RING_CASES)
def test_replayssm_ring_matches_unfused(case):
    _compare_case(case, num_warps=4, use_ring=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
