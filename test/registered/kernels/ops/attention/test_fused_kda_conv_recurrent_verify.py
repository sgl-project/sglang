import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.linear.fused_kda_conv_recurrent_verify import (
    fused_kda_conv_gating_verify,
)
from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.linear.kda_verify_gfx950 import (
    kda_verify_gfx950,
    supports_kda_verify_gfx950,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_update,
)
from sglang.srt.layers.attention.linear.kda_backend_hip import kda_chain_verify_gfx950
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: the gfx950 product-rounding boundary must stay bit-exact.
register_amd_ci(est_time=155, suite="stage-b-test-1-gpu-small-amd-mi35x")

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

_RING_CASES = list(_CASES)

_CASES += [
    # GLM-5.3 Flash TP4: 16 local heads, six-token EAGLE verification,
    # and a negative safe-gate lower bound.
    (1, 6, 16, 16, 128, 128, 4, False, -5.0, False, 9),
    (16, 8, 16, 16, 128, 128, 4, False, -5.0, True, 10),
]

# ReplaySSM ring-write cases add HV != H shapes, which exercise the
# per-k-head rawk vs per-v-head g/beta writer split (the GQA hazard: a wrong
# head index scribbles another head's ring silently).
_RING_CASES += [
    (2, 4, 2, 4, 128, 128, 4, True, None, False, 20),
    (1, 5, 2, 8, 64, 64, 4, True, 1.5, False, 21),
    (3, 4, 4, 8, 128, 128, 4, True, None, True, 22),
]

# Power of two >= 2 * max draft T in _RING_CASES, matching the pool invariant
# (memory_pool.py: ring length must be a power of two >= 2 * draft tokens).
_RING_LEN = 16


def _make_ring_buffers(H, HV, K, V, slots=8):
    # Garbage-filled so a full-tensor bitwise compare proves both that written
    # positions match and that neither kernel scribbles outside them.
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


def _make_inputs(
    B,
    T,
    H,
    HV,
    K,
    V,
    W,
    has_bias,
    neg_slot,
    seed,
    weight_dtype=torch.bfloat16,
):
    torch.manual_seed(seed)
    dim = 2 * H * K + HV * V
    seq_len = B * T
    lines = slots = max(8, B + 2)

    inputs = {
        "mixed": torch.randn(seq_len, dim, device=_DEVICE, dtype=torch.bfloat16) * 0.5,
        "w": torch.randn(dim, W, device=_DEVICE, dtype=weight_dtype) * 0.3,
        "bias": (
            torch.randn(dim, device=_DEVICE, dtype=weight_dtype) * 0.1
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


def _fused_args(inp, B, T, H, HV, K, V, lower_bound, num_warps, rings=None):
    conv = inp["conv_pool"].clone()
    ssm = inp["ssm"].clone()
    win = inp["win_pool"].clone()
    ic = inp["inter_ssm"].clone() if rings is None else None

    return dict(
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
        **({"num_warps": num_warps} if num_warps is not None else {}),
        cache_ring=rings is not None,
        replayssm_rawv=rings["rawv"] if rings is not None else None,
        replayssm_rawk=rings["rawk"] if rings is not None else None,
        replayssm_g=rings["g"] if rings is not None else None,
        replayssm_beta=rings["beta"] if rings is not None else None,
    )


def _run_fused(inp, B, T, H, HV, K, V, lower_bound, num_warps, rings=None):
    args = _fused_args(inp, B, T, H, HV, K, V, lower_bound, num_warps, rings)
    o = fused_kda_conv_gating_verify(**args)
    return (
        o,
        args["conv_state"].transpose(-1, -2),
        args["intermediate_conv_window"].transpose(-1, -2),
        args["intermediate_states_buffer"],
    )


def _compare_case(case, num_warps=None, use_ring=False, weight_dtype=torch.bfloat16):
    B, T, H, HV, K, V, W, has_bias, lower_bound, neg_slot, seed = case
    inp = _make_inputs(B, T, H, HV, K, V, W, has_bias, neg_slot, seed, weight_dtype)
    if use_ring:
        template = _make_ring_buffers(H, HV, K, V)
        rings_ref = {name: buf.clone() for name, buf in template.items()}
        rings_fus = {name: buf.clone() for name, buf in template.items()}
    else:
        rings_ref = rings_fus = None
    o_ref, _, win_ref, ic_ref = _run_reference(
        inp, B, T, H, HV, K, V, lower_bound, rings=rings_ref
    )
    o_fus, conv_fus, win_fus, ic_fus = _run_fused(
        inp, B, T, H, HV, K, V, lower_bound, num_warps, rings=rings_fus
    )

    idx_vals = inp["idx_vals"]
    valid_rows = [i for i, slot in enumerate(idx_vals) if slot >= 0]

    o_ref_v = o_ref.reshape(B, T, HV, V)[valid_rows]
    o_fus_v = o_fus.reshape(B, T, HV, V)[valid_rows]
    # One bf16 ulp: the fused and reference tiles reduce K in different orders.
    torch.testing.assert_close(o_fus_v, o_ref_v, rtol=2**-7, atol=1e-7)
    # conv_state is read-only in verify; the commit scatter advances it.
    assert torch.equal(inp["conv_pool"], conv_fus)
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


@pytest.mark.parametrize("case_index", [8, 9])
def test_glm_fp32_weights_match_unfused_reference(case_index):
    _compare_case(_CASES[case_index], weight_dtype=torch.float32)


@pytest.mark.skipif(not torch.version.hip, reason="ROCm default launch shape")
@pytest.mark.parametrize("case_index", [8, 9])
def test_glm_fp32_weights_bit_exact_on_rocm(case_index):
    # One wave keeps the reference reduction order and separately rounded
    # conv products keep LLVM from contracting them into FMAs.
    B, T, H, HV, K, V, W, bias, lower, neg_slot, seed = _CASES[case_index]
    inp = _make_inputs(B, T, H, HV, K, V, W, bias, neg_slot, seed, torch.float32)
    o_ref, _, _, ic_ref = _run_reference(inp, B, T, H, HV, K, V, lower)
    o_fus, _, _, ic_fus = _run_fused(inp, B, T, H, HV, K, V, lower, None)
    valid = [i for i, slot in enumerate(inp["idx_vals"]) if slot >= 0]
    assert torch.equal(
        o_fus.reshape(B, T, HV, V)[valid], o_ref.reshape(B, T, HV, V)[valid]
    )
    assert torch.equal(ic_fus[valid], ic_ref[valid])


def test_output_does_not_depend_on_cta_scheduling():
    """The verify output must not change with how the CTAs happen to be
    scheduled. H=1 with HV=16 shares one Q/K history across 16 V tiles."""
    if torch.version.hip or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("green contexts need CUDA SM90 or newer")
    from flashinfer.green_ctx import split_device_green_ctx_by_sm_count

    case = (1, 6, 1, 16, 128, 128, 4, False, None, False, 1)
    B, T, H, HV, K, V, W, has_bias, lower_bound, neg_slot, seed = case
    inp = _make_inputs(B, T, H, HV, K, V, W, has_bias, neg_slot, seed)
    full = _run_fused(inp, B, T, H, HV, K, V, lower_bound, num_warps=4)[0]

    streams, _ = split_device_green_ctx_by_sm_count(torch.device("cuda:0"), [8])
    # The green stream is non-blocking, so it must be told to wait for the
    # inputs produced above; synchronize() afterwards only waits on the consumer.
    streams[0].wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(streams[0]):
        squeezed = _run_fused(inp, B, T, H, HV, K, V, lower_bound, num_warps=4)[0]
    streams[0].synchronize()
    assert torch.equal(full, squeezed)


@pytest.mark.parametrize("case", _RING_CASES)
def test_replayssm_ring_matches_unfused(case):
    _compare_case(case, num_warps=4, use_ring=True)


def _gfx950_case(B, H, seed, T=8, lower_bound=-5.0, snapshot=False):
    inp = _make_inputs(B, T, H, H, 128, 128, 4, False, False, seed, torch.float32)
    inp["cache_indices"] = torch.tensor(
        [-1] + list(range(1, B)) if B > 1 else [1], device=_DEVICE, dtype=torch.int32
    )
    inp["inter_indices"] = inp["cache_indices"]
    inp["win_pool"].normal_()  # untouched window rows must keep their sentinel values
    inp["inter_ssm"].fill_(7.0)
    rings = (
        None if snapshot else _make_ring_buffers(H, H, 128, 128, inp["ssm"].shape[0])
    )
    args = _fused_args(inp, B, T, H, H, 128, 128, lower_bound, None, rings)
    del args["conv_bias"], args["conv_state_indices"]
    return args


def _gfx950_supported(args):
    return supports_kda_verify_gfx950(
        **{
            key: args[key]
            for key in (
                "num_q_heads",
                "num_v_heads",
                "head_k_dim",
                "head_v_dim",
                "T",
                "mixed_qkv",
                "conv_weight",
                "conv_state",
                "a",
                "b",
                "ssm_states",
                "replayssm_rawv",
                "intermediate_states_buffer",
                "intermediate_conv_window",
            )
        }
    )


def _gfx950_reference(args):
    return fused_kda_conv_gating_verify(
        conv_bias=None, conv_state_indices=args["cache_indices"], **args
    )


def _assert_gfx950_result(ref, new, o_ref, o_new, snapshot_atol=1e-4):
    ring = ref["cache_ring"]
    for name in ("conv_state", "intermediate_conv_window", "ssm_states") + (
        ("replayssm_rawv", "replayssm_beta") if ring else ()
    ):
        assert torch.equal(ref[name], new[name]), name
    if ring:
        torch.testing.assert_close(
            new["replayssm_rawk"], ref["replayssm_rawk"], rtol=8e-3, atol=1e-5
        )
        torch.testing.assert_close(
            new["replayssm_g"], ref["replayssm_g"], rtol=1e-5, atol=1e-6
        )
    else:
        torch.testing.assert_close(
            new["intermediate_states_buffer"],
            ref["intermediate_states_buffer"],
            rtol=1e-3,
            atol=snapshot_atol,
        )
    rows = (ref["cache_indices"] >= 0).repeat_interleave(ref["T"])
    torch.testing.assert_close(
        o_new[0][rows].float(), o_ref[0][rows].float(), rtol=2e-2, atol=2e-3
    )


def _gfx950_srt(args, *, conv_bias=None, **overrides):
    layer = SimpleNamespace(
        bias=conv_bias,
        conv_weights=args["conv_weight"],
        **{
            key: args[key]
            for key in (
                "A_log",
                "dt_bias",
                "num_q_heads",
                "num_v_heads",
                "head_k_dim",
                "head_v_dim",
                "lower_bound",
            )
        },
    )
    kwargs = dict(
        mixed_qkv=args["mixed_qkv"],
        a=args["a"],
        b=args["b"],
        draft_token_num=args["T"],
        conv_states=args["conv_state"].transpose(-1, -2),
        conv_state_indices=args["cache_indices"],
        ssm_states=args["ssm_states"],
        intermediate_state_cache=args["intermediate_states_buffer"],
        intermediate_conv_window_cache=args["intermediate_conv_window"].transpose(
            -1, -2
        ),
        intermediate_state_indices=args["intermediate_state_indices"],
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        ring_kwargs={
            key: args[key]
            for key in (
                "cache_ring",
                "replayssm_rawv",
                "replayssm_rawk",
                "replayssm_g",
                "replayssm_beta",
            )
        }
        if args["cache_ring"]
        else {},
    )
    kwargs.update(overrides)
    return kda_chain_verify_gfx950(layer, **kwargs)


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
class TestGfx950Verify:
    @pytest.mark.parametrize("B", [1, 24])
    @pytest.mark.parametrize("lower_bound", [-5.0, None])
    @pytest.mark.parametrize(
        "T,snapshot", [(8, False), (6, False), (3, False), (8, True), (6, True)]
    )
    def test_matches_triton_chain_verify(self, B, lower_bound, T, snapshot):
        """Keep every (BV, T, bounded gate, snapshot) specialization and padded state."""
        H = 12 if snapshot or T == 8 else 16
        seed = B * 17 + 5 if snapshot else B * 31 + H
        ref, new = (
            _gfx950_case(B, H, seed, T, lower_bound, snapshot) for _ in range(2)
        )
        assert _gfx950_supported(new)
        _assert_gfx950_result(
            ref, new, _gfx950_reference(ref), kda_verify_gfx950(**new)
        )

    @pytest.mark.parametrize(
        "field", ["conv_state", "intermediate_conv_window", "conv_weight"]
    )
    def test_conv_inputs_the_kernel_cannot_read_fall_back(self, field):
        args = _gfx950_case(1, 16, seed=0)
        assert _gfx950_supported(args)
        args[field] = (
            args[field][:, :3] if field == "conv_weight" else args[field].float()
        )
        assert not _gfx950_supported(args)

    @pytest.mark.parametrize("ring", [True, False], ids=["ring", "snapshots"])
    def test_srt_verify_indexes_state_by_slot_and_window_by_batch_row(self, ring):
        # No physical slot equals its scratch row; -1 must not write any state.
        ref, new = (_gfx950_case(6, 12, seed=11, snapshot=not ring) for _ in range(2))
        for args in (ref, new):
            args["cache_indices"] = torch.tensor(
                [7, -1, 4, 0, 6, 1], device=_DEVICE, dtype=torch.int32
            )
            args["intermediate_state_indices"] = torch.arange(
                6, device=_DEVICE, dtype=torch.int32
            )
        o_new = _gfx950_srt(new)
        assert o_new is not None
        _assert_gfx950_result(
            ref, new, _gfx950_reference(ref), o_new, snapshot_atol=1e-3
        )

    def test_reads_mixed_qkv_and_b_as_views_of_the_fused_projection(self):
        B, H, T = 3, 12, 8
        dense, views = (_gfx950_case(B, H, seed=23) for _ in range(2))
        sizes = [3 * H * 128, H * 128, 128, H]
        sizes.append(-sum(sizes) % 8)  # TP8 in-projection rows pad to 6288 columns
        front = torch.randn(B * T, sum(sizes), device=_DEVICE, dtype=torch.bfloat16)
        views["mixed_qkv"], _, _, views["b"], _ = torch.split(front, sizes, dim=-1)
        views["mixed_qkv"].copy_(dense["mixed_qkv"])
        views["b"].copy_(dense["b"])
        assert _gfx950_supported(views)
        o_dense, o_views = kda_verify_gfx950(**dense), kda_verify_gfx950(**views)
        for name in (
            "replayssm_rawv",
            "replayssm_rawk",
            "replayssm_g",
            "replayssm_beta",
            "intermediate_conv_window",
            "conv_state",
            "ssm_states",
        ):
            assert torch.equal(dense[name], views[name]), name
        rows = (dense["cache_indices"] >= 0).repeat_interleave(T)
        assert torch.equal(o_dense[0][rows], o_views[0][rows])

    @pytest.mark.parametrize(
        "unsupported", ["tree", "conv_bias", "bf16_A_log", "ring_shorter_than_T"]
    )
    def test_srt_verify_declines_what_the_kernel_cannot_serve(self, unsupported):
        args = _gfx950_case(6, 12, seed=0)
        args["cache_indices"] = torch.tensor(
            [7, -1, 4, 0, 6, 1], device=_DEVICE, dtype=torch.int32
        )
        args["intermediate_state_indices"] = torch.arange(
            6, device=_DEVICE, dtype=torch.int32
        )
        overrides = {}
        if unsupported == "tree":
            tree = torch.full((6, 8), -1, device=_DEVICE, dtype=torch.int64)
            overrides = dict(
                retrieve_next_token=tree,
                retrieve_next_sibling=tree,
                retrieve_parent_token=tree,
            )
        elif unsupported == "conv_bias":
            overrides["conv_bias"] = torch.zeros(
                args["conv_weight"].shape[0], device=_DEVICE
            )
        elif unsupported == "bf16_A_log":
            args["A_log"] = args["A_log"].bfloat16()
        else:
            for name in (
                "replayssm_rawv",
                "replayssm_rawk",
                "replayssm_g",
                "replayssm_beta",
            ):
                args[name] = torch.zeros_like(args[name][:, :, :7]).contiguous()
        assert _gfx950_srt(args, **overrides) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
