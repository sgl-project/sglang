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

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")

_DEVICE = "cuda"

# Trailing flag: whether the fused output is bit-identical to the unfused pair
# for THIS case's inputs. Empirical per case, not implied by the shape -- the
# same shape diverges at one seed and matches at another (H=8/HV=8 is exact at
# seeds 1, 20 and 21 but needs rtol 5.5e-3 at seed 9), and larger head counts
# only raise the odds. A new case starts False and is promoted only once it is
# measured exact.
_CASES = [
    (1, 4, 4, 4, 128, 128, 4, False, None, False, 1, True),
    (1, 4, 4, 4, 128, 128, 4, True, None, False, 2, True),
    (1, 4, 4, 4, 128, 128, 4, True, 2.0, False, 3, True),
    (3, 4, 4, 4, 128, 128, 4, True, None, False, 4, True),
    (3, 4, 4, 4, 128, 128, 4, True, None, True, 5, True),
    (2, 3, 4, 4, 128, 128, 4, True, None, False, 6, True),
    (2, 8, 2, 2, 128, 128, 4, True, 1.5, False, 7, True),
    (1, 4, 8, 8, 64, 64, 4, True, None, False, 8, True),
    (1, 6, 16, 16, 128, 128, 4, False, None, False, 9, False),
]


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


def _run_reference(inp, B, T, H, HV, K, V, lower_bound):
    dim = 2 * H * K + HV * V
    seq_len = B * T
    conv = inp["conv_pool"].clone()
    ssm = inp["ssm"].clone()
    win = inp["win_pool"].clone()
    ic = inp["inter_ssm"].clone()

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
        intermediate_states_buffer=ic,
        intermediate_state_indices=inp["inter_indices"],
        cache_steps=T,
        retrieve_parent_token=None,
        lower_bound=lower_bound,
    )
    return o, conv, win, ic


def _run_fused(inp, B, T, H, HV, K, V, lower_bound, num_warps):
    conv = inp["conv_pool"].clone()
    ssm = inp["ssm"].clone()
    win = inp["win_pool"].clone()
    ic = inp["inter_ssm"].clone()

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
    )
    return o, conv, win, ic


def _compare_case(case, num_warps):
    B, T, H, HV, K, V, W, has_bias, lower_bound, neg_slot, seed, bitexact = case
    inp = _make_inputs(B, T, H, HV, K, V, W, has_bias, neg_slot, seed)
    o_ref, conv_ref, win_ref, ic_ref = _run_reference(
        inp, B, T, H, HV, K, V, lower_bound
    )
    o_fus, conv_fus, win_fus, ic_fus = _run_fused(
        inp, B, T, H, HV, K, V, lower_bound, num_warps
    )

    idx_vals = inp["idx_vals"]
    valid_rows = [i for i, slot in enumerate(idx_vals) if slot >= 0]

    o_ref_v = o_ref.reshape(B, T, HV, V)[valid_rows]
    o_fus_v = o_fus.reshape(B, T, HV, V)[valid_rows]
    if bitexact:
        assert torch.equal(o_ref_v, o_fus_v)
    else:
        # rtol is 2 bf16 ulp (bf16 carries 8 mantissa bits, so 1 ulp ~ 3.9e-3).
        # The covered case needs 7.3e-3, so this is a real bound, not a rubber
        # stamp -- a third differing ulp fails it.
        torch.testing.assert_close(o_ref_v, o_fus_v, atol=1e-7, rtol=8e-3)
    assert torch.equal(inp["conv_pool"], conv_fus)
    assert torch.equal(win_ref[valid_rows], win_fus[valid_rows])
    torch.testing.assert_close(
        ic_ref[valid_rows], ic_fus[valid_rows], atol=4e-3, rtol=0
    )


@pytest.mark.parametrize("case", _CASES)
def test_matches_unfused_reference(case):
    _compare_case(case, num_warps=4)


def test_verify_accepts_merged_projection_views():
    """QKV and beta retain the merged GEMM's row stride after splitting."""
    B, T, H, HV, K, V = 1, 6, 16, 16, 128, 128
    inp = _make_inputs(B, T, H, HV, K, V, 4, False, False, 11)
    dense = _run_reference(inp, B, T, H, HV, K, V, -5.0)
    merged = torch.cat(
        [inp["mixed"], inp["b"], inp["mixed"].new_zeros(T, 2 * K)], dim=-1
    )
    inp["mixed"], inp["b"], _ = merged.split([3 * H * K, HV, 2 * K], dim=-1)
    unfused = _run_reference(inp, B, T, H, HV, K, V, -5.0)
    fused = _run_fused(inp, B, T, H, HV, K, V, -5.0, 4)
    for actual in (unfused, fused):
        # Output, speculative convolution windows, and SSM checkpoints.
        for i in (0, 2, 3):
            torch.testing.assert_close(actual[i], dense[i], atol=1e-7, rtol=8e-3)


@pytest.mark.parametrize("step", [-1, 0, 2, 5])
def test_verify_commits_only_the_selected_checkpoint(step):
    """Rejected steps must not replace persistent history or untouched slots."""
    from types import SimpleNamespace

    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        scatter_mamba_states_after_mtp_verify,
    )

    B, T, H, HV, K, V = 2, 6, 4, 4, 128, 128
    inp = _make_inputs(B, T, H, HV, K, V, 4, False, False, 10)
    _, _, win_ref, ic_ref = _run_reference(inp, B, T, H, HV, K, V, None)
    _, conv, win, ic = _run_fused(inp, B, T, H, HV, K, V, None, 4)
    ssm = inp["ssm"].clone()
    steps = torch.tensor([step, -1], device=_DEVICE, dtype=torch.int32)
    scatter_mamba_states_after_mtp_verify(
        SimpleNamespace(
            temporal=ssm.unsqueeze(0),
            intermediate_ssm=ic.unsqueeze(0),
            conv=[conv.unsqueeze(0)],
            intermediate_conv_window=[win.unsqueeze(0)],
        ),
        inp["cache_indices"],
        steps,
        None,
        None,
    )
    conv_expected = inp["conv_pool"].clone()
    ssm_expected = inp["ssm"].clone()
    if step >= 0:
        slot = inp["idx_vals"][0]
        conv_expected[slot] = win_ref[0, step]
        ssm_expected[slot] = ic_ref[0, step]
    torch.testing.assert_close(conv, conv_expected, atol=0, rtol=0)
    torch.testing.assert_close(ssm, ssm_expected, atol=2e-6, rtol=1e-5)


def test_verify_history_survives_later_cta_waves():
    """Verify output must not depend on when CTAs sharing Q/K history start."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires CUDA green contexts on SM90 or newer")
    from flashinfer.green_ctx import split_device_green_ctx_by_sm_count

    streams, resources = split_device_green_ctx_by_sm_count(torch.device("cuda:0"), [8])
    with torch.cuda.stream(streams[0]):
        _compare_case(
            (1, 6, 1, 16, 128, 128, 4, False, None, False, 1, False), num_warps=4
        )
    streams[0].synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
