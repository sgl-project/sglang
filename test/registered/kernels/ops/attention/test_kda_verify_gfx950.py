"""HIP KDA chain verify (gfx950, ReplaySSM ring) must match the Triton chain verify.

Conv windows, v ring rows and beta are bit-identical; k ring rows and gate decays sit
within an ulp (the HIP gate's fast exp / log); the output is close."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.linear.fused_kda_conv_recurrent_verify import (
    fused_kda_conv_gating_verify,
)
from sglang.kernels.ops.attention.linear.kda_verify_gfx950 import (
    kda_verify_gfx950,
    supports_kda_verify_gfx950,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_DEVICE = "cuda"
_T, _K, _W, _RING_LEN = 8, 128, 4, 16


def _inputs(B, H, has_bias, seed):
    torch.manual_seed(seed)
    dim = 3 * H * _K
    seq = B * _T
    rnd = lambda *s, dt=torch.bfloat16: torch.randn(*s, device=_DEVICE, dtype=dt)  # noqa: E731
    slots = B + 2
    return dict(
        mixed=rnd(seq, dim) * 0.5,
        w=rnd(dim, _W, dt=torch.float32) * 0.3,
        bias=rnd(dim, dt=torch.float32) * 0.1 if has_bias else None,
        a=rnd(seq, H * _K) * 0.5,
        b=rnd(seq, H),
        A_log=rnd(H, dt=torch.float32) * 0.5,
        dt_bias=rnd(H * _K, dt=torch.float32) * 0.5,
        conv=rnd(slots, _W - 1, dim),
        ssm=rnd(slots, H, _K, _K, dt=torch.float32) * 0.2,
        win=rnd(slots, _T, _W - 1, dim),
        rawv=rnd(slots, H, _RING_LEN, _K),
        rawk=rnd(slots, H, _RING_LEN, _K),
        g=rnd(slots, H, _RING_LEN, _K, dt=torch.float32),
        beta=rnd(slots, H, _RING_LEN, dt=torch.float32),
        # Row 0 is a padded graph slot (-1): neither kernel may touch its state.
        idx=torch.tensor([-1] + list(range(1, B)), device=_DEVICE, dtype=torch.int32)
        if B > 1
        else torch.tensor([1], device=_DEVICE, dtype=torch.int32),
    )


def _run(fn, inp, H, lower_bound):
    return fn(
        mixed_qkv=inp["mixed"],
        conv_weight=inp["w"],
        conv_bias=inp["bias"],
        conv_state=inp["conv"].transpose(-1, -2),
        conv_state_indices=inp["idx"],
        intermediate_conv_window=inp["win"].transpose(-1, -2),
        intermediate_state_indices=inp["idx"],
        a=inp["a"],
        b=inp["b"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        ssm_states=inp["ssm"],
        cache_indices=inp["idx"],
        intermediate_states_buffer=None,
        scale=_K**-0.5,
        T=_T,
        num_q_heads=H,
        num_v_heads=H,
        head_k_dim=_K,
        head_v_dim=_K,
        lower_bound=lower_bound,
        cache_ring=True,
        replayssm_rawv=inp["rawv"],
        replayssm_rawk=inp["rawk"],
        replayssm_g=inp["g"],
        replayssm_beta=inp["beta"],
    )


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
@pytest.mark.parametrize("B", [1, 3, 8, 24])
@pytest.mark.parametrize("H", [12, 16])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("has_bias", [False, True])
def test_matches_triton_chain_verify(B, H, lower_bound, has_bias):
    ref, new = (_inputs(B, H, has_bias, seed=B * 31 + H) for _ in range(2))
    assert supports_kda_verify_gfx950(
        num_q_heads=H,
        num_v_heads=H,
        head_k_dim=_K,
        head_v_dim=_K,
        T=_T,
        mixed_qkv=new["mixed"],
        conv_weight=new["w"],
        a=new["a"],
        b=new["b"],
        ssm_states=new["ssm"],
        replayssm_rawv=new["rawv"],
        intermediate_states_buffer=None,
        intermediate_conv_window=new["win"],
    )
    o_ref = _run(fused_kda_conv_gating_verify, ref, H, lower_bound)
    o_new = _run(kda_verify_gfx950, new, H, lower_bound)
    for name in ("rawv", "beta", "win", "conv", "ssm"):
        assert torch.equal(ref[name], new[name]), name
    torch.testing.assert_close(new["rawk"], ref["rawk"], rtol=8e-3, atol=1e-5)
    torch.testing.assert_close(new["g"], ref["g"], rtol=1e-5, atol=1e-6)
    valid = ref["idx"] >= 0
    rows = valid.repeat_interleave(_T)
    torch.testing.assert_close(
        o_new[0][rows].float(), o_ref[0][rows].float(), rtol=2e-2, atol=2e-3
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
