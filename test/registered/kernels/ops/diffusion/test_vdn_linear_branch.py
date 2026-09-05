"""VDN-H3 linear-branch kernels against the eager chains they replace.

``vdn_frame_stats_prep`` and ``vdn_gather_linear_state`` move values and form
the same products, so ``torch.equal`` / fp32 tolerance; the three activation
kernels round once at the store and are held to one bf16 ulp of the eager
chain (their documented contract). The whole-branch wiring (fused vs eager
chain through the real forward) lives with the model tests.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_vdn_frame_stats_prep,
    can_use_vdn_gather_linear_state,
    can_use_vdn_linear_epilogue,
    can_use_vdn_silu_l2norm,
    can_use_vdn_temporal_conv_act,
    vdn_frame_stats_prep,
    vdn_linear_epilogue,
    vdn_silu_l2norm,
    vdn_temporal_conv_act,
)
from sglang.multimodal_gen.configs.models.dits.minimax_h3_vdn import (
    VDNHybridAttentionArchConfig,
)
from sglang.multimodal_gen.runtime.models.dits import minimax_h3_vdn as vdn
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

F_, S_, H_, D_ = 6, 24, 3, 32


def _ulp_close(got: torch.Tensor, ref: torch.Tensor) -> bool:
    scale = max(1.0, ref.float().abs().max().item())
    return (got.float() - ref.float()).abs().max().item() <= 2e-2 * scale


def test_temporal_conv_act_matches_eager_chain() -> None:
    g = torch.Generator(device="cpu").manual_seed(4)
    x = torch.randn(F_, S_, H_ * D_, generator=g).to("cuda", torch.bfloat16)
    w = (torch.randn(H_ * D_, 5, generator=g) * 0.4).to("cuda", torch.bfloat16)
    assert can_use_vdn_temporal_conv_act(x, H_, D_)
    ref = vdn._activate(vdn._temporal_shift(x, w).reshape(-1, H_, D_), True)
    assert _ulp_close(vdn_temporal_conv_act(x, w, H_, D_, True), ref)
    frame_major = vdn_temporal_conv_act(x, w, H_, D_, True, frame_major=True)
    assert frame_major.shape == (F_, H_, S_, D_) and frame_major.is_contiguous()
    assert torch.equal(
        frame_major, ref.view(F_, S_, H_, D_).permute(0, 2, 1, 3)
    ) or _ulp_close(frame_major, ref.view(F_, S_, H_, D_).permute(0, 2, 1, 3))


def test_silu_l2norm_reads_strided_qkv_views() -> None:
    g = torch.Generator(device="cpu").manual_seed(4)
    tokens = torch.randn(F_ * S_, 3 * H_ * D_, generator=g).to("cuda", torch.bfloat16)
    strided = tokens[:, : H_ * D_].view(F_ * S_, H_, D_)
    assert can_use_vdn_silu_l2norm(strided)
    got = vdn_silu_l2norm(strided, True)
    assert got.is_contiguous() and _ulp_close(got, vdn._activate(strided, True))
    got_v = vdn_silu_l2norm(strided, False)
    assert _ulp_close(got_v, torch.nn.functional.silu(strided))
    frame_major = vdn_silu_l2norm(strided, True, per_frame=S_)
    assert frame_major.shape == (F_, H_, S_, D_)
    assert torch.equal(got.view(F_, S_, H_, D_).permute(0, 2, 1, 3), frame_major)
    with pytest.raises(ValueError):
        vdn_silu_l2norm(strided, True, per_frame=S_ + 1)


def test_frame_stats_prep_is_bit_exact() -> None:
    g = torch.Generator(device="cpu").manual_seed(4)
    key = torch.randn(F_ * S_, H_, D_, generator=g).to("cuda", torch.bfloat16)
    value = torch.randn(F_ * S_, H_, D_, generator=g).to("cuda", torch.bfloat16)
    beta = torch.rand(F_ * S_, H_, generator=g).to("cuda", torch.bfloat16)
    assert can_use_vdn_frame_stats_prep(key, value)
    k16, k32, kb32, vb = vdn_frame_stats_prep(key, value, beta, F_, S_)
    kf = key.view(F_, S_, H_, D_).permute(0, 2, 1, 3)
    vf = value.view(F_, S_, H_, D_).permute(0, 2, 1, 3)
    bf = beta.view(F_, S_, H_).permute(0, 2, 1)
    assert torch.equal(k16, kf.contiguous())
    assert torch.equal(k32, kf.float().contiguous())
    assert torch.equal(kb32, (kf.float() * bf.unsqueeze(-1).float()).contiguous())
    assert torch.equal(vb, (vf * bf.unsqueeze(-1).to(vf.dtype)).contiguous())


def test_linear_epilogue_matches_eager_chain() -> None:
    g = torch.Generator(device="cpu").manual_seed(4)
    readout = torch.randn(F_, H_, S_, D_, generator=g).to("cuda", torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(D_, generator=g)).to("cuda", torch.bfloat16)
    gate = torch.rand(F_ * S_, H_, D_, generator=g).to("cuda", torch.bfloat16)
    assert can_use_vdn_linear_epilogue(readout)
    got = vdn_linear_epilogue(readout, weight, gate, 1e-6)
    assert _ulp_close(got, vdn.linear_epilogue(readout, weight, gate, 1e-6))


@pytest.mark.parametrize("bridge", ["alpha", "none"])
@pytest.mark.parametrize("with_text_state", [False, True])
def test_gather_linear_state_matches_eager(bridge: str, with_text_state: bool) -> None:
    g = torch.Generator(device="cpu").manual_seed(5)
    frames, heads, dim = 9, 2, 32
    hybrid = VDNHybridAttentionArchConfig(chunk=3, radius=1, anchor_frames="none")
    bounds = hybrid.window_bounds(frames)
    prefix = torch.randn(frames, heads, dim, dim, generator=g).cuda()
    suffix = torch.randn(frames, heads, dim, dim, generator=g).cuda()
    alpha = (torch.rand(frames, heads, dim, generator=g) * 0.5 + 0.5).cuda()
    text = torch.randn(heads, dim, dim, generator=g).cuda() if with_text_state else None
    assert can_use_vdn_gather_linear_state(prefix)
    kwargs = dict(bridge=bridge, text_state=text, out_dtype=torch.float32)
    ref = vdn.gather_linear_state(prefix, suffix, alpha, bounds, fused=False, **kwargs)
    got = vdn.gather_linear_state(prefix, suffix, alpha, bounds, **kwargs)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_predicates_reject_unsupported_inputs() -> None:
    fp16 = torch.randn(4, 2, 32, device="cuda", dtype=torch.float16)
    assert not can_use_vdn_silu_l2norm(fp16)
    odd = torch.randn(4, 2, 48, device="cuda", dtype=torch.bfloat16)
    assert not can_use_vdn_silu_l2norm(odd)
    with pytest.raises(ValueError):
        vdn_silu_l2norm(odd, True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
