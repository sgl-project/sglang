# SPDX-License-Identifier: Apache-2.0
"""HIP mHC pre + caller-applied RMSNorm numerical contract.

hc_pre either folds the output RMSNorm (norm_fused=True) or returns an
unnormalized BF16 layer_input that matches _mhc_pre_torch after the caller
applies RMSNorm. Compare against an FP32 torch oracle at the production
H4096 hidden size with M=1/32/8192.
"""

import sys

import pytest
import torch

from sglang.srt.utils import is_hip

if not is_hip():
    pytest.skip("mHC HIP regression is ROCm-only.", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("Requires a ROCm GPU.", allow_module_level=True)

import sglang.kernels.ops.layernorm.mhc as mhc  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


def _rms_norm(norm_input, weight, eps, dtype):
    # FP32 oracle matching the production RMSNorm consumer.
    x = norm_input.float()
    rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return (x * rms * weight.float()).to(dtype)


def _relative_rms(actual, expected):
    diff = actual.float() - expected.float()
    return diff.square().mean().sqrt() / expected.float().square().mean().sqrt()


def _mhc_pre_oracle(
    residual,
    fn,
    scale,
    base,
    rms_eps,
    hc_eps,
    post_mult_value,
    sinkhorn_iters,
    norm_weight,
    norm_eps,
):
    """Pure-torch hc_pre + caller-applied RMSNorm.

    Mirrors _mhc_pre_torch: BF16-round layer_input, then the same weight/eps
    the caller uses when norm_fused=False.
    """
    s, n, hidden_size = residual.shape
    x_flat = residual.view(s, n * hidden_size).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.nn.functional.linear(x_flat, fn) * rsqrt

    pre_raw = mixes[:, :n]
    post_raw = mixes[:, n : 2 * n]
    comb_raw = mixes[:, 2 * n :].view(s, n, n)
    pre = torch.sigmoid(pre_raw * scale[0] + base[:n]) + hc_eps
    post = post_mult_value * torch.sigmoid(post_raw * scale[1] + base[n : 2 * n])
    comb = comb_raw * scale[2] + base[2 * n :].view(n, n)
    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    layer = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(torch.bfloat16)
    return (
        _rms_norm(layer, norm_weight, norm_eps, torch.bfloat16),
        comb,
        post.unsqueeze(-1),
    )


def test_mhc_hip_tilelang_fallback_keeps_existing_unfused_kernels(monkeypatch):
    """AITER-off HIP keeps TileLang, but leaves RMSNorm to the caller."""
    monkeypatch.setattr(mhc.envs.SGLANG_USE_AITER, "get", lambda: False)
    monkeypatch.setattr(mhc.envs.SGLANG_OPT_USE_TILELANG_MHC_PRE, "get", lambda: True)
    monkeypatch.setattr(mhc.envs.SGLANG_OPT_USE_TILELANG_MHC_POST, "get", lambda: True)
    monkeypatch.setattr(mhc, "is_gfx95_supported", lambda: True)

    residual = torch.empty(1, 4, 8)
    post_mix = torch.empty(1, 4, 1)
    comb_mix = torch.empty(1, 4, 4)
    layer_input = torch.empty(1, 8)
    norm_weight = torch.empty(8)
    calls = {"pre": 0, "post": 0}

    def native_pre(**kwargs):
        calls["pre"] += 1
        assert kwargs["norm_weight"] is None
        assert kwargs["norm_eps"] is None
        return post_mix, comb_mix, layer_input

    def native_post(*args):
        calls["post"] += 1
        return residual

    def unexpected_torch(*args, **kwargs):
        pytest.fail("enabled TileLang fallback was replaced by Torch")

    monkeypatch.setattr(mhc, "mhc_pre", native_pre)
    monkeypatch.setattr(mhc, "mhc_post", native_post)
    monkeypatch.setattr(mhc, "_mhc_pre_torch", unexpected_torch)
    monkeypatch.setattr(mhc, "_mhc_post_torch", unexpected_torch)

    pre = mhc._mhc_pre_dispatch(
        residual=residual,
        fn=torch.empty(24, 32),
        hc_scale=torch.empty(3),
        hc_base=torch.empty(24),
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=2,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    assert pre[0] is post_mix
    assert pre[1] is comb_mix
    assert pre[2] is layer_input
    assert pre[3] is False

    post = mhc._mhc_post_dispatch(torch.empty(1, 8), residual, post_mix, comb_mix)
    assert post is residual
    assert calls == {"pre": 1, "post": 1}


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 4096),  # M=1
        (32, 4, 4096),  # production M=32
        (8192, 4, 4096),  # M=8192
    ],
)
def test_mhc_hip_pre_and_post_match_torch_oracles(monkeypatch, shape):
    if not mhc.is_gfx95_supported():
        pytest.skip("AITER mHC dispatch requires gfx950.")

    try:
        import aiter.ops.mhc as aiter_mhc
    except ImportError as exc:
        pytest.skip(f"AITER mHC unavailable: {exc}")

    from contextlib import nullcontext

    # Force the AITER route explicitly; do not depend on CI env defaults.
    monkeypatch.setattr(mhc.envs.SGLANG_USE_AITER, "get", lambda: True)
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(mhc, "get_tp_group", lambda: None)
    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)

    calls = {"pre": 0, "post": 0}
    original_pre = aiter_mhc.mhc_pre
    original_post = aiter_mhc.mhc_post

    def tracked_pre(*args, **kwargs):
        calls["pre"] += 1
        return original_pre(*args, **kwargs)

    def tracked_post(*args, **kwargs):
        calls["post"] += 1
        return original_post(*args, **kwargs)

    monkeypatch.setattr(aiter_mhc, "mhc_pre", tracked_pre)
    monkeypatch.setattr(aiter_mhc, "mhc_post", tracked_post)

    torch.manual_seed(0)
    device = torch.device("cuda")
    s, hc_mult, hidden_size = shape
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden = hc_mult * hidden_size

    residual = (
        torch.randn(s, hc_mult, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    )
    fn = torch.randn(hc_mult3, hc_hidden, device=device, dtype=torch.float32) * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    norm_weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16) + 1.0
    rms_eps = 1e-6
    hc_eps = 1e-6
    sinkhorn_iters = 2

    layer_input, h_res, h_post, norm_fused = mhc.hc_pre(
        residual.view(s, -1),
        fn,
        scale,
        base,
        hc_mult,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        out_norm_weight=norm_weight,
        out_norm_eps=rms_eps,
    )

    if norm_fused:
        layer_normed = layer_input
    else:
        layer_normed = _rms_norm(layer_input, norm_weight, rms_eps, torch.bfloat16)

    ref, ref_h_res, ref_h_post = _mhc_pre_oracle(
        residual,
        fn,
        scale,
        base,
        rms_eps,
        hc_eps,
        2.0,
        sinkhorn_iters,
        norm_weight,
        rms_eps,
    )

    torch.cuda.synchronize()
    assert torch.isfinite(layer_normed).all(), "layer_input contains NaN/Inf"
    assert torch.isfinite(ref).all(), "oracle contains NaN/Inf"

    rel_rms = _relative_rms(layer_normed, ref)
    assert rel_rms < 0.005, f"relative RMS {rel_rms.item():.6f} >= 0.005"

    h_res_rel_rms = _relative_rms(h_res, ref_h_res.reshape_as(h_res))
    h_post_rel_rms = _relative_rms(h_post, ref_h_post.reshape_as(h_post))
    assert h_res_rel_rms < 0.005
    assert h_post_rel_rms < 0.005
    assert calls["pre"] == 1

    x = torch.randn(s, hidden_size, device=device, dtype=torch.bfloat16)
    actual_post = mhc.hc_post(x, residual.view(s, -1), h_post, h_res, hc_mult)
    expected_post = mhc._mhc_post_torch(
        x,
        residual,
        ref_h_post,
        ref_h_res,
    ).view(s, -1)
    post_rel_rms = _relative_rms(actual_post, expected_post)
    assert post_rel_rms < 0.005, f"post relative RMS {post_rel_rms.item():.6f} >= 0.005"
    assert calls["post"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
