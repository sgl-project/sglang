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


def _mhc_pre_oracle(residual, fn, scale, base, rms_eps, hc_eps, norm_weight, norm_eps):
    """Pure-torch hc_pre + caller-applied RMSNorm.

    Mirrors _mhc_pre_torch: BF16-round layer_input, then the same weight/eps
    the caller uses when norm_fused=False.
    """
    s, n, hidden_size = residual.shape
    x_flat = residual.view(s, n * hidden_size).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.nn.functional.linear(x_flat, fn) * rsqrt

    pre_raw = mixes[:, :n]
    pre = torch.sigmoid(pre_raw * scale[0] + base[:n]) + hc_eps
    layer = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(torch.bfloat16)
    return _rms_norm(layer, norm_weight, norm_eps, torch.bfloat16)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 4096),  # M=1
        (32, 4, 4096),  # production M=32
        (8192, 4, 4096),  # M=8192
    ],
)
def test_mhc_hip_pre_and_post_match_torch_oracles(monkeypatch, shape):
    try:
        import aiter.ops.mhc  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"AITER mHC unavailable: {exc}")

    from contextlib import nullcontext

    # Force the AITER route explicitly; do not depend on CI env defaults.
    monkeypatch.setattr(mhc.envs.SGLANG_USE_AITER, "get", lambda: True)
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(mhc, "get_tp_group", lambda: None)
    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)

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

    ref = _mhc_pre_oracle(
        residual, fn, scale, base, rms_eps, hc_eps, norm_weight, rms_eps
    )

    torch.cuda.synchronize()
    assert torch.isfinite(layer_normed).all(), "layer_input contains NaN/Inf"
    assert torch.isfinite(ref).all(), "oracle contains NaN/Inf"

    diff = layer_normed.float() - ref.float()
    rel_rms = diff.square().mean().sqrt() / ref.float().square().mean().sqrt()
    assert rel_rms < 0.005, f"relative RMS {rel_rms.item():.6f} >= 0.005"

    assert h_res.shape == (s, hc_mult * hc_mult)
    assert h_post.shape == (s, hc_mult)
    assert torch.isfinite(h_res).all()
    assert torch.isfinite(h_post).all()

    x = torch.randn(s, hidden_size, device=device, dtype=torch.bfloat16)
    actual_post = mhc.hc_post(x, residual.view(s, -1), h_post, h_res, hc_mult)
    expected_post = mhc._mhc_post_torch(
        x,
        residual,
        h_post.view(s, hc_mult, 1),
        h_res.view(s, hc_mult, hc_mult),
    ).view(s, -1)
    post_diff = actual_post.float() - expected_post.float()
    post_rel_rms = (
        post_diff.square().mean().sqrt() / expected_post.float().square().mean().sqrt()
    )
    assert post_rel_rms < 0.005, f"post relative RMS {post_rel_rms.item():.6f} >= 0.005"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
