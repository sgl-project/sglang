"""Ascend mHC kernels and the FP32 fallback for lagged pre-mix."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.utils import is_npu_arch22, is_npu_arch35

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v4_mhc import HcSubLayer, HcTriplet


def mix_stats_torch(hc: HcSubLayer, x: torch.Tensor) -> HcTriplet:
    """FP32 counterpart of hc_mix_stats + hc_split_sinkhorn.

    The first row normalization adds eps *after* softmax; subsequent row and
    column normalizations add it to the denominator, matching the fused kernel.
    The loop bound is model-static and introduces no host reads during capture.
    """
    cfg = hc.cfg
    flat = x.flatten(1).float()
    inv_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_eps)
    mixes = F.linear(flat, hc.fn.float()) * inv_rms
    h = cfg.mult
    pre = torch.sigmoid(mixes[:, :h] * hc.scale[0] + hc.base[:h]) + cfg.eps
    post_mix = 2.0 * torch.sigmoid(
        mixes[:, h : 2 * h] * hc.scale[1] + hc.base[h : 2 * h]
    )
    comb = (mixes[:, 2 * h :] * hc.scale[2] + hc.base[2 * h :]).reshape(-1, h, h)
    comb = torch.softmax(comb, dim=-1) + cfg.eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + cfg.eps)
    for _ in range(cfg.sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + cfg.eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + cfg.eps)
    return pre, post_mix, comb


def post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Accumulate residual mixing and the sublayer output in FP32."""
    return (
        torch.bmm(comb.float().transpose(1, 2), residual.float())
        + post_mix.float().unsqueeze(-1) * x.float().unsqueeze(1)
    ).to(x.dtype)


def try_mix_stats(hc: HcSubLayer, x: torch.Tensor) -> HcTriplet | None:
    # On 950 the four-output pre kernel supplies these stats at combine time.
    return None if is_npu_arch35() else mix_stats_torch(hc, x)


def _premix_torch(x, pre):
    return (pre.unsqueeze(-1) * x.float()).sum(dim=1).to(x.dtype)


def try_draft_premix(x, pre):
    # The 910 draft does not use the Triton mhc_combine kernel.
    return _premix_torch(x, pre) if is_npu_arch22() else None


def pre_with_stats(hc: HcSubLayer, x: torch.Tensor, previous_pre):
    """Apply the previous pre-mix, predict this sublayer's coefficients."""
    cfg = hc.cfg
    if x.shape[0] == 0:
        return x[:, 0, :], (
            x.new_empty((0, cfg.mult), dtype=torch.float32),
            x.new_empty((0, cfg.mult), dtype=torch.float32),
            x.new_empty((0, cfg.mult, cfg.mult), dtype=torch.float32),
        )

    stats = try_mix_stats(hc, x)
    if stats is not None:
        mixed = x[:, 0, :] if previous_pre is None else _premix_torch(x, previous_pre)
        return mixed, stats

    from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

    if previous_pre is None:
        # With no preceding sublayer the fused ABI needs the stream-0 selector.
        previous_pre = x.new_zeros((x.shape[0], cfg.mult), dtype=torch.float32)
        previous_pre[:, 0] = 1.0
    mixed, post_mix, comb, pre = mhc_pre_dpskv4(
        x,
        hc.fn,
        hc.scale,
        hc.base,
        cfg.mult,
        cfg.sinkhorn_iters,
        cfg.eps,
        cfg.rms_eps,
        pre_mix=previous_pre,
        return_pre=True,
        impl="cann_recipes_infer_cann920",
    )
    return mixed, (pre, post_mix, comb)


def post(x, residual, post_mix, comb, *, pre_from_prev):
    if not pre_from_prev:
        from sgl_kernel_npu.operators.hyper_conn import mhc_post_dpskv4

        return mhc_post_dpskv4(x, residual, post_mix, comb)
    if is_npu_arch35():
        # The fused 950 ABI uses (hRes, comb, sublayer output, post).
        return torch.ops.cann_ops_transformer.mhc_post(residual, comb, x, post_mix)
    return post_torch(x, residual, post_mix, comb)
