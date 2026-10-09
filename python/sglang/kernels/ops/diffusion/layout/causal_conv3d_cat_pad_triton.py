from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _fused_cat_pad_5d_kernel(
    x_ptr,
    cache_ptr,
    out_ptr,
    total,
    channels,
    t_size,
    h_size,
    w_size,
    cache_t,
    out_t,
    out_h,
    out_w,
    pad_d_left,
    pad_h_top,
    pad_w_left,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total

    ow = offsets % out_w
    tmp = offsets // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    out = tmp % out_t
    tmp = tmp // out_t
    oc = tmp % channels
    ob = tmp // channels

    iw = ow - pad_w_left
    ih = oh - pad_h_top
    src_t = out - pad_d_left

    valid = (
        mask
        & (iw >= 0)
        & (iw < w_size)
        & (ih >= 0)
        & (ih < h_size)
        & (src_t >= 0)
        & (src_t < cache_t + t_size)
    )
    from_cache = src_t < cache_t

    x_t = src_t - cache_t
    clamped_iw = tl.minimum(tl.maximum(iw, 0), w_size - 1)
    clamped_ih = tl.minimum(tl.maximum(ih, 0), h_size - 1)
    clamped_x_t = tl.minimum(tl.maximum(x_t, 0), t_size - 1)
    clamped_src_t = tl.minimum(tl.maximum(src_t, 0), cache_t - 1)

    x_offsets = (
        ((ob * channels + oc) * t_size + clamped_x_t) * h_size + clamped_ih
    ) * w_size + clamped_iw
    cache_offsets = (
        ((ob * channels + oc) * cache_t + clamped_src_t) * h_size + clamped_ih
    ) * w_size + clamped_iw

    x_vals = tl.load(x_ptr + x_offsets, mask=valid & ~from_cache, other=0.0)
    cache_vals = tl.load(cache_ptr + cache_offsets, mask=valid & from_cache, other=0.0)
    vals = tl.where(from_cache, cache_vals, x_vals)
    tl.store(out_ptr + offsets, vals, mask=mask)


def fused_causal_conv3d_cat_pad(
    x: torch.Tensor,
    cache_x: torch.Tensor,
    padding: list[int] | tuple[int, ...],
) -> torch.Tensor:
    width_left, width_right, height_top, height_bottom, depth_left, depth_right = (
        padding
    )
    depth_left -= cache_x.shape[2]
    assert depth_left >= 0
    assert depth_right == 0
    assert width_left == width_right
    assert height_top == height_bottom

    bsz, channels, t_size, h_size, w_size = x.shape
    cache_t = cache_x.shape[2]
    out = torch.empty(
        (
            bsz,
            channels,
            t_size + cache_t + depth_left + depth_right,
            h_size + height_top + height_bottom,
            w_size + width_left + width_right,
        ),
        device=x.device,
        dtype=x.dtype,
    )
    block_size = 256
    total = out.numel()
    grid = (triton.cdiv(total, block_size),)
    with torch.get_device_module().device(x.device):
        _fused_cat_pad_5d_kernel[grid](
            x,
            cache_x,
            out,
            total,
            channels,
            t_size,
            h_size,
            w_size,
            cache_t,
            out.shape[2],
            out.shape[3],
            out.shape[4],
            depth_left,
            height_top,
            width_left,
            block_size,
        )
    return out
