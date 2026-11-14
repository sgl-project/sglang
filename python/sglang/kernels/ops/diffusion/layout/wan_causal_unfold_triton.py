# SPDX-License-Identifier: Apache-2.0
"""Bit-exact causal Conv3d input gather for the ROCm batched-Conv2D VAE path.

``RocmPlatform._conv3d_as_batched_conv2d`` runs a ``Kt x Kh x Kw`` causal
Conv3d as a Conv2D over ``Kt * C_in`` channels. Its NHWC input is built with
``cat([cache_x, x]) -> F.pad -> unfold(T) -> copy_`` into
``[N, T_out, Hp, Wp, Kt, C]`` (cast to the compute dtype): two full fp32
passes plus a strided 6-D gather that aten runs at ~1.4 TB/s on gfx1250.
This kernel writes the same tensor in one pass straight from ``cache_x`` and
``x``. It only moves values, zero-fills the padding and casts once, so the
output is bitwise identical to the aten chain.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Sizes change per resolution and chunk; keep them out of the specialization key.
@triton.jit(do_not_specialize=["T_out", "H", "W", "Hp", "Wp", "cache_t", "pad_t_zero"])
def _causal_unfold_nhwc_kernel(
    x_ptr,
    cache_ptr,
    out_ptr,
    T_out,
    H,
    W,
    Hp,
    Wp,
    cache_t,
    pad_t_zero,
    sxn,
    sxt,
    sxh,
    sxw,
    scn,
    sct,
    sch,
    scw,
    C: tl.constexpr,
    KT: tl.constexpr,
    STRIDE_T: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    oh = row % Hp
    nt = row // Hp
    t = nt % T_out
    n = nt // T_out
    ih = oh - PAD_H
    row_ok = (ih >= 0) & (ih < H)

    ow = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    c = tl.arange(0, BLOCK_C)
    iw = ow - PAD_W
    out_mask = (ow < Wp)[:, None] & (c < C)[None, :]
    in_mask = out_mask & ((iw >= 0) & (iw < W))[:, None]
    out_row = (nt.to(tl.int64) * Hp + oh) * Wp
    out_offs = (out_row + ow.to(tl.int64))[:, None] * (KT * C) + c[None, :]

    for k in tl.static_range(KT):
        it = t * STRIDE_T + k - pad_t_zero
        x_t = it - cache_t
        x_off = (
            (n.to(tl.int64) * sxn + x_t.to(tl.int64) * sxt + ih.to(tl.int64) * sxh)
            + (iw.to(tl.int64) * sxw)[:, None]
            + c[None, :]
        )
        vals = tl.load(x_ptr + x_off, mask=in_mask & row_ok & (x_t >= 0), other=0.0)
        if HAS_CACHE:
            c_off = (
                (n.to(tl.int64) * scn + it.to(tl.int64) * sct + ih.to(tl.int64) * sch)
                + (iw.to(tl.int64) * scw)[:, None]
                + c[None, :]
            )
            from_cache = row_ok & (it >= 0) & (x_t < 0)
            c_vals = tl.load(cache_ptr + c_off, mask=in_mask & from_cache, other=0.0)
            vals = tl.where(from_cache, c_vals, vals)
        tl.store(
            out_ptr + out_offs + k * C,
            vals.to(out_ptr.dtype.element_ty),
            mask=out_mask,
        )


def _is_cl3d_dense_channels(t: torch.Tensor) -> bool:
    return (
        t.dim() == 5
        and t.stride(1) == 1
        and t.is_contiguous(memory_format=torch.channels_last_3d)
    )


def causal_unfold_nhwc(
    x: torch.Tensor,
    cache_x: torch.Tensor | None,
    padding: tuple[int, ...],
    kt: int,
    stride_t: int,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    """``[N, T_out, Hp, Wp, kt, C]`` gather of ``F.pad(cat([cache_x, x], 2), padding)``.

    ``padding`` follows ``WanCausalConv3d._padding``
    ``(w, w, h, h, t_front, 0)``; the front frames come from ``cache_x``
    first and the rest are zero, as in the patched forward. Returns ``None``
    for inputs it does not cover (the caller keeps the aten path).
    """
    pw_l, pw_r, ph_t, ph_b, pt_front, pt_back = padding
    if pw_l != pw_r or ph_t != ph_b or pt_back != 0:
        return None
    if (
        x.device.type != "cuda"
        or not _is_cl3d_dense_channels(x)
        or x.dtype != torch.float32
    ):
        return None
    cache_t = 0
    if cache_x is not None:
        if (
            not _is_cl3d_dense_channels(cache_x)
            or cache_x.dtype != x.dtype
            or cache_x.device != x.device
            or cache_x.shape[:2] != x.shape[:2]
            or cache_x.shape[3:] != x.shape[3:]
        ):
            return None
        cache_t = cache_x.shape[2]
    pad_t_zero = pt_front - cache_t
    if pad_t_zero < 0:
        return None

    N, C, T, H, W = x.shape
    Hp, Wp = H + 2 * ph_t, W + 2 * pw_l
    T_out = (pt_front + T - kt) // stride_t + 1
    if T_out <= 0 or C > 1024:
        return None
    out = torch.empty((N, T_out, Hp, Wp, kt, C), dtype=out_dtype, device=x.device)
    block_c = triton.next_power_of_2(C)
    block_w = max(1, 4096 // block_c)
    cache_arg = x if cache_x is None else cache_x
    scn, _, sct, sch, scw = cache_arg.stride()
    sxn, _, sxt, sxh, sxw = x.stride()
    grid = (N * T_out * Hp, triton.cdiv(Wp, block_w))
    _causal_unfold_nhwc_kernel[grid](
        x,
        cache_arg,
        out,
        T_out,
        H,
        W,
        Hp,
        Wp,
        cache_t,
        pad_t_zero,
        sxn,
        sxt,
        sxh,
        sxw,
        scn,
        sct,
        sch,
        scw,
        C=C,
        KT=kt,
        STRIDE_T=stride_t,
        PAD_H=ph_t,
        PAD_W=pw_l,
        HAS_CACHE=cache_x is not None,
        BLOCK_W=block_w,
        BLOCK_C=block_c,
        num_warps=8,
    )
    return out
