# SPDX-License-Identifier: Apache-2.0
"""W4A16 GEMM for prefill (M in the hundreds or thousands) on the gfx1151 matrix cores.

Weights are the Quark W4A16 layout SGLang leaves after loading: ``w[n, k // 8]`` int32,
eight codes per word with zero point 8, lane ``j`` at bit ``(j // 2) * 4 + (j % 2) * 16``,
and ``s[n, k // 128]`` scales.

Per 128-wide K block (one quantization group):

- ``(word >> 4i) & 0x000F000F | 0x43004300`` holds two bf16 values ``128 + q`` for lanes
  ``2i`` (low half) and ``2i + 1`` (high half); truncating to 16 bits and bitcasting gives
  them as bf16, exact, with no float arithmetic. That is the whole dequantization.
- A dot product does not care about the order of K, so instead of reshuffling the codes
  into K order the activations are read as the matching eight lane slices
  ``x[:, k8 * 8 + j]``: eight ``tl.dot`` calls of K = 16 each, WMMA's native K.
- The zero point and the group scale come out of the dot:
  ``acc += s[n] * (x . (128 + q) - 136 * sum_k x)``.

From 256 rows this beats ``w4a16_skinny_gemm`` by 1.12-1.34x on every Qwen3.5/3.8
projection on gfx1151.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.utils import use_gfx1151_kernels

GROUP = tl.constexpr(128)


@triton.jit
def _lane_bf16(words, shift: tl.constexpr, HIGH: tl.constexpr):
    """bf16 ``128 + q`` of one lane of every word."""
    pair = ((words >> shift) & 0x000F000F) | 0x43004300
    half = (pair >> 16) if HIGH else pair
    return (half & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _w4a16_prefill_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_wn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr,
    KORDER: tl.constexpr,
):
    BK8: tl.constexpr = GROUP // 8
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = pid // (GROUP_M * num_n)
    first_m = group * GROUP_M
    size_m = tl.minimum(num_m - first_m, GROUP_M)
    pm = first_m + (pid % size_m)
    pn = (pid % (GROUP_M * num_n)) // size_m
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk8 = tl.arange(0, BK8)
    mask_m = rm < M
    mask_n = rn < N
    K8 = K // 8
    G = K // GROUP
    acc = tl.zeros((BM, BN), tl.float32)
    for g in range(0, G):
        k8 = g * BK8 + rk8
        words = tl.load(
            w_ptr + rn[:, None] * stride_wn + k8[None, :],
            mask=mask_n[:, None],
            other=0,
        )
        if KORDER:
            # codes back in K order (k = k8 * 8 + 4 * i1 + 2 * i0 + p): one K = 128 dot
            # against a contiguous activation tile
            v0 = tl.join(_lane_bf16(words, 0, False), _lane_bf16(words, 0, True))
            v1 = tl.join(_lane_bf16(words, 4, False), _lane_bf16(words, 4, True))
            v2 = tl.join(_lane_bf16(words, 8, False), _lane_bf16(words, 8, True))
            v3 = tl.join(_lane_bf16(words, 12, False), _lane_bf16(words, 12, True))
            wk = tl.join(tl.join(v0, v1), tl.join(v2, v3))  # [BN, BK8, p, i0, i1]
            wk = tl.reshape(tl.permute(wk, (0, 1, 4, 3, 2)), (BN, GROUP))
            x = tl.load(
                x_ptr + rm[:, None] * K + (g * GROUP + tl.arange(0, GROUP))[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            d = tl.dot(x, tl.trans(wk))
            xs = tl.sum(x.to(tl.float32), axis=1)
        else:
            xrow = x_ptr + rm[:, None] * K + (k8 * 8)[None, :]
            d = tl.zeros((BM, BN), tl.float32)
            xs = tl.zeros((BM,), tl.float32)
            for i in tl.static_range(4):
                w_lo = _lane_bf16(words, 4 * i, False)  # lane 2i
                w_hi = _lane_bf16(words, 4 * i, True)  # lane 2i + 1
                x_lo = tl.load(xrow + 2 * i, mask=mask_m[:, None], other=0.0)
                x_hi = tl.load(xrow + 2 * i + 1, mask=mask_m[:, None], other=0.0)
                d = tl.dot(x_lo, tl.trans(w_lo), d)
                d = tl.dot(x_hi, tl.trans(w_hi), d)
                xs += tl.sum(x_lo.to(tl.float32), axis=1) + tl.sum(
                    x_hi.to(tl.float32), axis=1
                )
        scale = tl.load(s_ptr + rn * G + g, mask=mask_n, other=0.0).to(tl.float32)
        acc += (d - 136.0 * xs[:, None]) * scale[None, :]
    tl.store(
        out_ptr + rm[:, None] * N + rn[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


MIN_ROWS = 256


def covered(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> bool:
    """Whether ``w4a16_prefill_gemm`` should run: gfx1151, at least ``MIN_ROWS`` contiguous
    bf16/fp16 rows, symmetric group-128 weights in the skinny layout."""
    return (
        x.dim() == 2
        and x.shape[0] >= MIN_ROWS
        and use_gfx1151_kernels()
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.is_contiguous()
        and w.shape[1] * 8 == x.shape[1]
        and w.stride(1) == 1
        and s.shape == (w.shape[0], x.shape[1] // 128)
    )


def w4a16_prefill_gemm(
    x, w, s, bm=64, bn=256, num_warps=8, num_stages=1, group_m=8, korder=True
):
    """``x @ dequant(w).T`` for prefill rows; group size 128, symmetric. The defaults are
    the gfx1151 tuning."""
    m, k = x.shape
    n, k8 = w.shape
    assert k8 * 8 == k and s.shape == (n, k // 128) and x.is_contiguous()
    assert w.stride(1) == 1
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _w4a16_prefill_kernel[grid](
        x,
        w,
        s,
        out,
        m,
        n,
        k,
        w.stride(0),
        BM=bm,
        BN=bn,
        GROUP_M=group_m,
        KORDER=korder,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
