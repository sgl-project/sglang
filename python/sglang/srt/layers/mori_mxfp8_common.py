# SPDX-License-Identifier: Apache-2.0
"""Operand conversion shared by mori's two mxfp8 paths on ROCm gfx950.

`mori_gemm_ar` fuses the GEMM with wo_b's all-reduce; `mori_mxfp8_gemm` runs the
same GEMM on its own where there is no collective to fuse with. They differ in
what happens *after* the multiply and in nothing before it, so the operand
conversion lives here once.

Not duplicated, deliberately: the layouts below are permutations that no shape
check can catch. Two copies that drift produce a finite, plausible, wrong answer
on whichever path was not updated -- which is the failure mode this whole
integration keeps running into.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Rows one packed A-scale group spans, and the rows one quantiser program owns.
#: They are the same 64 by construction, which is what lets the quantiser write
#: mori's layout without any extra traffic -- see `_mxfp8_quant_packed_kernel`.
QUANT_BLOCK_M = 64


def mxfp8_shape(layer) -> tuple[int, int]:
    """The layer's logical ``(N, K)``.

    **Not** ``layer.weight.shape``: `prepare_mxfp8_native_weight` rebinds the
    weight to its shuffled form ``[N/16, K/128, 2048]``, so the first axis is
    N/16. Reading it as N silently disqualifies every layer -- 5120 becomes 320,
    the shape predicate says no because 320 is not a multiple of BLOCK_N, and
    the path falls back *without a word*. The ue8m0 scale keeps the logical
    shape, so take it from there.
    """
    sn, sk = layer.weight_scale_mx_e8m0.shape
    return sn * 32, sk * 32


def mxfp8_ready(layer) -> bool:
    """Whether sglang's native mxfp8 route already prepared this layer.

    `mxfp8_native_ready` means `prepare_mxfp8_native_weight` ran and left the
    shuffled fp8 bytes on ``layer.weight`` and the ``[N/32, K/32]`` exponent
    bytes on ``layer.weight_scale_mx_e8m0``. Handing an unprepared weight to
    mori does not fail -- it returns an uncorrelated result -- so this is
    checked rather than assumed.
    """
    return getattr(layer, "mxfp8_native_ready", False) and hasattr(
        layer, "weight_scale_mx_e8m0"
    )


def mori_weight(layer):
    """The layer's B operand and B scale in mori's layouts, cached on the layer.

    Two conversions, both once per layer:

    * **The weight.** `shuffle_mxfp8_weight` and mori's `preshuffle_b` give each
      lane the same K range and differ only in how the two 64-wide K sub-blocks
      sit -- sglang interleaves them inside a lane's 32 bytes, mori keeps them
      as two 16-byte blocks. The permutation below is exact, verified
      byte-for-byte at [5120, 2048].
    * **The B scale.** mori indexes it K-block major as int32, and the exponent
      bytes are already at the 32x32 granularity it wants (not the per-row
      ``[N, K/32]`` `tl.dot_scaled` takes), so it is a transpose and a widen.

    Cached because the permutation is a full copy of the weight; at 40 layers
    times two GEMMs that is not per-call work.
    """
    cached = getattr(layer, "_mori_b", None)
    if cached is not None:
        return cached
    n, k = mxfp8_shape(layer)
    w = layer.weight.data.contiguous().view(torch.uint8)
    # [N/16, K/128, 64 lanes, 2 sub-blocks, 16B] -> the two sub-blocks split out
    b = (
        w.reshape(n // 16, k // 128, 64, 2, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(n, k)
        .view(layer.weight.dtype)
    )
    b_scale = (
        layer.weight_scale_mx_e8m0.data.contiguous()
        .view(torch.uint8)
        .t()
        .contiguous()
        .to(torch.int32)
        .reshape(-1)
    )
    layer._mori_b = (b, b_scale)
    return layer._mori_b


@triton.jit
def _mxfp8_quant_packed_kernel(
    x_ptr,
    xq_ptr,
    s_ptr,
    M,
    K,
    sxm,
    sxk,
    sqm,
    sqk,
    BLOCK_M: tl.constexpr,
):
    """sglang's ``_mxfp8_quant_kernel`` writing mori's scale layout directly.

    A variant rather than a stride argument on the original, because the layout
    is not expressible as strides: mori wants element ``(m, kb)`` at
    ``kb*M + (m//64)*64 + (m%16)*4 + (m%64)//16``, which permutes *within* each
    64-row group so a lane's four M tiles land in one dword.

    That permutation is free here. The destination stays inside the 64 bytes
    this program already owns, so the store is the same cache line and only its
    order changes -- measured bit-identical to quantising and converting
    afterwards, and slightly faster than the stock kernel, since transposing
    also turns a strided store into a coalesced one.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_b * 32 + tl.arange(0, 32)
    m_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    sb = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
    descale = tl.exp2(sb - 127.0)
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(xq_ptr.dtype.element_ty)
    tl.store(
        xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
        xq,
        mask=m_mask[:, None],
    )
    dst = pid_b * M + (offs_m // 64) * 64 + (offs_m % 16) * 4 + (offs_m % 64) // 16
    tl.store(s_ptr + dst, sb.to(tl.uint8), mask=m_mask)


def quantize_packed(x: torch.Tensor):
    """bf16 ``[M, K]`` -> (fp8 e4m3 values, mori's packed ue8m0 A scale).

    ``M`` must be a multiple of `QUANT_BLOCK_M`; pad the bf16 input first, so
    the scale is built over the padded M and its packed layout needs no repair.
    """
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import MXFP8_VALUE_DTYPE

    m, k = x.shape
    xq = torch.empty((m, k), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scale = torch.empty((k // 32) * m, dtype=torch.uint8, device=x.device)
    _mxfp8_quant_packed_kernel[(triton.cdiv(m, QUANT_BLOCK_M), k // 32)](
        x,
        xq,
        scale,
        m,
        k,
        x.stride(0),
        x.stride(1),
        xq.stride(0),
        xq.stride(1),
        BLOCK_M=QUANT_BLOCK_M,
    )
    return xq, scale.view(torch.int32)
