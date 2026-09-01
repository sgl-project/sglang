# SPDX-License-Identifier: MIT
# ATOM-compatible SoA packing for DeepSeek-V4 fp8 unified/compressed KV.
#
# aiter's fp8 decode (mla_decode_fwd_v4_nm, #3112) and fp8 prefill
# (pa_sparse_prefill_fp8_opus, #3751) read a two-buffer SoA layout:
#
#   nope_scale_buf : per (token) 512 bytes fp8 =
#                      [ nope 448 fp8 | e8m0 scale 14 (each of 7 tile-scales x2)
#                        | pad 50 ]
#   rope_buf       : per (token) 64  bf16 (separate tensor, NOT quantized)
#
# sglang already produces `NopeFp8RopeBf16Pack` (k_nope_fp8[.,448] fp8,
# k_rope_bf16[.,64] bf16, scale_k_nope_ue8m0[.,7] uint8 e8m0=exp+127) with the
# identical fp8 numerics (FP8_MAX=448, 1x64 block, pow2 e8m0). The ONLY delta to
# feed aiter is the byte LAYOUT: duplicate each of the 7 scales into 14 inline
# bytes at [448:462] of a 512-wide nope block, and keep rope as its own tensor.
#
# The scale duplication is mandatory: the v4 nm asm shader reads each 64-tile
# scale TWICE consecutively (s0,s0,s1,s1,...,s6,s6); a 32-block prefill reader
# sees consistent pairs -> the SAME single cache feeds BOTH kernels (no double
# storage). This module is layout-only; it performs NO re-quantization.
from __future__ import annotations

import torch

from sglang.kernels.ops.attention.dsv4.index_buf_accessor import (
    NopeFp8RopeBf16Pack,
    fp8_dtype,
)

DIM_NOPE = 448
DIM_ROPE = 64
N_TILE = 7          # 448 / 64
N_SCALE_DUP = 14    # 2 * N_TILE
NOPE_BLOCK = 512    # 448 nope + 14 scale + 50 pad


def _validate(pack: NopeFp8RopeBf16Pack) -> None:
    assert pack.k_nope_fp8.shape[-1] == DIM_NOPE
    assert pack.k_rope_bf16.shape[-1] == DIM_ROPE
    assert pack.scale_k_nope_ue8m0.shape[-1] == N_TILE
    assert pack.k_nope_fp8.dtype == fp8_dtype
    assert pack.k_rope_bf16.dtype == torch.bfloat16
    assert pack.scale_k_nope_ue8m0.dtype == torch.uint8


def pack_to_atom_soa(pack: NopeFp8RopeBf16Pack):
    """Dense layout transform: NopeFp8RopeBf16Pack -> aiter two-buffer SoA.

    Returns
        nope_scale_buf : [R, 512] uint8  (view as fp8_e4m3 for the kernel)
        rope_buf       : [R, 64]  bf16   (separate tensor)
    No re-quantization; only re-arranges the bytes aiter's readers expect.
    """
    _validate(pack)
    nope = pack.k_nope_fp8
    rope = pack.k_rope_bf16
    scale = pack.scale_k_nope_ue8m0  # [R, 7] uint8
    R = nope.shape[0]
    dev = nope.device

    nope_scale_buf = torch.zeros(R, NOPE_BLOCK, dtype=torch.uint8, device=dev)
    # [0:448) nope fp8 bytes
    nope_scale_buf[:, :DIM_NOPE] = nope.view(torch.uint8)
    # [448:462) 14 scale bytes = each tile-scale duplicated (s0,s0,...,s6,s6)
    nope_scale_buf[:, DIM_NOPE : DIM_NOPE + N_SCALE_DUP : 2] = scale
    nope_scale_buf[:, DIM_NOPE + 1 : DIM_NOPE + N_SCALE_DUP : 2] = scale
    # [462:512) pad stays zero (asm reader never reads it)
    rope_buf = rope.contiguous()
    return nope_scale_buf, rope_buf


def atom_soa_to_bf16(nope_scale_buf: torch.Tensor, rope_buf: torch.Tensor):
    """Inverse (reference dequant): read the aiter SoA back to bf16 [R, 512].
    Reads only the first byte of each duplicated scale pair. Mirrors aiter's
    _quant_2buff_to_native."""
    R = nope_scale_buf.shape[0]
    dev = nope_scale_buf.device
    nope_fp8 = nope_scale_buf[:, :DIM_NOPE].view(fp8_dtype)
    scale_e8m0 = nope_scale_buf[:, DIM_NOPE : DIM_NOPE + N_SCALE_DUP : 2].to(torch.int32)
    # e8m0 byte b -> 2^(b-127)
    scale_f32 = torch.exp2((scale_e8m0 - 127).float())  # [R, 7]
    out = torch.empty(R, DIM_NOPE + DIM_ROPE, dtype=torch.bfloat16, device=dev)
    nope_f = nope_fp8.to(torch.float32).view(R, N_TILE, 64)
    deq = (nope_f * scale_f32.unsqueeze(-1)).view(R, DIM_NOPE)
    out[:, :DIM_NOPE] = deq.to(torch.bfloat16)
    out[:, DIM_NOPE:] = rope_buf
    return out


def store_atom_soa(
    nope_scale_buf: torch.Tensor,  # [num_pages, page_size * 512] uint8
    rope_buf: torch.Tensor,        # [num_pages, page_size * 64]  bf16
    loc: torch.Tensor,             # [num_tokens] int
    pack: NopeFp8RopeBf16Pack,
    page_size: int,
) -> None:
    """Paged scatter of a token pack into the aiter SoA buffers (torch ref).
    Slot ``loc`` -> page ``loc // page_size``, offset ``loc % page_size``.
    A Triton port can replace this for production throughput."""
    _validate(pack)
    dense_nope_scale, dense_rope = pack_to_atom_soa(pack)  # [T,512]u8, [T,64]bf16
    page = (loc // page_size).long()
    off = (loc % page_size).long()
    ns_view = nope_scale_buf.view(nope_scale_buf.shape[0], page_size, NOPE_BLOCK)
    rp_view = rope_buf.view(rope_buf.shape[0], page_size, DIM_ROPE)
    ns_view[page, off] = dense_nope_scale
    rp_view[page, off] = dense_rope
