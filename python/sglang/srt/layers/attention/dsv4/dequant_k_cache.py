"""Dequantize the DSV4-Flash K cache (MXFP8 + page-SOA layout) to BF16.

The DSV4-Flash K cache stores each token as 584 bytes laid out per-page in
SOA form (NOT per-token AOS like DSV3.2's 656-byte FP8 cache)::

    Per page (page_size = 256 tokens, total 149504 bytes):
      [token_0 nope (448B FP8) | token_0 rope (128B BF16)]
      [token_1 nope                  | token_1 rope        ]
      ...
      [token_255 nope                | token_255 rope      ]   <- data region ends at 147456B
      [token_0 scale (7 ue8m0 + 1 pad = 8B)]
      [token_1 scale 8B]
      ...
      [token_255 scale 8B]                                     <- scale region 2048B

Per-token decode:
* nope: 448 FP8 e4m3, scaled by 7 ue8m0 group scales (group_size = 64).
* rope: 64 BF16, copied through unchanged.
* output: 512 BF16 (= 448 dequantized nope ++ 64 rope).

Counterpart writer is `dsv4/index_buf_accessor.py::_set_k_and_s_triton_kernel`
and the per-token quant kernel is `dsv4/quant_k_cache.py::_quant_k_cache_fused_kernel`.

This module is needed for prefill paths that pass a BF16 KV tensor into
`flash_mla.flash_mla_sparse_fwd` (the existing `nsa/dequant_k_cache.py`
assumes the 656-byte DSV3.2 per-token AOS layout and is not compatible).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Layout constants (mirror the writer side in dsv4/index_buf_accessor.py).
DIM_NOPE_FP8: int = 448
DIM_ROPE_BF16: int = 64
HEAD_DIM_BF16: int = DIM_NOPE_FP8 + DIM_ROPE_BF16  # 512
GROUP_SIZE: int = 64
NUM_NOPE_TILES: int = DIM_NOPE_FP8 // GROUP_SIZE  # 7
PADDED_SCALE_BYTES: int = 8  # 7 ue8m0 + 1 byte alignment pad
NOPE_ROPE_BYTES_PER_TOKEN: int = DIM_NOPE_FP8 + DIM_ROPE_BF16 * 2  # 576
HEAD_BYTES: int = NOPE_ROPE_BYTES_PER_TOKEN + PADDED_SCALE_BYTES  # 584


@triton.jit
def _dequant_kernel(
    out_ptr,
    nope_fp8_ptr,
    rope_bf16_ptr,
    scale_u8_ptr,
    out_stride_token,
    nope_stride_token,
    rope_stride_token,
    scale_stride_token,
    NUM_TILES: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    PADDED_NUM_TILES: tl.constexpr,
):
    """One program per token. Loads nope as a 2D ``(PADDED_NUM_TILES,
    GROUP_SIZE)`` block in one shot, loads scales as a 1D vector once,
    broadcasts the scale row across the tile dimension, and writes the
    dequantized nope in one fat store. The padded dimension is masked to
    ``NUM_TILES`` valid lanes (next-power-of-2 padding is required by
    Triton's ``tl.arange``).

    Optimizations vs the original tile-serial version:
    * One ``NUM_TILES × GROUP_SIZE`` nope load + one ``PADDED_NUM_TILES``
      scale load per token (vs ``NUM_TILES`` × small loads under
      ``tl.static_range``).
    * ue8m0 → bf16 scale via ``(s << 7).bitcast<bf16>`` (one shift), vs
      ``tl.exp2`` (transcendental).
    """
    token_id = tl.program_id(0)

    # ----- rope: pure copy (BF16 in, BF16 out) -----
    rope_offs = tl.arange(0, GROUP_SIZE)
    rope_mask = rope_offs < DIM_ROPE
    rope = tl.load(
        rope_bf16_ptr + token_id * rope_stride_token + rope_offs,
        mask=rope_mask,
        other=0.0,
    )
    tl.store(
        out_ptr + token_id * out_stride_token + DIM_NOPE + rope_offs,
        rope,
        mask=rope_mask,
    )

    # ----- nope: 2D fused load + multiply + store -----
    tile_off = tl.arange(0, PADDED_NUM_TILES)  # (P,), P = next_pow2(NUM_TILES)
    elem_off = tl.arange(0, GROUP_SIZE)  # (G,)
    tile_mask = tile_off < NUM_TILES  # (P,)

    # Load the scale row (one load, NUM_TILES valid bytes padded to P).
    scales_u8 = tl.load(
        scale_u8_ptr + token_id * scale_stride_token + tile_off,
        mask=tile_mask,
        other=127,  # 2^0 = 1.0 for padded lanes
    )
    # ue8m0 → bf16: bit pattern (s << 7) is exactly the bf16 encoding of
    # 2^(s-127) (sign=0, exp=s, mantissa=0).
    scales_bf16 = (scales_u8.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)

    # 2D nope load: offs[tile, elem] = tile * GROUP_SIZE + elem.
    nope_offs_2d = tile_off[:, None] * GROUP_SIZE + elem_off[None, :]  # (P, G)
    nope_mask_2d = tile_mask[:, None]  # broadcasts to (P, G)
    nope_fp8 = tl.load(
        nope_fp8_ptr + token_id * nope_stride_token + nope_offs_2d,
        mask=nope_mask_2d,
        other=0.0,
    )

    # Broadcast scale across G; do FP32 multiply for full mantissa.
    nope_fp32 = nope_fp8.to(tl.float32) * scales_bf16[:, None].to(tl.float32)

    # Store back into the contiguous output row.
    tl.store(
        out_ptr + token_id * out_stride_token + nope_offs_2d,
        nope_fp32.to(tl.bfloat16),
        mask=nope_mask_2d,
    )


def dequant_k_cache_dsv4_flash(quant_k_cache: torch.Tensor) -> torch.Tensor:
    """Dequantize a DSV4-Flash K cache view to per-token BF16.

    Args:
        quant_k_cache: ``[num_pages, page_size, 1, 584]`` uint8 view over the
            raw page-SOA byte buffer. ``page_size`` must equal the K-cache
            pool's ``page_size`` (typically 256). The trailing dim is the
            head_bytes (584) for shape bookkeeping only — the actual bytes
            are page-SOA, not AOS, so this dim is **not** a real per-token
            slice and must be re-derived via byte offsets.

    Returns:
        ``[num_pages, page_size, 1, 512]`` bfloat16.
    """
    assert (
        quant_k_cache.is_cuda
    ), f"expected CUDA tensor, got device={quant_k_cache.device}"
    assert (
        quant_k_cache.dtype == torch.uint8
    ), f"expected uint8 raw cache, got {quant_k_cache.dtype}"
    assert quant_k_cache.ndim == 4, f"expected 4D view, got {quant_k_cache.shape}"
    num_pages, page_size, h_kv, head_bytes = quant_k_cache.shape
    assert h_kv == 1, f"expected h_kv=1, got {h_kv}"
    assert head_bytes == HEAD_BYTES, (
        f"expected head_bytes={HEAD_BYTES}, got {head_bytes}. "
        f"DSV4-Flash page-SOA cache must be 584 bytes/token."
    )
    # The caller's view may be a non-contiguous slice of a larger pool
    # buffer (e.g. swa_k_cache[:, : swa_window_size * head_bytes].view(...)).
    # Materialize a contiguous copy so the byte-region splits below are
    # well-defined.
    quant_k_cache = quant_k_cache.contiguous()

    page_bytes = page_size * head_bytes
    data_bytes = page_size * NOPE_ROPE_BYTES_PER_TOKEN
    scale_bytes = page_size * PADDED_SCALE_BYTES
    assert (
        page_bytes == data_bytes + scale_bytes
    ), f"page_bytes mismatch: {page_bytes} vs {data_bytes}+{scale_bytes}"

    # Flatten per-page bytes and split into the SOA data + scale regions.
    # The non-contig slice forces ``.reshape`` to copy into a packed
    # ``(num_pages, page_size, *)`` buffer; the resulting tensors are
    # already contiguous so no explicit ``.contiguous()`` is needed.
    buf_flat = quant_k_cache.view(num_pages, page_bytes)
    data_region = buf_flat[:, :data_bytes].reshape(
        num_pages, page_size, NOPE_ROPE_BYTES_PER_TOKEN
    )
    scale_region = buf_flat[:, data_bytes:].reshape(
        num_pages, page_size, PADDED_SCALE_BYTES
    )

    num_tokens = num_pages * page_size
    nope_fp8 = (
        data_region[:, :, :DIM_NOPE_FP8]
        .reshape(num_tokens, DIM_NOPE_FP8)
        .view(torch.float8_e4m3fn)
    )
    rope_bf16 = (
        data_region[:, :, DIM_NOPE_FP8:]
        .reshape(num_tokens, DIM_ROPE_BF16 * 2)
        .view(torch.bfloat16)
    )
    scale_u8 = scale_region.reshape(num_tokens, PADDED_SCALE_BYTES)

    out = torch.empty(
        (num_tokens, HEAD_DIM_BF16),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    _dequant_kernel[(num_tokens,)](
        out,
        nope_fp8,
        rope_bf16,
        scale_u8,
        out.stride(0),
        nope_fp8.stride(0),
        rope_bf16.stride(0),
        scale_u8.stride(0),
        NUM_TILES=NUM_NOPE_TILES,
        GROUP_SIZE=GROUP_SIZE,
        DIM_NOPE=DIM_NOPE_FP8,
        DIM_ROPE=DIM_ROPE_BF16,
        PADDED_NUM_TILES=triton.next_power_of_2(NUM_NOPE_TILES),
    )

    return out.view(num_pages, page_size, 1, HEAD_DIM_BF16)
