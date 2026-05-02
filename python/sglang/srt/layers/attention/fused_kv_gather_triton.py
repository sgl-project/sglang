"""Fused KV cache gather + dequantize Triton kernel for SM120 MLA decode.

Single kernel launch replaces ~16 PyTorch ops for the NoPE gather+dequant phase.
Each program handles one (batch, token) pair.
"""

import torch
import triton
import triton.language as tl

# Layout constants
DIM_NOPE = 448
DIM_ROPE = 64
TILE_SIZE = 64
NUM_TILES = DIM_NOPE // TILE_SIZE  # 7
BYTES_NOPE_ROPE = DIM_NOPE + DIM_ROPE * 2  # 576
BYTES_SCALE = NUM_TILES + 1  # 8
HEAD_DIM = DIM_NOPE + DIM_ROPE  # 512


@triton.jit
def _fused_nope_dequant_kernel(
    raw_fp8_ptr,  # float8_e4m3fn view (same memory as raw_buf)
    raw_u8_ptr,  # uint8 view for scale bytes
    indices_ptr,  # (batch, topk) flat token indices
    output_ptr,  # (batch, topk, 448) bf16 output (NoPE only)
    page_size: tl.constexpr,
    kv_dim: tl.constexpr,
    max_topk: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    TILE_SZ: tl.constexpr,
    BYTES_NR: tl.constexpr,
    BYTES_SC: tl.constexpr,
):
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    if tid >= max_topk:
        return

    flat_idx = tl.load(indices_ptr + bid * max_topk + tid)

    page_idx = flat_idx // page_size
    tok_in_page = flat_idx % page_size

    bytes_per_page = page_size * kv_dim
    page_offset = page_idx * bytes_per_page
    nope_base = page_offset + tok_in_page * BYTES_NR
    scale_section_base = page_offset + page_size * BYTES_NR
    scale_base = scale_section_base + tok_in_page * BYTES_SC

    out_base = bid * max_topk * DIM_NOPE + tid * DIM_NOPE

    # Dequantize NoPE tile-by-tile
    for tile in range(NUM_TILES):
        ts = tile * TILE_SZ

        # Load FP8 values (1 byte each, same layout as uint8)
        fp8_offs = nope_base + ts + tl.arange(0, TILE_SZ)
        fp8_raw = tl.load(raw_fp8_ptr + fp8_offs)
        # Triton loads as fp8 element type, .to(float32) does hardware conversion
        fp8_vals = fp8_raw.to(tl.float32)

        # Load UE8M0 scale byte
        sc_off = scale_base + tile
        sc_byte = tl.load(raw_u8_ptr + sc_off)
        scale_val = tl.exp2(sc_byte.to(tl.float32) - 127.0)

        # Dequantize and store
        dequant = (fp8_vals * scale_val).to(tl.bfloat16)
        out_offs = out_base + ts + tl.arange(0, TILE_SZ)
        tl.store(output_ptr + out_offs, dequant)


@triton.jit
def _fused_rope_copy_kernel(
    raw_u8_ptr,
    indices_ptr,
    output_ptr,
    page_size: tl.constexpr,
    kv_dim: tl.constexpr,
    max_topk: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    BYTES_NR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Copy RoPE bf16 bytes from cache to output."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    if tid >= max_topk:
        return

    flat_idx = tl.load(indices_ptr + bid * max_topk + tid)

    page_idx = flat_idx // page_size
    tok_in_page = flat_idx % page_size

    bytes_per_page = page_size * kv_dim
    nope_base = page_idx * bytes_per_page + tok_in_page * BYTES_NR
    rope_base = nope_base + DIM_NOPE

    out_base = bid * max_topk * HEAD_DIM + tid * HEAD_DIM + DIM_NOPE

    # Load RoPE bf16 values (2 bytes each = 64 values from 128 bytes)
    for r in range(DIM_ROPE):
        byte_off = rope_base + r * 2
        lo = tl.load(raw_u8_ptr + byte_off).to(tl.int32)
        hi = tl.load(raw_u8_ptr + byte_off + 1).to(tl.int32)
        bf16_bits = (hi << 8) | lo
        # Reinterpret bits as bf16: cast to uint16 then to bf16
        # In Triton, we use bitwise trick:
        # bf16 bits -> uint16 -> bf16 via pointer cast
        # Since we can't do this directly, store the raw bits and
        # fix up in PyTorch with view
        out_off = out_base + r
        tl.store(output_ptr + out_off, bf16_bits.to(tl.bfloat16))


def fused_gather_dequant(k_cache, indices):
    """Fused KV gather+dequant using Triton + minimal PyTorch.

    Returns (batch, topk, 512) bf16 tensor.
    """
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    kv_dim = k_cache.shape[3]

    batch = indices.shape[0]
    max_topk = indices.shape[-1]
    device = k_cache.device

    # Assume contiguous for CUDA graph compatibility (always true in decode)
    raw_buf_u8 = k_cache.view(torch.uint8).reshape(-1)
    raw_buf_fp8 = k_cache.reshape(-1)  # float8 view (same memory)

    total_tokens = num_pages * page_size
    idx_safe = indices.reshape(batch, -1).clamp(0, total_tokens - 1)

    # Allocate NoPE-only output for Triton kernel (contiguous)
    nope_out = torch.zeros(
        batch, max_topk, DIM_NOPE, dtype=torch.bfloat16, device=device
    )

    # Phase 1: Triton fused NoPE dequant (replaces ~10 PyTorch ops with 1 kernel)
    grid = (batch, max_topk)
    _fused_nope_dequant_kernel[grid](
        raw_buf_fp8,
        raw_buf_u8,
        idx_safe,
        nope_out,
        page_size=page_size,
        kv_dim=kv_dim,
        max_topk=max_topk,
        DIM_NOPE=DIM_NOPE,
        NUM_TILES=NUM_TILES,
        TILE_SZ=TILE_SIZE,
        BYTES_NR=BYTES_NOPE_ROPE,
        BYTES_SC=BYTES_SCALE,
    )

    # Phase 2: RoPE gather via PyTorch (small: 128 bytes per token, 1 gather)
    bytes_per_page = page_size * kv_dim
    page_idx = idx_safe // page_size
    tok_in_page = idx_safe % page_size
    page_offsets = page_idx * bytes_per_page
    nope_starts = page_offsets + tok_in_page * BYTES_NOPE_ROPE

    rope_byte_dim = DIM_ROPE * 2
    rope_offs = (nope_starts + DIM_NOPE).unsqueeze(-1) + torch.arange(
        rope_byte_dim, device=device
    )
    rope_offs_c = rope_offs.clamp(0, raw_buf_u8.shape[0] - 1)
    rope_bytes = raw_buf_u8[rope_offs_c.reshape(-1)].reshape(
        batch, max_topk, rope_byte_dim
    )
    rope_bf16 = rope_bytes.view(torch.bfloat16)

    # Concatenate
    result = torch.cat([nope_out, rope_bf16], dim=-1)

    # Zero invalid entries using torch.where (CUDA graph compatible — no .all() sync)
    valid_mask = (indices.reshape(batch, -1) >= 0) & (
        indices.reshape(batch, -1) < total_tokens
    )
    result = torch.where(valid_mask.unsqueeze(-1), result, torch.zeros_like(result))

    return result
