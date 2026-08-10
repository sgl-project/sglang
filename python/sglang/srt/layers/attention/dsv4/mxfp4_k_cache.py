"""MXFP4 codec for DeepSeek V4 attention KV entries.

Each token occupies one 368-byte row::

    [224 B packed E2M1 | 14 B E8M0 + 2 B pad | 128 B BF16 RoPE]

Only the 448-dimensional nope vector is quantized (block-32, E8M0 scale).
The 64-dimensional RoPE vector stays in bfloat16.  E8M0 covers the full
FP range (2⁻¹²⁷ to 2¹²⁷) without a separate global scale, so the API is
simpler than the NVFP4 codec.

Decode consumes the layout directly via the fused JIT CUDA kernel.
Sparse prefill dequantizes selected entries to BF16 on the fly.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MXFP4_GROUP_SIZE = 32
MXFP4_NOPE_DIM = 448
MXFP4_ROPE_DIM = 64
MXFP4_TOTAL_DIM = MXFP4_NOPE_DIM + MXFP4_ROPE_DIM  # 512
MXFP4_NUM_GROUPS = MXFP4_NOPE_DIM // MXFP4_GROUP_SIZE  # 14
MXFP4_PACKED_NOPE_BYTES = MXFP4_NOPE_DIM // 2  # 224
MXFP4_SCALE_BYTES = MXFP4_NUM_GROUPS + 2  # 16 (14 + 2 pad)
MXFP4_ROPE_BYTES = MXFP4_ROPE_DIM * 2  # 128
MXFP4_BYTES_PER_TOKEN = (
    MXFP4_PACKED_NOPE_BYTES + MXFP4_SCALE_BYTES + MXFP4_ROPE_BYTES
)  # 368

_E2M1_MAX = 6.0

# Triton requires tl.arange ranges to be powers of two.  We always use
# power-of-two tile sizes and mask to the actual group count (14).
# ----------------------------------------------------------------
_MAX_GROUPS = 16  # next power of two >= 14
_NOPE_TILE = 512  # next power of two >= 448, used for output masking

# Launch config: one CTA owns a complete token row for small batches;
# split across 2 CTAs for large prefill workloads.
_QUANTIZE_SMALL_BATCH_THRESHOLD = 64
_QUANTIZE_SMALL_GROUPS_PER_PROGRAM = 16  # one CTA per token (16 ≥ 14)
_QUANTIZE_SMALL_NUM_WARPS = 4
_QUANTIZE_LARGE_GROUPS_PER_PROGRAM = 8  # two CTAs per token
_QUANTIZE_LARGE_NUM_WARPS = 2


def _quantize_launch_config(num_tokens: int) -> tuple[int, int]:
    if num_tokens <= _QUANTIZE_SMALL_BATCH_THRESHOLD:
        return _QUANTIZE_SMALL_GROUPS_PER_PROGRAM, _QUANTIZE_SMALL_NUM_WARPS
    return _QUANTIZE_LARGE_GROUPS_PER_PROGRAM, _QUANTIZE_LARGE_NUM_WARPS


# ---------------------------------------------------------------------------
# Validation helpers (following PR #31269 NVFP4 pattern)
# ---------------------------------------------------------------------------


def _as_feature_matrix(x: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError(
            f"{name} must have shape [num_tokens, {dim}] or "
            f"[num_tokens, 1, {dim}], got {tuple(x.shape)}"
        )
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"{name} must use bfloat16, float16, or float32, got {x.dtype}")
    if x.stride(1) != 1:
        x = x.contiguous()
    return x


def _as_rows(kv_buffer: torch.Tensor, page_size: int) -> torch.Tensor:
    if kv_buffer.dtype != torch.uint8:
        raise TypeError(f"MXFP4 buffer must be uint8, got {kv_buffer.dtype}")
    if kv_buffer.ndim != 2:
        raise ValueError(
            f"MXFP4 buffer must be [num_pages, bytes_per_page], "
            f"got {tuple(kv_buffer.shape)}"
        )
    expected = page_size * MXFP4_BYTES_PER_TOKEN
    if kv_buffer.shape[1] != expected:
        raise ValueError(
            f"MXFP4 page must contain {expected} bytes, " f"got {kv_buffer.shape[1]}"
        )
    if not kv_buffer.is_contiguous():
        raise ValueError("MXFP4 buffer must be contiguous")
    return kv_buffer.view(-1, MXFP4_BYTES_PER_TOKEN)


def _validate_loc(loc: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    loc = loc.reshape(-1)
    if loc.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"loc must be int32 or int64, got {loc.dtype}")
    if loc.device != rows.device:
        raise ValueError("loc and KV buffer must be on one device")
    return loc.contiguous()


def _split_k(cache_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_k.ndim == 3 and cache_k.shape[1] == 1:
        cache_k = cache_k[:, 0, :]
    if cache_k.ndim != 2 or cache_k.shape[1] != MXFP4_TOTAL_DIM:
        raise ValueError(
            "cache_k must be [num_tokens, 512] or [num_tokens, 1, 512], "
            f"got {tuple(cache_k.shape)}"
        )
    return (
        _as_feature_matrix(cache_k[:, :MXFP4_NOPE_DIM], MXFP4_NOPE_DIM, "k_nope"),
        _as_feature_matrix(cache_k[:, MXFP4_NOPE_DIM:], MXFP4_ROPE_DIM, "k_rope"),
    )


# ---------------------------------------------------------------------------
# E2M1 helpers
# ---------------------------------------------------------------------------

_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _e2m1_rne_scaled_torch(x: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """E2M1 RNE encoding using scaled-midpoint comparisons (no division).

    Compares original magnitudes against midpoints derived from the stored
    scale to avoid reciprocal-induced FP32 ulp drift.
    """
    magnitude = x.abs()
    code = (
        (magnitude > denominator * 0.25).to(torch.uint8)
        + (magnitude >= denominator * 0.75).to(torch.uint8)
        + (magnitude > denominator * 1.25).to(torch.uint8)
        + (magnitude >= denominator * 1.75).to(torch.uint8)
        + (magnitude > denominator * 2.5).to(torch.uint8)
        + (magnitude >= denominator * 3.5).to(torch.uint8)
        + (magnitude > denominator * 5.0).to(torch.uint8)
    )
    signed_code = code | (torch.signbit(x).to(torch.uint8) << 3)
    return torch.where(
        denominator > 0,
        signed_code,
        torch.zeros((), dtype=torch.uint8, device=x.device),
    )


def _decode_e2m1_torch(code: torch.Tensor) -> torch.Tensor:
    return _E2M1_LUT.to(code.device)[code.long()]


# ---------------------------------------------------------------------------
# Triton helper: E2M1 RNE (scaled-midpoint, no division)
# ---------------------------------------------------------------------------


@triton.jit
def _e2m1_rne_scaled_triton(x, denominator):
    magnitude = tl.abs(x)
    code = (
        (magnitude > denominator * 0.25).to(tl.uint8)
        + (magnitude >= denominator * 0.75).to(tl.uint8)
        + (magnitude > denominator * 1.25).to(tl.uint8)
        + (magnitude >= denominator * 1.75).to(tl.uint8)
        + (magnitude > denominator * 2.5).to(tl.uint8)
        + (magnitude >= denominator * 3.5).to(tl.uint8)
        + (magnitude > denominator * 5.0).to(tl.uint8)
    )
    sign = ((x.to(tl.uint32, bitcast=True) >> 31).to(tl.uint8)) << 3
    return tl.where(denominator > 0.0, code | sign, 0).to(tl.uint8)


@triton.jit
def _decode_e2m1_triton(code):
    magnitude_code = code & 0x07
    magnitude = tl.where(
        magnitude_code == 0,
        0.0,
        tl.where(
            magnitude_code == 1,
            0.5,
            tl.where(
                magnitude_code == 2,
                1.0,
                tl.where(
                    magnitude_code == 3,
                    1.5,
                    tl.where(
                        magnitude_code == 4,
                        2.0,
                        tl.where(
                            magnitude_code == 5,
                            3.0,
                            tl.where(magnitude_code == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where((code & 0x08) != 0, -magnitude, magnitude)


# ---------------------------------------------------------------------------
# Triton quantize kernel
# ---------------------------------------------------------------------------


@triton.jit
def _quantize_mxfp4_k_cache_into_kernel(
    k_nope_ptr,
    k_rope_ptr,
    packed_out_ptr,
    scale_out_ptr,
    rope_out_ptr,
    loc_ptr,
    num_rows,
    k_nope_stride_0: tl.constexpr,
    k_rope_stride_0: tl.constexpr,
    packed_out_stride_0: tl.constexpr,
    scale_out_stride_0: tl.constexpr,
    rope_out_stride_0: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    token_id = tl.program_id(0)
    part_id = tl.program_id(1)
    dst_row = tl.load(loc_ptr + token_id).to(tl.int64)
    valid_dst = (dst_row >= 0) & (dst_row < num_rows)
    safe_row = tl.where(valid_dst, dst_row, 0)

    group_start = part_id * GROUPS_PER_PROGRAM
    # tl.arange demands a power-of-two range; GROUPS_PER_PROGRAM ∈ {8, 16}.
    group_ids = group_start + tl.arange(0, GROUPS_PER_PROGRAM)  # [GROUPS_PER_PROGRAM]
    valid_group = group_ids < NUM_GROUPS
    elem_ids = tl.arange(0, GROUP_SIZE)  # [32]  ← power of two

    # Load [GROUPS_PER_PROGRAM, 32] elements of k_nope
    input_offsets = group_ids[:, None] * GROUP_SIZE + elem_ids[None, :]
    x = tl.load(
        k_nope_ptr + token_id * k_nope_stride_0 + input_offsets,
        mask=valid_dst & valid_group[:, None],
        other=0.0,
    ).to(tl.float32)

    # E8M0 scale: byte = round(log2(amax / 6)) + 127
    amax = tl.max(tl.abs(x), axis=1)
    log2_ratio = tl.math.log2(tl.maximum(amax, 1e-40)) - 2.584962500721156  # log₂6
    # tl.math.round/rint are unavailable in Triton 3.6; floor(x+0.5) gives
    # round-to-nearest (ties go up, negligible for E8M0).
    scale_byte_raw = tl.math.floor(log2_ratio + 0.5).to(tl.int32) + 127
    # tl.clamp(·, 0, 255) on int32 is unsupported; use min/max.
    scale_byte = tl.minimum(tl.maximum(scale_byte_raw, 0), 255).to(tl.uint8)

    # E2M1 RNE: compare original values against scaled midpoints (no division)
    scale_float = tl.math.exp2((scale_byte.to(tl.float32) - 127.0))
    denominator = tl.expand_dims(scale_float, 1)
    codes = _e2m1_rne_scaled_triton(x, denominator)  # [GROUPS_PER_PROGRAM, 32] uint8

    # Pack nibbles: two adjacent E2M1 codes → one byte
    codes_2d = tl.reshape(codes, (GROUPS_PER_PROGRAM, GROUP_SIZE // 2, 2))
    low, high = tl.split(codes_2d)  # each [GROUPS_PER_PROGRAM, 16]
    packed = low | (high << 4)

    # Store packed nope: [GROUPS_PER_PROGRAM, 16] bytes per token
    byte_ids = tl.arange(0, GROUP_SIZE // 2)  # [16]
    packed_offsets = group_ids[:, None] * (GROUP_SIZE // 2) + byte_ids[None, :]
    tl.store(
        packed_out_ptr + safe_row * packed_out_stride_0 + packed_offsets,
        packed,
        mask=valid_dst & valid_group[:, None],
    )
    tl.store(
        scale_out_ptr + safe_row * scale_out_stride_0 + group_ids,
        scale_byte,
        mask=valid_dst & valid_group,
    )

    # RoPE: written once by the last part
    num_parts = tl.cdiv(NUM_GROUPS, GROUPS_PER_PROGRAM)
    if part_id == num_parts - 1:
        rope_offsets = tl.arange(0, ROPE_DIM)  # [64]  ← power of two
        rope = tl.load(
            k_rope_ptr + token_id * k_rope_stride_0 + rope_offsets,
            mask=valid_dst,
            other=0.0,
        )
        tl.store(
            rope_out_ptr + safe_row * rope_out_stride_0 + rope_offsets,
            rope,
            mask=valid_dst,
        )


# ---------------------------------------------------------------------------
# Triton dequant kernel
# ---------------------------------------------------------------------------


@triton.jit
def _dequantize_mxfp4_k_cache_paged_kernel(
    packed_ptr,
    scale_ptr,
    rope_ptr,
    token_indices_ptr,
    output_ptr,
    num_rows,
    packed_stride_0: tl.constexpr,
    scale_stride_0: tl.constexpr,
    rope_stride_0: tl.constexpr,
    output_stride_0: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_TILE: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    token_id = tl.program_id(0)
    src_row = tl.load(token_indices_ptr + token_id).to(tl.int64)
    valid_src = (src_row >= 0) & (src_row < num_rows)

    # Load packed nope as [MAX_GROUPS, GROUP_SIZE//2] bytes.
    # MAX_GROUPS = 16 ≥ the real 14 groups; the extra 2 groups land in the
    # scale / rope region and are masked away after decode.
    group_ids = tl.arange(0, MAX_GROUPS)[:, None]  # [16, 1]  ← power of two
    byte_ids = tl.arange(0, GROUP_SIZE // 2)[None, :]  # [1, 16]  ← power of two
    packed_offsets = group_ids * (GROUP_SIZE // 2) + byte_ids  # [16, 16]
    packed = tl.load(
        packed_ptr + src_row * packed_stride_0 + packed_offsets,
        mask=valid_src,
        other=0,
    ).to(tl.uint8)

    # Load scales: [MAX_GROUPS] with mask, last 2 are padding
    scale_offsets = tl.arange(0, MAX_GROUPS)  # [16]  ← power of two
    scale_byte = tl.load(
        scale_ptr + src_row * scale_stride_0 + scale_offsets,
        mask=valid_src & (scale_offsets < NOPE_DIM // GROUP_SIZE),
        other=127,  # scale = 2^(127-127) = 1.0 (harmless for masked groups)
    ).to(tl.uint8)
    scale = tl.math.exp2((scale_byte.to(tl.float32) - 127.0))

    # Unpack nibbles → decode E2M1 → apply scale
    low = _decode_e2m1_triton(packed & 0x0F) * scale[:, None]  # [16, 16]
    high = _decode_e2m1_triton(packed >> 4) * scale[:, None]  # [16, 16]
    nope_all = tl.reshape(tl.join(low, high), (NOPE_TILE,))  # [512]

    # Store nope: NOPE_TILE=512 ≥ NOPE_DIM=448; mask trims to the real dim
    nope_offsets = tl.arange(0, NOPE_TILE)  # [512]  ← power of two
    tl.store(
        output_ptr + token_id * output_stride_0 + nope_offsets,
        nope_all,
        mask=valid_src & (nope_offsets < NOPE_DIM),
    )

    # RoPE: ROPE_DIM=64 is a power of two, no masking needed
    rope_offsets = tl.arange(0, ROPE_DIM)
    rope = tl.load(
        rope_ptr + src_row * rope_stride_0 + rope_offsets,
        mask=valid_src,
        other=0.0,
    )
    tl.store(
        output_ptr + token_id * output_stride_0 + NOPE_DIM + rope_offsets,
        rope,
        mask=valid_src,
    )


# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------


def _quantize_mxfp4_reference(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    packed_rows: torch.Tensor,
    scale_rows: torch.Tensor,
    rope_rows: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    """CPU reference: quantize + scatter into destination views."""
    blocks = k_nope.float().reshape(-1, MXFP4_NUM_GROUPS, MXFP4_GROUP_SIZE)
    amax = blocks.abs().amax(dim=-1)  # [N, 14]
    log2_ratio = torch.log2(amax.clamp(min=1e-40)) - 2.584962500721156
    scale_byte = (log2_ratio.round().long() + 127).clamp(0, 255).to(torch.uint8)
    scale_float = torch.pow(2.0, (scale_byte.float() - 127.0))
    denominator = scale_float.unsqueeze(-1)
    codes = _e2m1_rne_scaled_torch(blocks, denominator).reshape(-1, MXFP4_NOPE_DIM)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)

    valid = (loc >= 0) & (loc < packed_rows.shape[0])
    if bool(valid.any()):
        dst = loc[valid].long()
        packed_rows[dst] = packed[valid]
        scale_rows[dst, :MXFP4_NUM_GROUPS] = scale_byte[valid]
        rope_rows[dst] = k_rope[valid].to(torch.bfloat16)


def _dequantize_mxfp4_reference(
    packed_rows: torch.Tensor,
    scale_rows: torch.Tensor,
    rope_rows: torch.Tensor,
    token_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """CPU reference: gather + dequantize."""
    out.zero_()
    valid = (token_indices >= 0) & (token_indices < packed_rows.shape[0])
    if not bool(valid.any()):
        return

    selected_packed = packed_rows[token_indices[valid].long()]  # [V, 224]
    codes = torch.zeros(
        selected_packed.shape[0],
        MXFP4_NOPE_DIM,
        dtype=torch.uint8,
        device=selected_packed.device,
    )
    codes[:, 0::2] = selected_packed & 0x0F
    codes[:, 1::2] = selected_packed >> 4

    selected_scales = scale_rows[
        token_indices[valid].long(), :MXFP4_NUM_GROUPS
    ].float()  # [V, 14]
    scale = torch.pow(2.0, selected_scales - 127.0)

    nope = (
        _decode_e2m1_torch(codes).reshape(-1, MXFP4_NUM_GROUPS, MXFP4_GROUP_SIZE)
        * scale.unsqueeze(-1)
    ).reshape(-1, MXFP4_NOPE_DIM)

    rope = rope_rows[token_indices[valid].long()].to(torch.bfloat16)
    out[valid, 0, :MXFP4_NOPE_DIM] = nope.to(torch.bfloat16)
    out[valid, 0, MXFP4_NOPE_DIM:] = rope


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quantize_dsv4_mxfp4_k_cache_into(
    cache_k: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    page_size: int,
) -> None:
    """Quantize BF16/FP16/FP32 DSV4 keys → MXFP4 and scatter by token ID.

    Args:
        cache_k:  [num_tokens, 512] or [num_tokens, 1, 512] — input K (nope+rope).
        kv_buffer: [num_pages, page_size * 368] uint8 — destination pool.
        loc:       [num_tokens] int32/int64 — flat row index per token.
        page_size: tokens per page (typically 128 or 256).
    """
    k_nope, k_rope = _split_k(cache_k)
    rows = _as_rows(kv_buffer, page_size)
    loc = _validate_loc(loc, rows)
    if k_nope.shape[0] != loc.numel():
        raise ValueError(
            f"cache_k and loc token counts differ: "
            f"{k_nope.shape[0]} vs {loc.numel()}"
        )
    if not (k_nope.device == k_rope.device == rows.device):
        raise ValueError("cache_k and KV buffer must be on one device")
    if loc.numel() == 0:
        return

    packed_rows = rows[:, :MXFP4_PACKED_NOPE_BYTES]
    scale_rows = rows[
        :,
        MXFP4_PACKED_NOPE_BYTES : MXFP4_PACKED_NOPE_BYTES + MXFP4_SCALE_BYTES,
    ]
    rope_rows = rows[:, -MXFP4_ROPE_BYTES:].view(torch.bfloat16)

    if rows.is_cuda:
        groups_per_program, num_warps = _quantize_launch_config(loc.numel())
        num_parts = triton.cdiv(MXFP4_NUM_GROUPS, groups_per_program)
        _quantize_mxfp4_k_cache_into_kernel[(loc.numel(), num_parts)](
            k_nope,
            k_rope,
            packed_rows,
            scale_rows,
            rope_rows,
            loc,
            rows.shape[0],
            k_nope.stride(0),
            k_rope.stride(0),
            packed_rows.stride(0),
            scale_rows.stride(0),
            rope_rows.stride(0),
            NUM_GROUPS=MXFP4_NUM_GROUPS,
            GROUP_SIZE=MXFP4_GROUP_SIZE,
            GROUPS_PER_PROGRAM=groups_per_program,
            ROPE_DIM=MXFP4_ROPE_DIM,
            num_warps=num_warps,
        )
    else:
        _quantize_mxfp4_reference(
            k_nope, k_rope, packed_rows, scale_rows, rope_rows, loc
        )


def dequantize_dsv4_mxfp4_k_cache_paged(
    kv_buffer: torch.Tensor,
    token_indices: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather and dequantize selected MXFP4 cache rows → BF16.

    Args:
        kv_buffer:     [num_pages, page_size * 368] uint8.
        token_indices: [num_tokens] int32/int64 — flat row indices to gather.
        page_size:     tokens per page.
        out:           optional [num_tokens, 1, 512] BF16 output tensor.

    Returns:
        [num_tokens, 1, 512] BF16 tensor.
    """
    rows = _as_rows(kv_buffer, page_size)
    token_indices = _validate_loc(token_indices, rows)
    shape = (token_indices.numel(), 1, MXFP4_TOTAL_DIM)
    if out is None:
        out = torch.empty(shape, dtype=torch.bfloat16, device=rows.device)
    elif out.shape != shape or out.dtype != torch.bfloat16:
        raise ValueError(
            f"out must be BF16 with shape {shape}, got {out.dtype} {tuple(out.shape)}"
        )
    elif out.device != rows.device or not out.is_contiguous():
        raise ValueError("out and KV buffer must be contiguous and on one device")
    if token_indices.numel() == 0:
        return out

    packed_rows = rows[:, :MXFP4_PACKED_NOPE_BYTES]
    scale_rows = rows[
        :,
        MXFP4_PACKED_NOPE_BYTES : MXFP4_PACKED_NOPE_BYTES + MXFP4_SCALE_BYTES,
    ]
    rope_rows = rows[:, -MXFP4_ROPE_BYTES:].view(torch.bfloat16)

    if rows.is_cuda:
        output_2d = out.view(-1, MXFP4_TOTAL_DIM)
        _dequantize_mxfp4_k_cache_paged_kernel[(token_indices.numel(),)](
            packed_rows,
            scale_rows,
            rope_rows,
            token_indices,
            output_2d,
            rows.shape[0],
            packed_rows.stride(0),
            scale_rows.stride(0),
            rope_rows.stride(0),
            output_2d.stride(0),
            MAX_GROUPS=_MAX_GROUPS,
            GROUP_SIZE=MXFP4_GROUP_SIZE,
            NOPE_DIM=MXFP4_NOPE_DIM,
            NOPE_TILE=_NOPE_TILE,
            ROPE_DIM=MXFP4_ROPE_DIM,
            num_warps=4,
        )
    else:
        _dequantize_mxfp4_reference(
            packed_rows, scale_rows, rope_rows, token_indices, out
        )

    return out


__all__ = [
    "MXFP4_BYTES_PER_TOKEN",
    "MXFP4_NOPE_DIM",
    "MXFP4_ROPE_DIM",
    "MXFP4_TOTAL_DIM",
    "MXFP4_GROUP_SIZE",
    "MXFP4_NUM_GROUPS",
    "MXFP4_PACKED_NOPE_BYTES",
    "MXFP4_SCALE_BYTES",
    "MXFP4_ROPE_BYTES",
    "quantize_dsv4_mxfp4_k_cache_into",
    "dequantize_dsv4_mxfp4_k_cache_paged",
]
