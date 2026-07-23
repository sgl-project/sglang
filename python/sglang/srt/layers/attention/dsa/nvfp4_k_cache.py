"""NVFP4 codec for the mixed-format NSA key cache.

Each token occupies one 416-byte row::

    [ E2M1 latent (256 B) | E4M3 scales (32 B) | BF16 RoPE (128 B) ]

Only the 512-dimensional latent vector is quantized.  The 64-dimensional
RoPE vector stays in bfloat16, matching the existing NSA FP8 cache policy.
The CUDA path uses Triton and scatters directly into the destination cache;
the PyTorch implementations are kept as executable references and CPU
fallbacks.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
import triton
import triton.language as tl

NVFP4_BLOCK_SIZE = 16
NVFP4_LATENT_DIM = 512
NVFP4_ROPE_DIM = 64
NVFP4_PACKED_LATENT_BYTES = 256
NVFP4_SCALE_BYTES = 32
NVFP4_ROPE_BYTES = 128
NVFP4_BYTES_PER_TOKEN = 416

_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
_NUM_LATENT_BLOCKS = NVFP4_LATENT_DIM // NVFP4_BLOCK_SIZE

# Large prefill writes have enough token-level parallelism for one CTA to own
# a complete row.  Decode and other small batches retain four CTAs per token
# so a single row can still use more than one SM.  These are private launch
# details shared with the DSV4 codec; the public cache API is unchanged.
_QUANTIZE_SMALL_BATCH_THRESHOLD = 64
_QUANTIZE_SMALL_BLOCKS_PER_PROGRAM = 8
_QUANTIZE_SMALL_NUM_WARPS = 1
_QUANTIZE_LARGE_BLOCKS_PER_PROGRAM = 32
_QUANTIZE_LARGE_NUM_WARPS = 4


def _quantize_nvfp4_launch_config(num_tokens: int) -> tuple[int, int]:
    if num_tokens <= _QUANTIZE_SMALL_BATCH_THRESHOLD:
        return (
            _QUANTIZE_SMALL_BLOCKS_PER_PROGRAM,
            _QUANTIZE_SMALL_NUM_WARPS,
        )
    return (
        _QUANTIZE_LARGE_BLOCKS_PER_PROGRAM,
        _QUANTIZE_LARGE_NUM_WARPS,
    )


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


def _as_cache_rows(kv_buffer: torch.Tensor) -> torch.Tensor:
    if kv_buffer.dtype != torch.uint8:
        raise TypeError(f"kv_buffer must be uint8, got {kv_buffer.dtype}")
    if kv_buffer.ndim < 2 or kv_buffer.shape[-1] != NVFP4_BYTES_PER_TOKEN:
        raise ValueError(
            "kv_buffer must have shape [..., "
            f"{NVFP4_BYTES_PER_TOKEN}], got {tuple(kv_buffer.shape)}"
        )
    if not kv_buffer.is_contiguous():
        raise ValueError("kv_buffer must be contiguous")
    return kv_buffer.view(-1, NVFP4_BYTES_PER_TOKEN)


def _as_global_scale(
    global_scale: torch.Tensor | Real, device: torch.device
) -> torch.Tensor:
    if isinstance(global_scale, torch.Tensor):
        if global_scale.numel() != 1:
            raise ValueError(
                f"global_scale must contain exactly one value, got {global_scale.numel()}"
            )
        scale = (
            global_scale.to(device=device, dtype=torch.float32).reshape(1).contiguous()
        )
        # CUDA callers pass the persistent per-layer scale on every token.  Its
        # finite-positive invariant must be checked when that persistent tensor
        # is configured; synchronizing here would break CUDA-graph friendliness.
        # CPU/reference calls can reject invalid tensor values directly.
        if not scale.is_cuda and not bool(
            torch.isfinite(scale).all() & (scale > 0).all()
        ):
            raise ValueError("global_scale tensor must be finite and positive")
        return scale
    if not isinstance(global_scale, Real):
        raise TypeError("global_scale must be a scalar number or one-element tensor")
    value = float(global_scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"global_scale must be positive, got {global_scale}")
    return torch.tensor([value], dtype=torch.float32, device=device)


def _validate_common_quant_inputs(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k_nope = _as_feature_matrix(k_nope, NVFP4_LATENT_DIM, "k_nope")
    k_rope = _as_feature_matrix(k_rope, NVFP4_ROPE_DIM, "k_rope")
    rows = _as_cache_rows(kv_buffer)
    loc = loc.reshape(-1)

    if k_nope.shape[0] != k_rope.shape[0] or k_nope.shape[0] != loc.numel():
        raise ValueError(
            "k_nope, k_rope, and loc must have the same token count, got "
            f"{k_nope.shape[0]}, {k_rope.shape[0]}, and {loc.numel()}"
        )
    if loc.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"loc must be int32 or int64, got {loc.dtype}")
    if not (k_nope.device == k_rope.device == rows.device == loc.device):
        raise ValueError("k_nope, k_rope, kv_buffer, and loc must be on one device")
    return k_nope, k_rope, rows, loc.contiguous()


def _e2m1_rne_scaled_torch(
    x: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    """Return E2M1 codes for ``x / denominator`` without dividing.

    Computing a reciprocal and multiplying can move an exact E2M1 midpoint
    by one FP32 ulp and therefore break ties-to-even.  Comparing the original
    magnitude with each midpoint derived from the rounded, stored block scale
    avoids that reciprocal perturbation and follows the recipe's mathematical
    rounding rule. Non-positive denominators represent an all-zero block and
    produce positive zero codes.
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
    lut = torch.tensor(
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
        device=code.device,
    )
    return lut[code.long()]


def _quantize_nvfp4_k_cache_into_reference(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    global_scale: torch.Tensor | Real,
) -> None:
    """PyTorch reference for :func:`quantize_nvfp4_k_cache_into`."""

    k_nope, k_rope, rows, loc = _validate_common_quant_inputs(
        k_nope, k_rope, kv_buffer, loc
    )
    if loc.numel() == 0:
        return

    g = _as_global_scale(global_scale, k_nope.device)
    blocks = k_nope.float().reshape(-1, _NUM_LATENT_BLOCKS, NVFP4_BLOCK_SIZE)
    scale = (blocks.abs().amax(dim=-1) / (_E2M1_MAX * g)).clamp(min=0.0, max=_E4M3_MAX)

    # This cast is deliberately before normalization.  It is the E4M3 RNE
    # value stored in the cache, and therefore the value the E2M1 codes must
    # be derived from.
    scale_fp8 = scale.to(torch.float8_e4m3fn)
    rounded_scale = scale_fp8.float()
    denominator = rounded_scale.unsqueeze(-1) * g
    codes = _e2m1_rne_scaled_torch(blocks, denominator).reshape(
        -1, NVFP4_LATENT_DIM
    )
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    rope_bytes = k_rope.to(torch.bfloat16).contiguous().view(torch.uint8)

    valid = (loc >= 0) & (loc < rows.shape[0])
    if not bool(valid.any()):
        return
    dst = loc[valid].long()
    rows[dst, :NVFP4_PACKED_LATENT_BYTES] = packed[valid]
    rows[
        dst,
        NVFP4_PACKED_LATENT_BYTES : NVFP4_PACKED_LATENT_BYTES + NVFP4_SCALE_BYTES,
    ] = (
        scale_fp8[valid].contiguous().view(torch.uint8)
    )
    rows[dst, -NVFP4_ROPE_BYTES:] = rope_bytes[valid]


def _dequantize_nvfp4_k_cache_paged_reference(
    kv_buffer: torch.Tensor,
    page_indices: torch.Tensor,
    global_scale: torch.Tensor | Real,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """PyTorch reference for :func:`dequantize_nvfp4_k_cache_paged`."""

    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Unsupported output dtype: {dtype}")
    rows = _as_cache_rows(kv_buffer)
    page_indices = page_indices.reshape(-1)
    if page_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"page_indices must be int32 or int64, got {page_indices.dtype}"
        )
    if page_indices.device != rows.device:
        raise ValueError("page_indices and kv_buffer must be on one device")

    num_tokens = page_indices.numel()
    output = torch.zeros(
        (num_tokens, 1, NVFP4_LATENT_DIM + NVFP4_ROPE_DIM),
        dtype=dtype,
        device=rows.device,
    )
    if num_tokens == 0:
        return output

    valid = (page_indices >= 0) & (page_indices < rows.shape[0])
    if not bool(valid.any()):
        return output
    selected = rows[page_indices[valid].long()]
    packed = selected[:, :NVFP4_PACKED_LATENT_BYTES]
    codes = torch.empty(
        (selected.shape[0], NVFP4_LATENT_DIM),
        dtype=torch.uint8,
        device=rows.device,
    )
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    scales = (
        selected[
            :,
            NVFP4_PACKED_LATENT_BYTES : NVFP4_PACKED_LATENT_BYTES + NVFP4_SCALE_BYTES,
        ]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    g = _as_global_scale(global_scale, rows.device)
    latent = (
        _decode_e2m1_torch(codes).reshape(-1, _NUM_LATENT_BLOCKS, NVFP4_BLOCK_SIZE)
        * scales.unsqueeze(-1)
        * g
    ).reshape(-1, NVFP4_LATENT_DIM)
    rope = (
        selected[:, -NVFP4_ROPE_BYTES:]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(-1, NVFP4_ROPE_DIM)
    )
    output[valid, 0, :NVFP4_LATENT_DIM] = latent.to(dtype)
    output[valid, 0, NVFP4_LATENT_DIM:] = rope.to(dtype)
    return output


@triton.jit
def _e2m1_rne_scaled_triton(x, denominator):
    """Encode ``x / denominator`` with exact scaled-midpoint comparisons."""

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
    # Preserve E2M1's signed zero.  The quantization input is FP32 here, so
    # its sign bit is the high bit of the bitcast uint32 value.
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


@triton.jit
def _quantize_nvfp4_k_cache_into_kernel(
    k_nope_ptr,
    k_rope_ptr,
    packed_ptr,
    scale_ptr,
    rope_ptr,
    loc_ptr,
    global_scale_ptr,
    num_rows,
    k_nope_stride_0: tl.constexpr,
    k_rope_stride_0: tl.constexpr,
    packed_stride_0: tl.constexpr,
    scale_stride_0: tl.constexpr,
    rope_stride_0: tl.constexpr,
    NUM_LATENT_BLOCKS: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    token_id = tl.program_id(0)
    part_id = tl.program_id(1)
    dst_row = tl.load(loc_ptr + token_id).to(tl.int64)
    valid_dst = (dst_row >= 0) & (dst_row < num_rows)
    safe_dst_row = tl.where(valid_dst, dst_row, 0)

    # A program quantizes several adjacent block-16 groups.  The previous
    # kernel launched one CTA per block and loaded every input twice (once for
    # the reduction and again for packing).  Reusing this [blocks, 16] tile
    # removes both costs while preserving E4M3-before-E2M1 rounding semantics.
    block_ids = part_id * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    element_ids = tl.arange(0, 16)
    valid_block = block_ids < NUM_LATENT_BLOCKS
    input_offsets = block_ids[:, None] * 16 + element_ids[None, :]
    x = tl.load(
        k_nope_ptr + token_id * k_nope_stride_0 + input_offsets,
        mask=valid_dst & valid_block[:, None],
        other=0.0,
    ).to(tl.float32)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    unrounded_scale = tl.max(tl.abs(x), axis=1) / (6.0 * global_scale)
    unrounded_scale = tl.clamp(unrounded_scale, 0.0, 448.0)
    rounded_scale_fp8 = unrounded_scale.to(scale_ptr.dtype.element_ty)
    rounded_scale = rounded_scale_fp8.to(tl.float32)
    denominator = tl.expand_dims(rounded_scale * global_scale, 1)
    codes = _e2m1_rne_scaled_triton(x, denominator)
    code_pairs = tl.reshape(codes, (BLOCKS_PER_PROGRAM, 8, 2))
    low_code, high_code = tl.split(code_pairs)
    packed = low_code | (high_code << 4)

    byte_ids = tl.arange(0, 8)
    packed_offsets = block_ids[:, None] * 8 + byte_ids[None, :]
    tl.store(
        packed_ptr + safe_dst_row * packed_stride_0 + packed_offsets,
        packed,
        mask=valid_dst & valid_block[:, None],
    )
    tl.store(
        scale_ptr + safe_dst_row * scale_stride_0 + block_ids,
        rounded_scale_fp8,
        mask=valid_dst & valid_block,
    )

    num_parts = tl.cdiv(NUM_LATENT_BLOCKS, BLOCKS_PER_PROGRAM)
    if part_id == num_parts - 1:
        rope_offsets = tl.arange(0, 64)
        rope = tl.load(
            k_rope_ptr + token_id * k_rope_stride_0 + rope_offsets,
            mask=valid_dst,
            other=0.0,
        )
        tl.store(
            rope_ptr + safe_dst_row * rope_stride_0 + rope_offsets,
            rope,
            mask=valid_dst,
        )


@triton.jit
def _dequantize_nvfp4_k_cache_paged_kernel(
    packed_ptr,
    scale_ptr,
    rope_ptr,
    page_indices_ptr,
    global_scale_ptr,
    output_ptr,
    num_rows,
    packed_stride_0: tl.constexpr,
    scale_stride_0: tl.constexpr,
    rope_stride_0: tl.constexpr,
    output_stride_0: tl.constexpr,
    NUM_LATENT_BLOCKS: tl.constexpr,
    LATENT_DIM: tl.constexpr,
):
    token_id = tl.program_id(0)
    src_row = tl.load(page_indices_ptr + token_id).to(tl.int64)
    valid_src = (src_row >= 0) & (src_row < num_rows)

    # One CTA owns one complete token row.  The old layout launched 33
    # independent one-warp programs per token (32 latent blocks plus RoPE),
    # repeatedly loading the row index/global scale and paying CTA scheduling
    # overhead for only 16 useful latent lanes.  Here a [32, 8] tensor loads
    # every packed byte and every block scale exactly once, then joins the low
    # and high nibbles into one contiguous 512-value output vector.
    # DSA uses all 32 blocks while DSV4 reuses this kernel with 28 blocks.
    # Keep the compile-time tile power-of-two and mask the padded blocks so a
    # single implementation covers both row layouts.
    block_ids = tl.arange(0, 32)[:, None]
    byte_ids = tl.arange(0, 8)[None, :]
    packed_offsets = block_ids * 8 + byte_ids
    valid_block = block_ids < NUM_LATENT_BLOCKS
    packed = tl.load(
        packed_ptr + src_row * packed_stride_0 + packed_offsets,
        mask=valid_src & valid_block,
        other=0,
    ).to(tl.uint8)
    scale = tl.load(
        scale_ptr + src_row * scale_stride_0 + block_ids,
        mask=valid_src & valid_block,
        other=0.0,
    ).to(tl.float32)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    low = _decode_e2m1_triton(packed & 0x0F) * scale * global_scale
    high = _decode_e2m1_triton(packed >> 4) * scale * global_scale
    latent = tl.reshape(tl.join(low, high), (512,))
    latent_offsets = tl.arange(0, 512)
    tl.store(
        output_ptr + token_id * output_stride_0 + latent_offsets,
        latent,
        mask=latent_offsets < LATENT_DIM,
    )

    rope_offsets = tl.arange(0, 64)
    rope = tl.load(
        rope_ptr + src_row * rope_stride_0 + rope_offsets,
        mask=valid_src,
        other=0.0,
    )
    tl.store(
        output_ptr + token_id * output_stride_0 + LATENT_DIM + rope_offsets,
        rope,
    )


def quantize_nvfp4_k_cache_into(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    global_scale: torch.Tensor | Real,
) -> None:
    """Quantize and scatter NSA keys directly into a mixed NVFP4 cache.

    ``loc`` contains flattened token-row indices. Entries outside the cache
    capacity are ignored, which is useful for padded graph batches. Duplicate
    valid indices are not supported because their writes would race on CUDA. A tensor
    ``global_scale`` is a persistent per-layer device scalar and must be
    configured to a finite positive value before entering this hot path.
    """

    k_nope, k_rope, rows, loc = _validate_common_quant_inputs(
        k_nope, k_rope, kv_buffer, loc
    )
    if loc.numel() == 0:
        return
    global_scale_tensor = _as_global_scale(global_scale, rows.device)
    if not rows.is_cuda:
        _quantize_nvfp4_k_cache_into_reference(
            k_nope, k_rope, rows, loc, global_scale_tensor
        )
        return

    packed_rows = rows[:, :NVFP4_PACKED_LATENT_BYTES]
    scale_rows = rows[
        :,
        NVFP4_PACKED_LATENT_BYTES : NVFP4_PACKED_LATENT_BYTES + NVFP4_SCALE_BYTES,
    ].view(torch.float8_e4m3fn)
    rope_rows = rows[:, -NVFP4_ROPE_BYTES:].view(torch.bfloat16)
    blocks_per_program, num_warps = _quantize_nvfp4_launch_config(loc.numel())
    num_parts = triton.cdiv(_NUM_LATENT_BLOCKS, blocks_per_program)
    _quantize_nvfp4_k_cache_into_kernel[(loc.numel(), num_parts)](
        k_nope,
        k_rope,
        packed_rows,
        scale_rows,
        rope_rows,
        loc,
        global_scale_tensor,
        rows.shape[0],
        k_nope.stride(0),
        k_rope.stride(0),
        packed_rows.stride(0),
        scale_rows.stride(0),
        rope_rows.stride(0),
        NUM_LATENT_BLOCKS=_NUM_LATENT_BLOCKS,
        BLOCKS_PER_PROGRAM=blocks_per_program,
        num_warps=num_warps,
    )


def dequantize_nvfp4_k_cache_paged(
    kv_buffer: torch.Tensor,
    page_indices: torch.Tensor,
    global_scale: torch.Tensor | Real,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather and dequantize selected NSA cache rows.

    The output has shape ``[page_indices.numel(), 1, 576]``. An index outside
    ``[0, cache_capacity)`` produces one all-zero row and never dereferences
    the cache. A tensor ``global_scale`` follows the same finite-positive
    invariant as the quantization entry point.
    """

    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Unsupported output dtype: {dtype}")
    rows = _as_cache_rows(kv_buffer)
    page_indices = page_indices.reshape(-1)
    if page_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"page_indices must be int32 or int64, got {page_indices.dtype}"
        )
    if page_indices.device != rows.device:
        raise ValueError("page_indices and kv_buffer must be on one device")
    if not page_indices.is_contiguous():
        page_indices = page_indices.contiguous()
    if not rows.is_cuda:
        return _dequantize_nvfp4_k_cache_paged_reference(
            rows, page_indices, global_scale, dtype
        )

    global_scale_tensor = _as_global_scale(global_scale, rows.device)
    output = torch.empty(
        (page_indices.numel(), 1, NVFP4_LATENT_DIM + NVFP4_ROPE_DIM),
        dtype=dtype,
        device=rows.device,
    )
    if page_indices.numel() == 0:
        return output

    packed_rows = rows[:, :NVFP4_PACKED_LATENT_BYTES]
    scale_rows = rows[
        :,
        NVFP4_PACKED_LATENT_BYTES : NVFP4_PACKED_LATENT_BYTES + NVFP4_SCALE_BYTES,
    ].view(torch.float8_e4m3fn)
    rope_rows = rows[:, -NVFP4_ROPE_BYTES:].view(torch.bfloat16)
    output_2d = output.view(-1, NVFP4_LATENT_DIM + NVFP4_ROPE_DIM)
    _dequantize_nvfp4_k_cache_paged_kernel[(page_indices.numel(),)](
        packed_rows,
        scale_rows,
        rope_rows,
        page_indices,
        global_scale_tensor,
        output_2d,
        rows.shape[0],
        packed_rows.stride(0),
        scale_rows.stride(0),
        rope_rows.stride(0),
        output_2d.stride(0),
        NUM_LATENT_BLOCKS=_NUM_LATENT_BLOCKS,
        LATENT_DIM=NVFP4_LATENT_DIM,
        num_warps=4,
    )
    return output


__all__ = [
    "NVFP4_BLOCK_SIZE",
    "NVFP4_LATENT_DIM",
    "NVFP4_ROPE_DIM",
    "NVFP4_PACKED_LATENT_BYTES",
    "NVFP4_SCALE_BYTES",
    "NVFP4_ROPE_BYTES",
    "NVFP4_BYTES_PER_TOKEN",
    "quantize_nvfp4_k_cache_into",
    "dequantize_nvfp4_k_cache_paged",
]
