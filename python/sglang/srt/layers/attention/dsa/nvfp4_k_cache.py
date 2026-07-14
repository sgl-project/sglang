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


def _e2m1_rne_torch(x: torch.Tensor) -> torch.Tensor:
    """Return E2M1 codes using round-to-nearest, ties-to-even.

    Positive magnitudes are ``[0, .5, 1, 1.5, 2, 3, 4, 6]``.  At each
    midpoint the code with an even low significand bit wins, hence the
    alternating strict/non-strict comparisons below.
    """

    magnitude = x.abs()
    code = (
        (magnitude > 0.25).to(torch.uint8)
        + (magnitude >= 0.75).to(torch.uint8)
        + (magnitude > 1.25).to(torch.uint8)
        + (magnitude >= 1.75).to(torch.uint8)
        + (magnitude > 2.5).to(torch.uint8)
        + (magnitude >= 3.5).to(torch.uint8)
        + (magnitude > 5.0).to(torch.uint8)
    )
    return code | (torch.signbit(x).to(torch.uint8) << 3)


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
    normalized = torch.where(denominator > 0, blocks / denominator, 0.0)
    codes = _e2m1_rne_torch(normalized).reshape(-1, NVFP4_LATENT_DIM)
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
def _e2m1_rne_triton(x):
    magnitude = tl.abs(x)
    code = (
        (magnitude > 0.25).to(tl.uint8)
        + (magnitude >= 0.75).to(tl.uint8)
        + (magnitude > 1.25).to(tl.uint8)
        + (magnitude >= 1.75).to(tl.uint8)
        + (magnitude > 2.5).to(tl.uint8)
        + (magnitude >= 3.5).to(tl.uint8)
        + (magnitude > 5.0).to(tl.uint8)
    )
    sign = (x < 0.0).to(tl.uint8) << 3
    return code | sign


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
):
    token_id = tl.program_id(0)
    part_id = tl.program_id(1)
    dst_row = tl.load(loc_ptr + token_id).to(tl.int64)
    lanes = tl.arange(0, 64)
    valid_dst = (dst_row >= 0) & (dst_row < num_rows)

    if part_id < NUM_LATENT_BLOCKS:
        block_offset = part_id * 16
        block_lanes = lanes < 16
        x = tl.load(
            k_nope_ptr + token_id * k_nope_stride_0 + block_offset + lanes,
            mask=block_lanes,
            other=0.0,
        ).to(tl.float32)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
        unrounded_scale = tl.max(tl.abs(x)) / (6.0 * global_scale)
        unrounded_scale = tl.clamp(unrounded_scale, 0.0, 448.0)
        # Cast through the typed destination pointer.  This is the same Triton
        # FP8 conversion pattern used by the existing NSA FP8 cache kernel and
        # gives the rounded E4M3 value needed for the following normalization.
        rounded_scale_fp8 = unrounded_scale.to(scale_ptr.dtype.element_ty)
        rounded_scale = rounded_scale_fp8.to(tl.float32)
        denominator = rounded_scale * global_scale
        inverse_denominator = tl.where(denominator > 0.0, 1.0 / denominator, 0.0)

        packed_lanes = lanes < 8
        even = tl.load(
            k_nope_ptr + token_id * k_nope_stride_0 + block_offset + 2 * lanes,
            mask=packed_lanes,
            other=0.0,
        ).to(tl.float32)
        odd = tl.load(
            k_nope_ptr + token_id * k_nope_stride_0 + block_offset + 2 * lanes + 1,
            mask=packed_lanes,
            other=0.0,
        ).to(tl.float32)
        even_code = _e2m1_rne_triton(even * inverse_denominator)
        odd_code = _e2m1_rne_triton(odd * inverse_denominator)
        packed = even_code | (odd_code << 4)

        tl.store(
            packed_ptr + dst_row * packed_stride_0 + part_id * 8 + lanes,
            packed,
            mask=valid_dst & packed_lanes,
        )
        tl.store(
            scale_ptr + dst_row * scale_stride_0 + part_id,
            rounded_scale_fp8,
            mask=valid_dst,
        )
    else:
        rope_lanes = lanes < 64
        rope = tl.load(
            k_rope_ptr + token_id * k_rope_stride_0 + lanes,
            mask=rope_lanes,
            other=0.0,
        )
        tl.store(
            rope_ptr + dst_row * rope_stride_0 + lanes,
            rope,
            mask=valid_dst & rope_lanes,
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
    part_id = tl.program_id(1)
    src_row = tl.load(page_indices_ptr + token_id).to(tl.int64)
    lanes = tl.arange(0, 64)
    valid_src = (src_row >= 0) & (src_row < num_rows)

    if part_id < NUM_LATENT_BLOCKS:
        latent_lanes = lanes < 16
        packed = tl.load(
            packed_ptr + src_row * packed_stride_0 + part_id * 8 + lanes // 2,
            mask=valid_src & latent_lanes,
            other=0,
        ).to(tl.uint8)
        code = tl.where((lanes & 1) == 0, packed & 0x0F, packed >> 4)
        scale = tl.load(
            scale_ptr + src_row * scale_stride_0 + part_id,
            mask=valid_src,
            other=0.0,
        ).to(tl.float32)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
        value = _decode_e2m1_triton(code) * scale * global_scale
        tl.store(
            output_ptr + token_id * output_stride_0 + part_id * 16 + lanes,
            value,
            mask=latent_lanes,
        )
    else:
        rope_lanes = lanes < 64
        rope = tl.load(
            rope_ptr + src_row * rope_stride_0 + lanes,
            mask=valid_src & rope_lanes,
            other=0.0,
        )
        tl.store(
            output_ptr + token_id * output_stride_0 + LATENT_DIM + lanes,
            rope,
            mask=rope_lanes,
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
    _quantize_nvfp4_k_cache_into_kernel[(loc.numel(), _NUM_LATENT_BLOCKS + 1)](
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
        num_warps=1,
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
    _dequantize_nvfp4_k_cache_paged_kernel[
        (page_indices.numel(), _NUM_LATENT_BLOCKS + 1)
    ](
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
        num_warps=1,
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
