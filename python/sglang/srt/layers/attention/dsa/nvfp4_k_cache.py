"""GLM-5.2 DSA mixed NVFP4/BF16 key-cache codec.

The SM100 cache row is intentionally byte-addressed and has one canonical
layout per physical token::

    [ E2M1 latent (256 B) | E4M3 block-16 scales (32 B) | BF16 RoPE (128 B) ]

Only the 512-dimensional latent vector uses NVFP4 block scaling.  The
64-dimensional RoPE vector is stored directly as BF16.  The fused attention
producer dequantizes the latent vector to BF16 and feeds the unchanged
FlashMLA FP8-cache BF16 QK/PV consumer.  A persistent FP32 global scale is
stored once per layer by the KV pool, outside this row.

The PyTorch functions are executable, bit-exact references.  The Triton path
scatters directly into the cache and is used for production cache writes; it
does not materialize a full-cache dequantization workspace.
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
_QUANTIZE_SMALL_BATCH_THRESHOLD = 64


def _as_feature_matrix(x: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError(
            f"{name} must have shape [num_tokens, {dim}] or "
            f"[num_tokens, 1, {dim}], got {tuple(x.shape)}"
        )
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"{name} must be BF16, FP16, or FP32, got {x.dtype}")
    return x if x.stride(1) == 1 else x.contiguous()


def _as_cache_rows(kv_buffer: torch.Tensor) -> torch.Tensor:
    if kv_buffer.dtype != torch.uint8:
        raise TypeError(f"kv_buffer must be uint8, got {kv_buffer.dtype}")
    if kv_buffer.ndim < 2 or kv_buffer.shape[-1] != NVFP4_BYTES_PER_TOKEN:
        raise ValueError(
            f"kv_buffer must have shape [..., {NVFP4_BYTES_PER_TOKEN}], "
            f"got {tuple(kv_buffer.shape)}"
        )
    if not kv_buffer.is_contiguous():
        raise ValueError("kv_buffer must be contiguous")
    return kv_buffer.view(-1, NVFP4_BYTES_PER_TOKEN)


def _as_global_scale(
    global_scale: torch.Tensor | Real, device: torch.device
) -> torch.Tensor:
    if isinstance(global_scale, torch.Tensor):
        if global_scale.numel() != 1:
            raise ValueError("global_scale must contain exactly one value")
        scale = global_scale.to(device=device, dtype=torch.float32).reshape(1)
        if not scale.is_cuda and not bool(
            torch.isfinite(scale).all() & (scale > 0).all()
        ):
            raise ValueError("global_scale must be finite and positive")
        return scale.contiguous()
    if not isinstance(global_scale, Real):
        raise TypeError("global_scale must be a number or one-element tensor")
    value = float(global_scale)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"global_scale must be finite and positive, got {value}")
    return torch.tensor([value], dtype=torch.float32, device=device)


def _validate_quant_inputs(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
):
    k_nope = _as_feature_matrix(k_nope, NVFP4_LATENT_DIM, "k_nope")
    k_rope = _as_feature_matrix(k_rope, NVFP4_ROPE_DIM, "k_rope")
    rows = _as_cache_rows(kv_buffer)
    loc = loc.reshape(-1)
    if k_nope.shape[0] != k_rope.shape[0] or k_nope.shape[0] != loc.numel():
        raise ValueError("k_nope, k_rope, and loc must have the same token count")
    if loc.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"loc must be int32 or int64, got {loc.dtype}")
    if not (k_nope.device == k_rope.device == rows.device == loc.device):
        raise ValueError("all codec tensors must be on one device")
    return k_nope, k_rope, rows, loc.contiguous()


def _sanitize_nonfinite_torch(x: torch.Tensor, global_scale: torch.Tensor):
    """Canonical finite-only policy for E2M1FN/E4M3FN storage.

    NaN becomes positive zero.  Positive/negative infinity saturates to the
    largest value representable by the recipe at the current global scale.
    This policy is deterministic in both the CPU reference and CUDA writer.
    """

    bound = _E2M1_MAX * _E4M3_MAX * global_scale
    return torch.nan_to_num(
        x, nan=0.0, posinf=float("inf"), neginf=float("-inf")
    ).clamp(min=-bound, max=bound)


def _e2m1_rne_scaled_torch(x: torch.Tensor, denominator: torch.Tensor):
    """Encode x/denominator using midpoint comparisons, without reciprocal.

    Strict/non-strict comparisons alternate at the seven E2M1 midpoints to
    implement round-to-nearest-even exactly.  Reciprocal multiplication is not
    equivalent at exact ties because it can move the value by one FP32 ULP.
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
    code |= torch.signbit(x).to(torch.uint8) << 3
    return torch.where(denominator > 0, code, torch.zeros_like(code))


def _decode_e2m1_torch(code: torch.Tensor):
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


def quantize_nvfp4_k_cache_into_reference(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    global_scale: torch.Tensor | Real,
) -> None:
    """Bit-exact PyTorch reference for the mixed DSA row writer."""

    k_nope, k_rope, rows, loc = _validate_quant_inputs(k_nope, k_rope, kv_buffer, loc)
    if loc.numel() == 0:
        return
    g = _as_global_scale(global_scale, rows.device)
    blocks = _sanitize_nonfinite_torch(k_nope.float(), g).reshape(
        -1, _NUM_LATENT_BLOCKS, NVFP4_BLOCK_SIZE
    )
    scale = (blocks.abs().amax(-1) / (_E2M1_MAX * g)).clamp(0, _E4M3_MAX)
    scale_fp8 = scale.to(torch.float8_e4m3fn)
    denominator = scale_fp8.float().unsqueeze(-1) * g
    codes = _e2m1_rne_scaled_torch(blocks, denominator).reshape(-1, NVFP4_LATENT_DIM)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    rope = torch.nan_to_num(
        k_rope.float(),
        nan=0.0,
        posinf=torch.finfo(torch.bfloat16).max,
        neginf=torch.finfo(torch.bfloat16).min,
    )
    rope_bytes = rope.to(torch.bfloat16).contiguous().view(torch.uint8)

    valid = (loc >= 0) & (loc < rows.shape[0])
    if not bool(valid.any()):
        return
    dst = loc[valid].long()
    rows[dst, :NVFP4_PACKED_LATENT_BYTES] = packed[valid]
    rows[
        dst,
        NVFP4_PACKED_LATENT_BYTES : NVFP4_PACKED_LATENT_BYTES + NVFP4_SCALE_BYTES,
    ] = scale_fp8[valid].contiguous().view(torch.uint8)
    rows[dst, -NVFP4_ROPE_BYTES:] = rope_bytes[valid]


def dequantize_nvfp4_k_cache_paged_reference(
    kv_buffer: torch.Tensor,
    indices: torch.Tensor,
    global_scale: torch.Tensor | Real,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather and decode rows using the exact stored bytes as the reference."""

    allowed = (torch.bfloat16, torch.float16, torch.float32, torch.float8_e4m3fn)
    if dtype not in allowed:
        raise ValueError(f"unsupported output dtype: {dtype}")
    rows = _as_cache_rows(kv_buffer)
    indices = indices.reshape(-1)
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be int32 or int64")
    if indices.device != rows.device:
        raise ValueError("indices and cache must be on one device")
    output = torch.zeros(
        (indices.numel(), 1, NVFP4_LATENT_DIM + NVFP4_ROPE_DIM),
        dtype=dtype,
        device=rows.device,
    )
    if indices.numel() == 0:
        return output
    valid = (indices >= 0) & (indices < rows.shape[0])
    if not bool(valid.any()):
        return output
    selected = rows[indices[valid].long()]
    packed = selected[:, :NVFP4_PACKED_LATENT_BYTES]
    codes = torch.empty(
        (selected.shape[0], NVFP4_LATENT_DIM), dtype=torch.uint8, device=rows.device
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
    rope = selected[:, -NVFP4_ROPE_BYTES:].contiguous().view(torch.bfloat16)
    output[valid, 0, :NVFP4_LATENT_DIM] = latent.to(dtype)
    output[valid, 0, NVFP4_LATENT_DIM:] = rope.to(dtype)
    return output


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
    return tl.where(denominator > 0, code | sign, 0).to(tl.uint8)


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
    rows_ptr,
    loc_ptr,
    global_scale_ptr,
    num_rows,
    k_nope_stride: tl.constexpr,
    k_rope_stride: tl.constexpr,
    row_stride: tl.constexpr,
    NUM_LATENT_BLOCKS: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    token = tl.program_id(0)
    part = tl.program_id(1)
    dst = tl.load(loc_ptr + token).to(tl.int64)
    valid_dst = (dst >= 0) & (dst < num_rows)
    safe_dst = tl.where(valid_dst, dst, 0)
    g = tl.load(global_scale_ptr).to(tl.float32)

    block = part * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    lane = tl.arange(0, 16)
    valid_block = block < NUM_LATENT_BLOCKS
    offsets = block[:, None] * 16 + lane[None, :]
    x = tl.load(
        k_nope_ptr + token * k_nope_stride + offsets,
        mask=valid_dst & valid_block[:, None],
        other=0.0,
    ).to(tl.float32)
    # E2M1FN has no non-finite encoding: NaN -> +0, Inf -> recipe maximum.
    bound = 6.0 * 448.0 * g
    x = tl.where(
        x != x,  # noqa: PLR0124  # Triton NaN test
        0.0,
        tl.maximum(tl.minimum(x, bound), -bound),
    )
    scale = tl.clamp(tl.max(tl.abs(x), axis=1) / (6.0 * g), 0.0, 448.0)
    scale_fp8 = scale.to(tl.float8e4nv)
    denominator = tl.expand_dims(scale_fp8.to(tl.float32) * g, 1)
    codes = _e2m1_rne_scaled_triton(x, denominator)
    pairs = tl.reshape(codes, (BLOCKS_PER_PROGRAM, 8, 2))
    low, high = tl.split(pairs)
    byte = tl.arange(0, 8)
    tl.store(
        rows_ptr + safe_dst * row_stride + block[:, None] * 8 + byte[None, :],
        low | (high << 4),
        mask=valid_dst & valid_block[:, None],
    )
    tl.store(
        rows_ptr + safe_dst * row_stride + PACKED_BYTES + block,
        scale_fp8.to(tl.uint8, bitcast=True),
        mask=valid_dst & valid_block,
    )

    if part == tl.cdiv(NUM_LATENT_BLOCKS, BLOCKS_PER_PROGRAM) - 1:
        rope_offset = tl.arange(0, ROPE_DIM)
        rope = tl.load(
            k_rope_ptr + token * k_rope_stride + rope_offset,
            mask=valid_dst,
            other=0.0,
        ).to(tl.float32)
        rope = tl.where(
            rope != rope,  # noqa: PLR0124  # Triton NaN test
            0.0,
            tl.maximum(tl.minimum(rope, 3.3895313892515355e38), -3.3895313892515355e38),
        ).to(tl.bfloat16)
        rope_bits = rope.to(tl.uint16, bitcast=True)
        tl.store(
            rows_ptr
            + safe_dst * row_stride
            + PACKED_BYTES
            + SCALE_BYTES
            + 2 * rope_offset,
            rope_bits & 0xFF,
            mask=valid_dst,
        )
        tl.store(
            rows_ptr
            + safe_dst * row_stride
            + PACKED_BYTES
            + SCALE_BYTES
            + 2 * rope_offset
            + 1,
            rope_bits >> 8,
            mask=valid_dst,
        )


def quantize_nvfp4_k_cache_into(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    global_scale: torch.Tensor | Real,
) -> None:
    """Quantize and scatter DSA keys into the 416-byte mixed cache rows."""

    k_nope, k_rope, rows, loc = _validate_quant_inputs(k_nope, k_rope, kv_buffer, loc)
    if loc.numel() == 0:
        return
    scale = _as_global_scale(global_scale, rows.device)
    if not rows.is_cuda:
        quantize_nvfp4_k_cache_into_reference(k_nope, k_rope, rows, loc, scale)
        return
    blocks_per_program = 8 if loc.numel() <= _QUANTIZE_SMALL_BATCH_THRESHOLD else 32
    num_warps = 1 if blocks_per_program == 8 else 4
    parts = triton.cdiv(_NUM_LATENT_BLOCKS, blocks_per_program)
    _quantize_nvfp4_k_cache_into_kernel[(loc.numel(), parts)](
        k_nope,
        k_rope,
        rows,
        loc,
        scale,
        rows.shape[0],
        k_nope.stride(0),
        k_rope.stride(0),
        rows.stride(0),
        NUM_LATENT_BLOCKS=_NUM_LATENT_BLOCKS,
        PACKED_BYTES=NVFP4_PACKED_LATENT_BYTES,
        SCALE_BYTES=NVFP4_SCALE_BYTES,
        ROPE_DIM=NVFP4_ROPE_DIM,
        BLOCKS_PER_PROGRAM=blocks_per_program,
        num_warps=num_warps,
    )


def dequantize_nvfp4_k_cache_paged(
    kv_buffer: torch.Tensor,
    indices: torch.Tensor,
    global_scale: torch.Tensor | Real,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather NVFP4 rows and decode them with one allocation and one kernel.

    Production decode consumes the packed cache directly.  The current
    FlashMLA sparse-prefill compatibility path calls this helper only for the
    selected prefix rows; cache inspection, copy/offload validation, and
    backend-attribution benchmarks also use it.  Keeping gather and decode
    fused avoids the large chain of temporary tensors created by the bit-exact
    PyTorch reference.
    """

    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported production dequant dtype: {dtype}")
    rows = _as_cache_rows(kv_buffer)
    flat_indices = indices.reshape(-1)
    if flat_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be int32 or int64")
    if flat_indices.device != rows.device:
        raise ValueError("indices and cache must be on one device")
    output = torch.empty(
        (flat_indices.numel(), 1, NVFP4_LATENT_DIM + NVFP4_ROPE_DIM),
        dtype=dtype,
        device=rows.device,
    )
    if flat_indices.numel() == 0:
        return output
    scale = _as_global_scale(global_scale, rows.device)
    _dequantize_nvfp4_k_cache_paged_kernel[(flat_indices.numel(),)](
        rows,
        flat_indices,
        scale,
        output,
        rows.shape[0],
        rows.stride(0),
        output.stride(0),
        BLOCK=1024,
        num_warps=4,
    )
    return output


@triton.jit
def _dequantize_nvfp4_k_cache_paged_kernel(
    rows_ptr,
    indices_ptr,
    global_scale_ptr,
    output_ptr,
    num_rows,
    row_stride: tl.constexpr,
    output_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    dimension = tl.arange(0, BLOCK)
    physical = tl.load(indices_ptr + token).to(tl.int64)
    valid_row = (physical >= 0) & (physical < num_rows)
    safe_physical = tl.where(valid_row, physical, 0)
    row = rows_ptr + safe_physical * row_stride

    latent_mask = (dimension < 512) & valid_row
    packed = tl.load(
        row + dimension // 2,
        mask=latent_mask,
        other=0,
    ).to(tl.uint8)
    code = tl.where((dimension & 1) == 0, packed & 0x0F, packed >> 4)
    scale_bits = tl.load(
        row + 256 + dimension // 16,
        mask=latent_mask,
        other=0,
    ).to(tl.uint8)
    block_scale = scale_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    latent = _decode_e2m1_triton(code) * block_scale * global_scale

    rope_dimension = dimension - 512
    rope_mask = (dimension >= 512) & (dimension < 576) & valid_row
    rope_low = tl.load(
        row + 288 + 2 * rope_dimension,
        mask=rope_mask,
        other=0,
    ).to(tl.uint8)
    rope_high = tl.load(
        row + 288 + 2 * rope_dimension + 1,
        mask=rope_mask,
        other=0,
    ).to(tl.uint8)
    rope_bits = rope_low.to(tl.uint16) | (rope_high.to(tl.uint16) << 8)
    rope = rope_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    value = tl.where(dimension < 512, latent, rope)
    tl.store(
        output_ptr + token * output_stride + dimension,
        value,
        mask=dimension < 576,
    )


__all__ = [
    "NVFP4_BLOCK_SIZE",
    "NVFP4_BYTES_PER_TOKEN",
    "NVFP4_LATENT_DIM",
    "NVFP4_PACKED_LATENT_BYTES",
    "NVFP4_ROPE_BYTES",
    "NVFP4_ROPE_DIM",
    "NVFP4_SCALE_BYTES",
    "dequantize_nvfp4_k_cache_paged",
    "dequantize_nvfp4_k_cache_paged_reference",
    "quantize_nvfp4_k_cache_into",
    "quantize_nvfp4_k_cache_into_reference",
]
