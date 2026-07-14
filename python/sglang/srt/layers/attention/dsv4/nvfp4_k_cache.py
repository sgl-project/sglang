"""Experimental NVFP4 codec for DeepSeek V4 attention KV entries.

Each logical entry stores 448 non-RoPE values in E2M1 with block-16 E4M3
scales and keeps the 64 RoPE values in BF16::

    [224 B packed E2M1 | 28 B E4M3 scales | 128 B BF16 RoPE] = 380 B

The physical pool is page-major ``[num_pages, page_size * 380]``. This module
only owns storage conversion; the initial attention integration dequantizes
selected entries before calling the existing BF16 sparse FlashMLA kernel.
"""

from __future__ import annotations

from numbers import Real
from typing import Optional

import torch

from sglang.srt.layers.attention.dsa.nvfp4_k_cache import (
    NVFP4_BLOCK_SIZE,
    _as_feature_matrix,
    _as_global_scale,
    _decode_e2m1_torch,
    _dequantize_nvfp4_k_cache_paged_kernel,
    _e2m1_rne_torch,
    _quantize_nvfp4_k_cache_into_kernel,
)

DSV4_NVFP4_NOPE_DIM = 448
DSV4_NVFP4_ROPE_DIM = 64
DSV4_NVFP4_PACKED_NOPE_BYTES = DSV4_NVFP4_NOPE_DIM // 2
DSV4_NVFP4_SCALE_BYTES = DSV4_NVFP4_NOPE_DIM // NVFP4_BLOCK_SIZE
DSV4_NVFP4_ROPE_BYTES = DSV4_NVFP4_ROPE_DIM * 2
DSV4_NVFP4_BYTES_PER_TOKEN = (
    DSV4_NVFP4_PACKED_NOPE_BYTES + DSV4_NVFP4_SCALE_BYTES + DSV4_NVFP4_ROPE_BYTES
)
_NUM_NOPE_BLOCKS = DSV4_NVFP4_NOPE_DIM // NVFP4_BLOCK_SIZE


def _as_rows(kv_buffer: torch.Tensor, page_size: int) -> torch.Tensor:
    if kv_buffer.dtype != torch.uint8:
        raise TypeError(
            f"DeepSeek V4 NVFP4 buffer must be uint8, got {kv_buffer.dtype}"
        )
    if kv_buffer.ndim != 2:
        raise ValueError(
            "DeepSeek V4 NVFP4 buffer must be [num_pages, bytes_per_page], "
            f"got {tuple(kv_buffer.shape)}"
        )
    expected = page_size * DSV4_NVFP4_BYTES_PER_TOKEN
    if kv_buffer.shape[1] != expected:
        raise ValueError(
            f"DeepSeek V4 NVFP4 page must contain {expected} bytes, "
            f"got {kv_buffer.shape[1]}"
        )
    if not kv_buffer.is_contiguous():
        raise ValueError("DeepSeek V4 NVFP4 buffer must be contiguous")
    return kv_buffer.view(-1, DSV4_NVFP4_BYTES_PER_TOKEN)


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
    if cache_k.ndim != 2 or cache_k.shape[1] != (
        DSV4_NVFP4_NOPE_DIM + DSV4_NVFP4_ROPE_DIM
    ):
        raise ValueError(
            "DeepSeek V4 K must be [num_tokens, 512] or [num_tokens, 1, 512], "
            f"got {tuple(cache_k.shape)}"
        )
    return (
        _as_feature_matrix(
            cache_k[:, :DSV4_NVFP4_NOPE_DIM],
            DSV4_NVFP4_NOPE_DIM,
            "k_nope",
        ),
        _as_feature_matrix(
            cache_k[:, DSV4_NVFP4_NOPE_DIM:],
            DSV4_NVFP4_ROPE_DIM,
            "k_rope",
        ),
    )


def quantize_dsv4_nvfp4_k_cache_into(
    cache_k: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    page_size: int,
    global_scale: torch.Tensor | Real,
) -> None:
    """Quantize BF16/FP16/FP32 DeepSeek V4 keys and scatter by token ID."""
    k_nope, k_rope = _split_k(cache_k)
    rows = _as_rows(kv_buffer, page_size)
    loc = _validate_loc(loc, rows)
    if k_nope.shape[0] != loc.numel():
        raise ValueError(
            f"cache_k and loc token counts differ: {k_nope.shape[0]} vs {loc.numel()}"
        )
    if not (k_nope.device == k_rope.device == rows.device):
        raise ValueError("cache_k and KV buffer must be on one device")
    if loc.numel() == 0:
        return

    global_scale_tensor = _as_global_scale(global_scale, rows.device)
    packed_rows = rows[:, :DSV4_NVFP4_PACKED_NOPE_BYTES]
    scale_rows = rows[
        :,
        DSV4_NVFP4_PACKED_NOPE_BYTES : DSV4_NVFP4_PACKED_NOPE_BYTES
        + DSV4_NVFP4_SCALE_BYTES,
    ].view(torch.float8_e4m3fn)
    rope_rows = rows[:, -DSV4_NVFP4_ROPE_BYTES:].view(torch.bfloat16)

    if rows.is_cuda:
        _quantize_nvfp4_k_cache_into_kernel[(loc.numel(), _NUM_NOPE_BLOCKS + 1)](
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
            NUM_LATENT_BLOCKS=_NUM_NOPE_BLOCKS,
            num_warps=1,
        )
        return

    blocks = k_nope.float().reshape(-1, _NUM_NOPE_BLOCKS, NVFP4_BLOCK_SIZE)
    scale = (blocks.abs().amax(dim=-1) / (6.0 * global_scale_tensor)).clamp(
        min=0.0, max=448.0
    )
    scale_fp8 = scale.to(torch.float8_e4m3fn)
    denominator = scale_fp8.float().unsqueeze(-1) * global_scale_tensor
    normalized = torch.where(denominator > 0, blocks / denominator, 0.0)
    codes = _e2m1_rne_torch(normalized).reshape(-1, DSV4_NVFP4_NOPE_DIM)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    valid = (loc >= 0) & (loc < rows.shape[0])
    if bool(valid.any()):
        dst = loc[valid].long()
        packed_rows[dst] = packed[valid]
        scale_rows[dst] = scale_fp8[valid]
        rope_rows[dst] = k_rope[valid].to(torch.bfloat16)


def dequantize_dsv4_nvfp4_k_cache_paged(
    kv_buffer: torch.Tensor,
    token_indices: torch.Tensor,
    page_size: int,
    global_scale: torch.Tensor | Real,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather selected entries into a BF16 ``[tokens, 1, 512]`` tensor."""
    rows = _as_rows(kv_buffer, page_size)
    token_indices = _validate_loc(token_indices, rows)
    shape = (token_indices.numel(), 1, DSV4_NVFP4_NOPE_DIM + DSV4_NVFP4_ROPE_DIM)
    if out is None:
        out = torch.empty(shape, dtype=torch.bfloat16, device=rows.device)
    elif out.shape != shape or out.dtype != torch.bfloat16:
        raise ValueError(
            f"out must be BF16 with shape {shape}, got {out.dtype} {out.shape}"
        )
    elif out.device != rows.device or not out.is_contiguous():
        raise ValueError("out and KV buffer must be contiguous and on one device")
    if token_indices.numel() == 0:
        return out

    global_scale_tensor = _as_global_scale(global_scale, rows.device)
    packed_rows = rows[:, :DSV4_NVFP4_PACKED_NOPE_BYTES]
    scale_rows = rows[
        :,
        DSV4_NVFP4_PACKED_NOPE_BYTES : DSV4_NVFP4_PACKED_NOPE_BYTES
        + DSV4_NVFP4_SCALE_BYTES,
    ].view(torch.float8_e4m3fn)
    rope_rows = rows[:, -DSV4_NVFP4_ROPE_BYTES:].view(torch.bfloat16)

    if rows.is_cuda:
        _dequantize_nvfp4_k_cache_paged_kernel[
            (token_indices.numel(), _NUM_NOPE_BLOCKS + 1)
        ](
            packed_rows,
            scale_rows,
            rope_rows,
            token_indices,
            global_scale_tensor,
            out,
            rows.shape[0],
            packed_rows.stride(0),
            scale_rows.stride(0),
            rope_rows.stride(0),
            out.stride(0),
            NUM_LATENT_BLOCKS=_NUM_NOPE_BLOCKS,
            LATENT_DIM=DSV4_NVFP4_NOPE_DIM,
            num_warps=1,
        )
        return out

    out.zero_()
    valid = (token_indices >= 0) & (token_indices < rows.shape[0])
    if not bool(valid.any()):
        return out
    selected = rows[token_indices[valid].long()]
    packed = selected[:, :DSV4_NVFP4_PACKED_NOPE_BYTES]
    codes = torch.empty(
        (selected.shape[0], DSV4_NVFP4_NOPE_DIM),
        dtype=torch.uint8,
        device=rows.device,
    )
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    scales = (
        selected[
            :,
            DSV4_NVFP4_PACKED_NOPE_BYTES : DSV4_NVFP4_PACKED_NOPE_BYTES
            + DSV4_NVFP4_SCALE_BYTES,
        ]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    nope = (
        _decode_e2m1_torch(codes).reshape(-1, _NUM_NOPE_BLOCKS, NVFP4_BLOCK_SIZE)
        * scales.unsqueeze(-1)
        * global_scale_tensor
    ).reshape(-1, DSV4_NVFP4_NOPE_DIM)
    rope = selected[:, -DSV4_NVFP4_ROPE_BYTES:].contiguous().view(torch.bfloat16)
    out[valid, 0, :DSV4_NVFP4_NOPE_DIM] = nope.to(torch.bfloat16)
    out[valid, 0, DSV4_NVFP4_NOPE_DIM:] = rope
    return out


__all__ = [
    "DSV4_NVFP4_BYTES_PER_TOKEN",
    "DSV4_NVFP4_NOPE_DIM",
    "DSV4_NVFP4_ROPE_DIM",
    "quantize_dsv4_nvfp4_k_cache_into",
    "dequantize_dsv4_nvfp4_k_cache_paged",
]
