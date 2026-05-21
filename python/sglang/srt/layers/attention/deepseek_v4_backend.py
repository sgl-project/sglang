from __future__ import annotations

import enum
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
    # NOTE: should eventually be the only compressor backend
    from sglang.srt.layers.attention.dsv4.compressor_v2 import (
        CompressorBackendMixin,
        FusedCompressMetadata,
        create_paged_compressor_data,
    )
else:
    from sglang.srt.layers.attention.dsv4.compressor import (
        CompressorBackendMixin,
        FusedCompressMetadata,
        create_paged_compressor_data,
    )

from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import (
    PagedIndexerMetadata,
    copy_metadata,
    maybe_copy_inplace,
)
from sglang.srt.layers.attention.dsv4.metadata_kernel import (
    init_compression_metadata as _init_compression_metadata_triton,
)
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    fp8_dtype as _dsv4_k_cache_fp8_dtype,
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import ceil_align

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

SWA_WINDOW = 128
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64


T = TypeVar("T", bound=Optional[torch.Tensor])


def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
    if x is None:
        return None
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    return F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)


def _create_flashmla_metadata():
    import flash_mla

    return flash_mla.get_mla_metadata()[0]


def _dsv4_use_bf16_sparse_prefill() -> bool:
    return envs.SGLANG_DSV4_USE_BF16_SPARSE_PREFILL.get()


_DSV4_BF16_SPARSE_PREFILL_LAST_LOG_TIME_BY_RATIO: Dict[int, float] = {}


def _dsv4_apply_topk_length(
    indices: torch.Tensor, topk_length: Optional[torch.Tensor]
) -> torch.Tensor:
    if topk_length is None:
        return indices

    indices = indices.clone()
    arange = torch.arange(indices.shape[-1], device=indices.device).view(
        *([1] * (indices.ndim - 1)), indices.shape[-1]
    )
    length = topk_length
    while length.ndim < indices.ndim:
        length = length.unsqueeze(-1)
    indices[arange >= length] = -1
    return indices


def _dsv4_shift_valid_indices(indices: torch.Tensor, offset: int) -> torch.Tensor:
    return torch.where(indices >= 0, indices + offset, indices)


def _dsv4_pad_indices_last_dim(
    indices: torch.Tensor, multiple: int = 128
) -> torch.Tensor:
    pad = (-indices.shape[-1]) % multiple
    if pad == 0:
        return indices
    return F.pad(indices, (0, pad), value=-1)


def _dsv4_dequantize_model1_fp8_sparse_k_cache_torch_ref(
    k_cache: torch.Tensor,
) -> torch.Tensor:
    num_blocks, block_size, h_k, _ = k_cache.shape
    assert h_k == 1
    d, d_nope, d_rope, tile_size, num_tiles = 512, 448, 64, 64, 7
    k_cache = k_cache.view(num_blocks, -1)
    nope_rope = k_cache[:, : block_size * (d_nope + 2 * d_rope)].view(
        num_blocks, block_size, d_nope + 2 * d_rope
    )
    nope = nope_rope[:, :, :d_nope]
    rope = nope_rope[:, :, d_nope:].view(torch.bfloat16)
    scale = (
        k_cache[:, block_size * (d_nope + 2 * d_rope) :]
        .view(num_blocks, block_size, 8)[:, :, :num_tiles]
        .view(torch.float8_e8m0fnu)
    )

    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=k_cache.device
    )
    result[..., d_nope:] = rope
    for tile_idx in range(num_tiles):
        start = tile_idx * tile_size
        end = start + tile_size
        result[..., start:end] = (
            nope[..., start:end].to(torch.bfloat16)
            * scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
        )
    return result.view(num_blocks, block_size, 1, d)


_dsv4_dequantize_model1_fp8_sparse_k_cache_torch = (
    _dsv4_dequantize_model1_fp8_sparse_k_cache_torch_ref
)


@triton.jit
def _dsv4_dequantize_model1_fp8_sparse_k_cache_kernel(
    k_cache_fp8_ptr,
    k_cache_bf16_ptr,
    k_cache_u8_ptr,
    out_ptr,
    k_stride_0: tl.constexpr,
    k_bf16_stride_0: tl.constexpr,
    out_stride_0: tl.constexpr,
    out_token_offset: tl.constexpr,
    block_size: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    VALUE_BYTES_PER_TOKEN: tl.constexpr,
    VALUE_BF16_PER_TOKEN: tl.constexpr,
    SCALE_BYTES_PER_TOKEN: tl.constexpr,
    SCALE_BIAS: tl.constexpr,
):
    token_linear = tl.program_id(0)
    block_id = token_linear // block_size
    token_offset = token_linear - block_id * block_size
    out_token_id = out_token_offset + token_linear

    offsets = tl.arange(0, BLOCK_DIM)
    nope_mask = offsets < DIM_NOPE
    rope_mask = (offsets >= DIM_NOPE) & (offsets < DIM_NOPE + DIM_ROPE)

    value_base = block_id * k_stride_0 + token_offset * VALUE_BYTES_PER_TOKEN
    x_fp32 = tl.load(
        k_cache_fp8_ptr + value_base + offsets,
        mask=nope_mask,
        other=0.0,
    ).to(tl.float32)

    scale_offsets = (
        block_id * k_stride_0
        + block_size * VALUE_BYTES_PER_TOKEN
        + token_offset * SCALE_BYTES_PER_TOKEN
        + offsets // 64
    )
    scale_u8 = tl.load(
        k_cache_u8_ptr + scale_offsets,
        mask=nope_mask,
        other=127,
    ).to(tl.float32)
    y_nope = (x_fp32 * tl.exp2(scale_u8 - SCALE_BIAS)).to(out_ptr.dtype.element_ty)

    rope_offsets = (
        block_id * k_bf16_stride_0
        + token_offset * VALUE_BF16_PER_TOKEN
        + (DIM_NOPE // 2)
        + (offsets - DIM_NOPE)
    )
    y_rope = tl.load(k_cache_bf16_ptr + rope_offsets, mask=rope_mask, other=0.0)

    y = tl.where(nope_mask, y_nope, y_rope)
    tl.store(
        out_ptr + out_token_id * out_stride_0 + offsets,
        y,
        mask=offsets < DIM_NOPE + DIM_ROPE,
    )


@triton.jit
def _dsv4_dequantize_two_model1_fp8_sparse_k_caches_kernel(
    swa_k_cache_fp8_ptr,
    swa_k_cache_bf16_ptr,
    swa_k_cache_u8_ptr,
    extra_k_cache_fp8_ptr,
    extra_k_cache_bf16_ptr,
    extra_k_cache_u8_ptr,
    out_ptr,
    swa_num_tokens,
    swa_k_stride_0: tl.constexpr,
    swa_k_bf16_stride_0: tl.constexpr,
    extra_k_stride_0: tl.constexpr,
    extra_k_bf16_stride_0: tl.constexpr,
    out_stride_0: tl.constexpr,
    SWA_BLOCK_SIZE: tl.constexpr,
    EXTRA_BLOCK_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    VALUE_BYTES_PER_TOKEN: tl.constexpr,
    VALUE_BF16_PER_TOKEN: tl.constexpr,
    SCALE_BYTES_PER_TOKEN: tl.constexpr,
    SCALE_BIAS: tl.constexpr,
):
    token_linear = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_DIM)
    is_extra = token_linear >= swa_num_tokens
    nope_mask = offsets < DIM_NOPE
    rope_mask = (offsets >= DIM_NOPE) & (offsets < DIM_NOPE + DIM_ROPE)

    swa_block_id = token_linear // SWA_BLOCK_SIZE
    swa_token_offset = token_linear - swa_block_id * SWA_BLOCK_SIZE
    extra_token_linear = token_linear - swa_num_tokens
    extra_block_id = extra_token_linear // EXTRA_BLOCK_SIZE
    extra_token_offset = extra_token_linear - extra_block_id * EXTRA_BLOCK_SIZE

    swa_value_base = (
        swa_block_id * swa_k_stride_0 + swa_token_offset * VALUE_BYTES_PER_TOKEN
    )
    extra_value_base = (
        extra_block_id * extra_k_stride_0 + extra_token_offset * VALUE_BYTES_PER_TOKEN
    )
    x_swa = tl.load(
        swa_k_cache_fp8_ptr + swa_value_base + offsets,
        mask=(~is_extra) & nope_mask,
        other=0.0,
    ).to(tl.float32)
    x_extra = tl.load(
        extra_k_cache_fp8_ptr + extra_value_base + offsets,
        mask=is_extra & nope_mask,
        other=0.0,
    ).to(tl.float32)
    x_fp32 = tl.where(is_extra, x_extra, x_swa)

    scale_offsets = offsets // 64
    swa_scale_offsets = (
        swa_block_id * swa_k_stride_0
        + SWA_BLOCK_SIZE * VALUE_BYTES_PER_TOKEN
        + swa_token_offset * SCALE_BYTES_PER_TOKEN
        + scale_offsets
    )
    extra_scale_offsets = (
        extra_block_id * extra_k_stride_0
        + EXTRA_BLOCK_SIZE * VALUE_BYTES_PER_TOKEN
        + extra_token_offset * SCALE_BYTES_PER_TOKEN
        + scale_offsets
    )
    scale_swa = tl.load(
        swa_k_cache_u8_ptr + swa_scale_offsets,
        mask=(~is_extra) & nope_mask,
        other=127,
    ).to(tl.float32)
    scale_extra = tl.load(
        extra_k_cache_u8_ptr + extra_scale_offsets,
        mask=is_extra & nope_mask,
        other=127,
    ).to(tl.float32)
    scale_u8 = tl.where(is_extra, scale_extra, scale_swa)
    y_nope = (x_fp32 * tl.exp2(scale_u8 - SCALE_BIAS)).to(out_ptr.dtype.element_ty)

    swa_rope_offsets = (
        swa_block_id * swa_k_bf16_stride_0
        + swa_token_offset * VALUE_BF16_PER_TOKEN
        + (DIM_NOPE // 2)
        + (offsets - DIM_NOPE)
    )
    extra_rope_offsets = (
        extra_block_id * extra_k_bf16_stride_0
        + extra_token_offset * VALUE_BF16_PER_TOKEN
        + (DIM_NOPE // 2)
        + (offsets - DIM_NOPE)
    )
    y_rope_swa = tl.load(
        swa_k_cache_bf16_ptr + swa_rope_offsets,
        mask=(~is_extra) & rope_mask,
        other=0.0,
    )
    y_rope_extra = tl.load(
        extra_k_cache_bf16_ptr + extra_rope_offsets,
        mask=is_extra & rope_mask,
        other=0.0,
    )
    y_rope = tl.where(is_extra, y_rope_extra, y_rope_swa)

    y = tl.where(nope_mask, y_nope, y_rope)
    tl.store(
        out_ptr + token_linear * out_stride_0 + offsets,
        y,
        mask=offsets < DIM_NOPE + DIM_ROPE,
    )


def _dsv4_get_k_cache_fp8_view(k_cache: torch.Tensor) -> torch.Tensor:
    if k_cache.dtype == torch.uint8:
        return k_cache.view(_dsv4_k_cache_fp8_dtype)
    return k_cache


def _dsv4_dequantize_model1_fp8_sparse_k_cache_into(
    k_cache: torch.Tensor,
    out: torch.Tensor,
    out_token_offset: int,
) -> None:
    num_blocks, block_size, h_k, dim_quant = k_cache.shape
    assert h_k == 1
    assert dim_quant == 584
    assert out.dtype == torch.bfloat16
    assert out.shape[-1] == 512

    num_tokens = num_blocks * block_size
    if num_tokens == 0:
        return

    k_cache_fp8 = _dsv4_get_k_cache_fp8_view(k_cache)
    k_cache_bf16 = k_cache_fp8.view(torch.bfloat16)
    k_cache_u8 = k_cache_fp8.view(torch.uint8)

    d_nope, d_rope, block_dim = 448, 64, 512
    _dsv4_dequantize_model1_fp8_sparse_k_cache_kernel[(num_tokens,)](
        k_cache_fp8,
        k_cache_bf16,
        k_cache_u8,
        out,
        k_cache_fp8.stride(0),
        k_cache_bf16.stride(0),
        out.stride(0),
        out_token_offset,
        block_size,
        DIM_NOPE=d_nope,
        DIM_ROPE=d_rope,
        BLOCK_DIM=block_dim,
        VALUE_BYTES_PER_TOKEN=d_nope + 2 * d_rope,
        VALUE_BF16_PER_TOKEN=(d_nope + 2 * d_rope) // 2,
        SCALE_BYTES_PER_TOKEN=8,
        SCALE_BIAS=127.0,
        num_warps=4,
    )


def _dsv4_dequantize_two_model1_fp8_sparse_k_caches_into(
    swa_k_cache: torch.Tensor,
    extra_k_cache: torch.Tensor,
    out: torch.Tensor,
) -> None:
    swa_num_blocks, swa_block_size, swa_h_k, swa_dim_quant = swa_k_cache.shape
    extra_num_blocks, extra_block_size, extra_h_k, extra_dim_quant = extra_k_cache.shape
    assert swa_h_k == 1 and extra_h_k == 1
    assert swa_dim_quant == 584 and extra_dim_quant == 584
    assert out.dtype == torch.bfloat16
    assert out.shape[-1] == 512

    swa_num_tokens = swa_num_blocks * swa_block_size
    extra_num_tokens = extra_num_blocks * extra_block_size
    assert out.shape[0] >= swa_num_tokens + extra_num_tokens
    if swa_num_tokens == 0:
        _dsv4_dequantize_model1_fp8_sparse_k_cache_into(extra_k_cache, out, 0)
        return
    if extra_num_tokens == 0:
        _dsv4_dequantize_model1_fp8_sparse_k_cache_into(swa_k_cache, out, 0)
        return

    swa_k_cache_fp8 = _dsv4_get_k_cache_fp8_view(swa_k_cache)
    swa_k_cache_bf16 = swa_k_cache_fp8.view(torch.bfloat16)
    swa_k_cache_u8 = swa_k_cache_fp8.view(torch.uint8)
    extra_k_cache_fp8 = _dsv4_get_k_cache_fp8_view(extra_k_cache)
    extra_k_cache_bf16 = extra_k_cache_fp8.view(torch.bfloat16)
    extra_k_cache_u8 = extra_k_cache_fp8.view(torch.uint8)

    d_nope, d_rope, block_dim = 448, 64, 512
    _dsv4_dequantize_two_model1_fp8_sparse_k_caches_kernel[
        (swa_num_tokens + extra_num_tokens,)
    ](
        swa_k_cache_fp8,
        swa_k_cache_bf16,
        swa_k_cache_u8,
        extra_k_cache_fp8,
        extra_k_cache_bf16,
        extra_k_cache_u8,
        out,
        swa_num_tokens,
        swa_k_cache_fp8.stride(0),
        swa_k_cache_bf16.stride(0),
        extra_k_cache_fp8.stride(0),
        extra_k_cache_bf16.stride(0),
        out.stride(0),
        SWA_BLOCK_SIZE=swa_block_size,
        EXTRA_BLOCK_SIZE=extra_block_size,
        DIM_NOPE=d_nope,
        DIM_ROPE=d_rope,
        BLOCK_DIM=block_dim,
        VALUE_BYTES_PER_TOKEN=d_nope + 2 * d_rope,
        VALUE_BF16_PER_TOKEN=(d_nope + 2 * d_rope) // 2,
        SCALE_BYTES_PER_TOKEN=8,
        SCALE_BIAS=127.0,
        num_warps=4,
    )


def _dsv4_dequantize_model1_fp8_sparse_k_cache(k_cache: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, h_k, dim_quant = k_cache.shape
    assert h_k == 1
    assert dim_quant == 584
    out = torch.empty(
        (num_blocks * block_size, 1, 512),
        dtype=torch.bfloat16,
        device=k_cache.device,
    )
    _dsv4_dequantize_model1_fp8_sparse_k_cache_into(k_cache, out, 0)
    return out.view(num_blocks, block_size, 1, 512)


def _dsv4_topk_length_2d(topk_length: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if topk_length is None:
        return None
    if topk_length.ndim == 1:
        return topk_length.unsqueeze(1)
    if topk_length.ndim == 2:
        return topk_length
    if topk_length.ndim == 3 and topk_length.shape[-1] == 1:
        return topk_length.squeeze(-1)
    raise AssertionError(f"unsupported topk_length shape: {tuple(topk_length.shape)}")


@triton.jit
def _dsv4_build_unified_prefill_indices_kernel(
    swa_indices_ptr,
    extra_indices_ptr,
    swa_lengths_ptr,
    extra_lengths_ptr,
    out_ptr,
    swa_stride_0: tl.constexpr,
    swa_stride_1: tl.constexpr,
    swa_stride_2: tl.constexpr,
    extra_stride_0: tl.constexpr,
    extra_stride_1: tl.constexpr,
    extra_stride_2: tl.constexpr,
    swa_len_stride_0: tl.constexpr,
    swa_len_stride_1: tl.constexpr,
    extra_len_stride_0: tl.constexpr,
    extra_len_stride_1: tl.constexpr,
    out_stride_0: tl.constexpr,
    out_stride_1: tl.constexpr,
    out_stride_2: tl.constexpr,
    H_KV: tl.constexpr,
    SWA_TOPK: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    OUT_TOPK: tl.constexpr,
    SWA_LENGTH_H: tl.constexpr,
    EXTRA_LENGTH_H: tl.constexpr,
    EXTRA_OFFSET: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_SWA_LENGTH: tl.constexpr,
    HAS_EXTRA_LENGTH: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row_id = tl.program_id(0)
    q_id = row_id // H_KV
    h_id = row_id - q_id * H_KV
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)

    values = tl.full((BLOCK_TOPK,), -1, dtype=tl.int32)

    swa_mask = offsets < SWA_TOPK
    swa_raw = tl.load(
        swa_indices_ptr
        + q_id * swa_stride_0
        + h_id * swa_stride_1
        + offsets * swa_stride_2,
        mask=swa_mask,
        other=-1,
    )
    swa_length_valid = swa_mask
    if HAS_SWA_LENGTH:
        swa_len_h_id = tl.minimum(h_id, SWA_LENGTH_H - 1)
        swa_len = tl.load(
            swa_lengths_ptr + q_id * swa_len_stride_0 + swa_len_h_id * swa_len_stride_1
        )
        swa_length_valid = swa_mask & (offsets < swa_len)
    values = tl.where(swa_length_valid, swa_raw, values)

    if HAS_EXTRA:
        extra_offsets = offsets - SWA_TOPK
        extra_mask = (offsets >= SWA_TOPK) & (extra_offsets < EXTRA_TOPK)
        extra_raw = tl.load(
            extra_indices_ptr
            + q_id * extra_stride_0
            + h_id * extra_stride_1
            + extra_offsets * extra_stride_2,
            mask=extra_mask,
            other=-1,
        )
        extra_length_valid = extra_mask
        if HAS_EXTRA_LENGTH:
            extra_len_h_id = tl.minimum(h_id, EXTRA_LENGTH_H - 1)
            extra_len = tl.load(
                extra_lengths_ptr
                + q_id * extra_len_stride_0
                + extra_len_h_id * extra_len_stride_1
            )
            extra_length_valid = extra_mask & (extra_offsets < extra_len)
        extra_shifted = tl.where(extra_raw >= 0, extra_raw + EXTRA_OFFSET, extra_raw)
        values = tl.where(extra_length_valid, extra_shifted, values)

    out_mask = offsets < OUT_TOPK
    tl.store(
        out_ptr + q_id * out_stride_0 + h_id * out_stride_1 + offsets * out_stride_2,
        values,
        mask=out_mask,
    )


def _dsv4_build_unified_prefill_indices_triton(
    *,
    swa_page_indices: torch.Tensor,
    extra_indices: Optional[torch.Tensor],
    swa_topk_lengths: Optional[torch.Tensor],
    extra_topk_lengths: Optional[torch.Tensor],
    extra_offset: int,
) -> torch.Tensor:
    assert swa_page_indices.ndim == 3
    num_q, h_kv, swa_topk = swa_page_indices.shape
    has_extra = extra_indices is not None
    extra_topk = 0
    if has_extra:
        assert extra_indices is not None
        assert extra_indices.ndim == 3
        assert extra_indices.shape[:2] == (num_q, h_kv)
        extra_topk = extra_indices.shape[-1]

    out_topk = ceil_align(swa_topk + extra_topk, 128)
    out = torch.empty(
        (num_q, h_kv, out_topk),
        dtype=swa_page_indices.dtype,
        device=swa_page_indices.device,
    )

    swa_lengths_2d = _dsv4_topk_length_2d(swa_topk_lengths)
    extra_lengths_2d = _dsv4_topk_length_2d(extra_topk_lengths)

    empty_indices = swa_page_indices
    empty_lengths = torch.empty((1, 1), dtype=torch.int32, device=swa_page_indices.device)
    extra_indices_arg = extra_indices if extra_indices is not None else empty_indices
    swa_lengths_arg = swa_lengths_2d if swa_lengths_2d is not None else empty_lengths
    extra_lengths_arg = (
        extra_lengths_2d if extra_lengths_2d is not None else empty_lengths
    )

    block_topk = 128
    grid = (num_q * h_kv, triton.cdiv(out_topk, block_topk))
    _dsv4_build_unified_prefill_indices_kernel[grid](
        swa_page_indices,
        extra_indices_arg,
        swa_lengths_arg,
        extra_lengths_arg,
        out,
        swa_page_indices.stride(0),
        swa_page_indices.stride(1),
        swa_page_indices.stride(2),
        extra_indices_arg.stride(0),
        extra_indices_arg.stride(1),
        extra_indices_arg.stride(2),
        swa_lengths_arg.stride(0),
        swa_lengths_arg.stride(1),
        extra_lengths_arg.stride(0),
        extra_lengths_arg.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H_KV=h_kv,
        SWA_TOPK=swa_topk,
        EXTRA_TOPK=extra_topk,
        OUT_TOPK=out_topk,
        SWA_LENGTH_H=swa_lengths_arg.shape[1],
        EXTRA_LENGTH_H=extra_lengths_arg.shape[1],
        EXTRA_OFFSET=extra_offset,
        HAS_EXTRA=has_extra,
        HAS_SWA_LENGTH=swa_lengths_2d is not None,
        HAS_EXTRA_LENGTH=extra_lengths_2d is not None,
        BLOCK_TOPK=block_topk,
    )
    return out


def _dsv4_build_unified_prefill_inputs_from_real_decode(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    swa_page_indices: torch.Tensor,
    extra_indices: Optional[torch.Tensor],
    swa_topk_lengths: Optional[torch.Tensor],
    extra_topk_lengths: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return _dsv4_build_unified_prefill_inputs_from_real_decode_triton(
        q=q,
        swa_k_cache=swa_k_cache,
        extra_k_cache=extra_k_cache,
        swa_page_indices=swa_page_indices,
        extra_indices=extra_indices,
        swa_topk_lengths=swa_topk_lengths,
        extra_topk_lengths=extra_topk_lengths,
    )


def _dsv4_build_unified_prefill_inputs_from_real_decode_torch_ref(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    swa_page_indices: torch.Tensor,
    extra_indices: Optional[torch.Tensor],
    swa_topk_lengths: Optional[torch.Tensor],
    extra_topk_lengths: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if q.ndim != 4 or q.shape[1] != 1:
        return None
    if q.shape[-1] != 512:
        return None
    if swa_k_cache.shape[-1] != 584:
        return None

    q_prefill = q.squeeze(1).contiguous()
    swa_indices = _dsv4_apply_topk_length(swa_page_indices, swa_topk_lengths)
    swa_k_dequant = _dsv4_dequantize_model1_fp8_sparse_k_cache_torch_ref(swa_k_cache)
    kv_parts = [swa_k_dequant.reshape(-1, 1, q.shape[-1])]
    index_parts = [swa_indices]

    if extra_k_cache is not None:
        if extra_indices is None:
            return None
        if extra_k_cache.shape[-1] != 584:
            return None
        extra_indices = _dsv4_apply_topk_length(extra_indices, extra_topk_lengths)
        extra_k_dequant = _dsv4_dequantize_model1_fp8_sparse_k_cache_torch_ref(
            extra_k_cache
        )
        extra_offset = kv_parts[0].shape[0]
        kv_parts.append(extra_k_dequant.reshape(-1, 1, q.shape[-1]))
        index_parts.append(_dsv4_shift_valid_indices(extra_indices, extra_offset))

    kv_unified = torch.cat(kv_parts, dim=0).contiguous()
    indices_unified = torch.cat(index_parts, dim=-1).contiguous()
    indices_unified = _dsv4_pad_indices_last_dim(indices_unified)
    return q_prefill, kv_unified, indices_unified


def _dsv4_build_unified_prefill_inputs_from_real_decode_triton(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    swa_page_indices: torch.Tensor,
    extra_indices: Optional[torch.Tensor],
    swa_topk_lengths: Optional[torch.Tensor],
    extra_topk_lengths: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if q.ndim != 4 or q.shape[1] != 1:
        return None
    if q.shape[-1] != 512:
        return None
    if swa_k_cache.shape[-1] != 584:
        return None

    q_prefill = q.squeeze(1).contiguous()
    swa_num_tokens = swa_k_cache.shape[0] * swa_k_cache.shape[1]
    extra_num_tokens = 0

    if extra_k_cache is not None:
        if extra_indices is None:
            return None
        if extra_k_cache.shape[-1] != 584:
            return None
        extra_num_tokens = extra_k_cache.shape[0] * extra_k_cache.shape[1]

    kv_unified = torch.empty(
        (swa_num_tokens + extra_num_tokens, 1, q.shape[-1]),
        dtype=torch.bfloat16,
        device=q.device,
    )
    if extra_k_cache is None:
        _dsv4_dequantize_model1_fp8_sparse_k_cache_into(
            swa_k_cache, kv_unified, out_token_offset=0
        )
    else:
        _dsv4_dequantize_two_model1_fp8_sparse_k_caches_into(
            swa_k_cache, extra_k_cache, kv_unified
        )

    indices_unified = _dsv4_build_unified_prefill_indices_triton(
        swa_page_indices=swa_page_indices,
        extra_indices=extra_indices,
        swa_topk_lengths=swa_topk_lengths,
        extra_topk_lengths=extra_topk_lengths,
        extra_offset=swa_num_tokens,
    )
    return q_prefill, kv_unified, indices_unified


def _dsv4_run_bf16_sparse_prefill_attention(
    *,
    q_prefill: torch.Tensor,
    kv_unified: torch.Tensor,
    indices_unified: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
    compress_ratio: Literal[0, 4, 128],
) -> Optional[torch.Tensor]:
    global _DSV4_BF16_SPARSE_PREFILL_LAST_LOG_TIME_BY_RATIO

    import flash_mla

    if q_prefill.ndim != 3 or kv_unified.ndim != 3:
        return None
    if q_prefill.shape[-1] != kv_unified.shape[-1]:
        return None
    topk_length_unified = None

    now = time.monotonic()
    last_log_time = _DSV4_BF16_SPARSE_PREFILL_LAST_LOG_TIME_BY_RATIO.get(
        compress_ratio, 0.0
    )
    should_log_hit = q_prefill.shape[0] >= 512 and (now - last_log_time >= 0.2)
    if should_log_hit:
        _DSV4_BF16_SPARSE_PREFILL_LAST_LOG_TIME_BY_RATIO[compress_ratio] = now
        logger.warning(
            "DSV4 BF16 sparse prefill hit: source=fp8_dequant ratio=%s q=%s kv_unified=%s indices=%s",
            compress_ratio,
            tuple(q_prefill.shape),
            tuple(kv_unified.shape),
            tuple(indices_unified.shape),
        )

    return flash_mla.flash_mla_sparse_fwd(
        q_prefill,
        kv_unified,
        indices_unified,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length_unified,
    )[0]


def _create_dummy_paged_compress_data(compress_ratio: int):
    return None


@dataclass
class DSV4AttnMetadata:
    page_size: int
    page_table: torch.Tensor
    raw_out_loc: torch.Tensor
    cuda_int32_kwargs: dict

    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor

    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor

    c4_sparse_topk: int
    c4_out_loc: Optional[torch.Tensor] = None
    c4_topk_lengths_raw: Optional[torch.Tensor] = None
    c4_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)
    c4_sparse_page_indices: torch.Tensor = field(init=False)

    c128_out_loc: Optional[torch.Tensor] = None
    c128_page_indices: Optional[torch.Tensor] = None
    c128_topk_lengths_clamp1: Optional[torch.Tensor] = None

    c1_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c4_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c128_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)

    @property
    def positions(self) -> torch.Tensor:
        return self.positions_casual

    def get_flashmla_metadata(self, compress_ratio: Literal[0, 4, 128]):
        if compress_ratio == 0:
            return self.c1_flashmla_metadata
        elif compress_ratio == 4:
            return self.c4_flashmla_metadata
        elif compress_ratio == 128:
            return self.c128_flashmla_metadata
        else:
            raise ValueError(f"invalid {compress_ratio=}")

    def copy_(self, other: DSV4AttnMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[
                "c4_sparse_topk",
                "page_size",
                "cuda_int32_kwargs",
            ],
            copy_fields=[
                "raw_out_loc",
                "seq_lens_casual",
                "positions_casual",
                "c4_out_loc",
                "c128_out_loc",
                "page_table",
                "swa_page_indices",
                "swa_topk_lengths",
                "c128_page_indices",
                "c128_topk_lengths_clamp1",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_page_indices",
            ],
            assign_fields=[
                "c1_flashmla_metadata",
                "c4_flashmla_metadata",
                "c128_flashmla_metadata",
            ],
        )

    def init_compression_metadata(self):
        assert self.page_table.dim() == 2
        assert (
            self.raw_out_loc.shape == self.seq_lens_casual.shape
        ), f"{self.raw_out_loc.shape=}, {self.seq_lens_casual.shape=}"

        (
            self.c4_out_loc,
            _,
            self.c4_topk_lengths_raw,
            self.c4_topk_lengths_clamp1,
            self.c128_out_loc,
            _,
            self.c128_topk_lengths_clamp1,
            self.c128_page_indices,
        ) = _init_compression_metadata_triton(
            self.seq_lens_casual,
            self.positions_casual,
            self.raw_out_loc,
            self.page_table,
            self.page_size,
            compute_page_indices=True,
        )

        self.c128_page_indices = _pad_last_dim(self.c128_page_indices)
        self.swa_page_indices = _pad_last_dim(self.swa_page_indices)

    _CP_REINDEX_FIELDS = [
        "seq_lens_casual",
        "positions_casual",
        "swa_page_indices",
        "swa_topk_lengths",
        "page_table",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c128_page_indices",
        "c128_topk_lengths_clamp1",
    ]
    _CP_GLOBAL_FIELDS = [
        "raw_out_loc",
        "c4_out_loc",
        "c128_out_loc",
    ]

    def apply_cp_reindex(self) -> None:
        cp_rank = get_attention_cp_rank()
        cp_size = get_attention_cp_size()
        idx = slice(cp_rank, None, cp_size)
        pre_global_len = self.seq_lens_casual.shape[0]
        assert pre_global_len % cp_size == 0, (
            f"apply_cp_reindex: global token count {pre_global_len} is not divisible by cp_size={cp_size}. "
            "CP round-robin requires padding to ensure divisibility."
        )
        expected_local_len = pre_global_len // cp_size
        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name, None)
            assert isinstance(
                val, torch.Tensor
            ), f"CP reindex: {field_name} is {type(val)}, expected Tensor"
            setattr(self, field_name, val[idx].contiguous())

        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name)
            assert val.shape[0] == expected_local_len, (
                f"apply_cp_reindex post-condition: {field_name}.shape[0]={val.shape[0]} "
                f"!= expected_local_len={expected_local_len} (cp_size={cp_size})"
            )
        for field_name in self._CP_GLOBAL_FIELDS:
            val = getattr(self, field_name, None)
            if val is None:
                continue
            assert val.shape[0] == pre_global_len, (
                f"apply_cp_reindex post-condition: global field {field_name}.shape[0]={val.shape[0]} "
                f"!= pre_global_len={pre_global_len} (must remain global for compressor write path)"
            )

    def init_flashmla_related(self):
        # c4_sparse_topk is set from model_config.index_topk per-model
        # (small model: 512, large model: 1024).
        assert self.c4_sparse_topk in (512, 1024), (
            f"unexpected c4_sparse_topk={self.c4_sparse_topk}; "
            "supported: 512 (small) or 1024 (large)"
        )
        assert self.c4_topk_lengths_clamp1 is not None
        self.c4_sparse_topk_lengths = torch.clamp(
            self.c4_topk_lengths_clamp1, max=self.c4_sparse_topk
        )
        self.c4_sparse_page_indices = torch.full(
            (self.c4_topk_lengths_clamp1.size(0), self.c4_sparse_topk),
            -1,
            dtype=torch.int32,
            device=self.c4_topk_lengths_clamp1.device,
        )
        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)
        self.c1_flashmla_metadata = _create_flashmla_metadata()
        self.c4_flashmla_metadata = _create_flashmla_metadata()
        self.c128_flashmla_metadata = _create_flashmla_metadata()


@dataclass
class DSV4Metadata:
    core_attn_metadata: DSV4AttnMetadata
    indexer_metadata: Optional[PagedIndexerMetadata]

    c4_compress_metadata: Optional[FusedCompressMetadata] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    @property
    def core_metadata(self) -> DSV4AttnMetadata:
        return self.core_attn_metadata

    def copy_(self, other: DSV4Metadata):
        self.core_attn_metadata.copy_(other.core_attn_metadata)
        maybe_copy_inplace(self.indexer_metadata, src=other.indexer_metadata)
        maybe_copy_inplace(self.c4_compress_metadata, src=other.c4_compress_metadata)
        maybe_copy_inplace(
            self.c128_compress_metadata, src=other.c128_compress_metadata
        )


@dataclass
class DSV4RawVerifyMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    extend_seq_lens: Optional[torch.Tensor] = None

    def copy_(self, other: DSV4RawVerifyMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)

        self.extend_seq_lens = other.extend_seq_lens


@dataclass
class DSV4RawDecodeMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    def copy_(self, other: DSV4RawDecodeMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)


class _GraphBucket(enum.Enum):
    DECODE_OR_IDLE = "decode_or_idle"
    TARGET_VERIFY = "target_verify"
    DRAFT_EXTEND = "draft_extend"

    @classmethod
    def of(cls, forward_mode: ForwardMode) -> _GraphBucket:
        if forward_mode.is_decode_or_idle():
            return cls.DECODE_OR_IDLE
        if forward_mode.is_target_verify():
            return cls.TARGET_VERIFY
        if forward_mode.is_draft_extend(include_v2=True):
            return cls.DRAFT_EXTEND
        raise NotImplementedError(f"unsupported {forward_mode=}")


class DeepseekV4AttnBackend(
    AttentionBackend, C4IndexerBackendMixin, CompressorBackendMixin
):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.device = torch.device(model_runner.device)
        head_dim = model_runner.model_config.head_dim
        assert (
            head_dim == 512
        ), "DSV4 MQA head_dim = qk_nope_head_dim(448) + qk_rope_head_dim(64) = 512"
        self.softmax_scale: float = head_dim**-0.5
        self.head_dim_v: int = model_runner.model_config.v_head_dim
        self.cuda_int32_kwargs = {"device": self.device, "dtype": torch.int32}
        self.swa_page_size = 128
        assert model_runner.page_size is not None
        assert model_runner.req_to_token_pool is not None
        self.page_size = model_runner.page_size
        assert self.page_size == 256, "the system hardcodes page_size=256"

        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool: DeepSeekV4TokenToKVPool = model_runner.token_to_kv_pool
        self.MAX_SEQ_LEN_FOR_CAPTURE = self.req_to_token.shape[1]

        assert isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
        self.c4_topk = getattr(
            model_runner.model_config.hf_text_config, "index_topk", C4_TOPK
        )

        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        assert self.topk in [0, 1], "MTP Topk > 1 not supported for DeepSeek V4"
        self.mtp_enabled = self.topk > 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens: int = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id
        self.forward_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ] = None
        self._replay_forward_batch: Optional[ForwardBatch] = None  # FIXME: out-of-band

    def _move_to_device(self, x: List[int]) -> torch.Tensor:
        pin_tensor = torch.tensor(x, dtype=torch.int32, pin_memory=True)
        return pin_tensor.to(self.device, non_blocking=True)

    def init_forward_metadata_indexer(self, core_attn_metadata: DSV4AttnMetadata):
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=core_attn_metadata.page_table,
            c4_seq_lens=core_attn_metadata.c4_topk_lengths_raw,
        )

    def init_forward_metadata_decode(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> Union[DSV4Metadata, DSV4RawDecodeMetadata]:
        assert (
            req_pool_indices.shape[0] == seq_lens.shape[0] == out_cache_loc.shape[0]
        ), f"{req_pool_indices.shape=} {seq_lens.shape=} {out_cache_loc.shape=}"

        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            return DSV4RawDecodeMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
            )

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=True,
        )

        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_prefill(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        out_cache_loc: torch.Tensor,
        num_tokens: int,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: List[int],
        need_compress: bool = True,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        seq_lens_casual, req_pool_indices_repeated = self.expand_prefill_casually(
            num_tokens=num_tokens,
            seq_lens=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            padded_num_tokens=out_cache_loc.shape[0],
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=need_compress,
            is_prefill=True,
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if need_compress
            else None
        )
        if not need_compress:
            create = _create_dummy_paged_compress_data
        else:
            create = functools.partial(
                create_paged_compressor_data,
                is_prefill=True,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                extend_lens=extend_seq_lens,
                extend_lens_cpu=extend_seq_lens_cpu,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_target_verify(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> Union[DSV4Metadata, DSV4RawVerifyMetadata]:
        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            assert out_cache_loc is not None
            if not hasattr(self, "extend_seq_lens_buffer"):
                self.extend_seq_lens_buffer = torch.tensor(
                    [self.speculative_num_draft_tokens] * 1025, device=self.device
                )
            extend_seq_lens = self.extend_seq_lens_buffer[: len(seq_lens)]

            return DSV4RawVerifyMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                extend_seq_lens=extend_seq_lens,
            )
        else:
            seq_lens_cpu = seq_lens.tolist()
            return self.init_forward_metadata_target_verify_old(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )

    def init_forward_metadata_target_verify_old(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[List[int]] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        seq_lens = seq_lens + self.speculative_num_draft_tokens
        seq_lens_cpu = [x + self.speculative_num_draft_tokens for x in seq_lens_cpu]
        extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = self.speculative_num_draft_tokens * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=True,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def make_forward_metadata_from_raw_verify(
        self, raw_metadata: DSV4RawVerifyMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        bs, num_draft_tokens = len(seq_lens), self.speculative_num_draft_tokens
        seq_lens = seq_lens + self.speculative_num_draft_tokens
        extend_seq_lens = raw_metadata.extend_seq_lens

        seq_lens_casual, req_pool_indices_repeated = (
            self.expand_extend_with_same_length(
                bs, num_draft_tokens, seq_lens, req_pool_indices
            )
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)
        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=True,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_lens=extend_seq_lens,
            seq_lens_cpu=None,
            extend_lens_cpu=None,
            use_prefill_cuda_graph=True,
            num_q_tokens=num_draft_tokens * bs,
        )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def make_forward_metadata_from_raw_decode(
        self, raw_metadata: DSV4RawDecodeMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_draft_extend(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        num_tokens_per_bs: int,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        extend_seq_lens_cpu = [num_tokens_per_bs] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = num_tokens_per_bs * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=False,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert forward_batch.req_to_token_pool.req_to_token is self.req_to_token

        assert self.swa_page_size % SWA_WINDOW == 0 and self.page_size % 128 == 0
        assert seq_lens_cpu is not None
        max_seq_len = int(seq_lens_cpu.max().item())

        if forward_batch.forward_mode.is_decode_or_idle():
            metadata = self.init_forward_metadata_decode(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
            )
        elif forward_batch.forward_mode.is_target_verify():
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
            )
        elif forward_batch.forward_mode.is_prefill(include_draft_extend_v2=True):
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            extend_seq_lens = forward_batch.extend_seq_lens
            assert (
                seq_lens is not None
                and seq_lens_cpu is not None
                and extend_seq_lens is not None
                and extend_seq_lens_cpu is not None
            )
            is_draft = forward_batch.forward_mode.is_draft_extend(include_v2=True)
            metadata = self.init_forward_metadata_prefill(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                out_cache_loc=forward_batch.out_cache_loc,
                num_tokens=sum(extend_seq_lens_cpu),
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                need_compress=not is_draft,
            )
        else:
            raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")

        self.forward_metadata = metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        self.cuda_graph_metadata_of_bucket_and_bs: Dict[
            _GraphBucket,
            Dict[
                int,
                Union[DSV4Metadata, DSV4RawDecodeMetadata, DSV4RawVerifyMetadata],
            ],
        ] = {bucket: {} for bucket in _GraphBucket}
        self.draft_extend_num_tokens_per_bs = (
            max_num_tokens // max_bs if max_bs > 0 else 1
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        assert req_pool_indices.size(0) == bs
        assert seq_lens.size(0) == bs

        bucket = _GraphBucket.of(forward_mode)
        raw_type: Optional[type] = None
        if bucket == _GraphBucket.DECODE_OR_IDLE:
            metadata = self.init_forward_metadata_decode(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=torch.zeros_like(seq_lens),
            )
            raw_type = DSV4RawDecodeMetadata
        elif bucket == _GraphBucket.TARGET_VERIFY:
            out_cache_loc = torch.zeros(num_tokens, **self.cuda_int32_kwargs)
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=True,
            )
            raw_type = DSV4RawVerifyMetadata
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            num_tokens_per_bs = num_tokens // bs
            metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
        else:
            raise NotImplementedError(f"{forward_mode=} not supported yet")

        self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs] = metadata
        self.forward_metadata = metadata
        if raw_type is not None:
            self._current_capture_raw = (
                metadata if isinstance(metadata, raw_type) else None
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> None:
        bucket = _GraphBucket.of(forward_mode)

        # FIXME: see cuda_graph_runner — this attribute is set out-of-band.
        fb = self._replay_forward_batch
        out_cache_loc = fb.out_cache_loc
        actual_forward_mode = fb.forward_mode

        if actual_forward_mode == ForwardMode.IDLE:
            logger.debug(
                f"[IDLE replay] bs={bs}, "
                f"local_seq_lens_len={len(seq_lens)}, "
                f"has_graph={bs in self.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.DECODE_OR_IDLE]}"
            )
            device = seq_lens.device
            seq_lens = torch.ones(bs, dtype=seq_lens.dtype, device=device)
            seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
            seq_lens_sum = bs
            req_pool_indices = torch.zeros(
                bs, dtype=req_pool_indices.dtype, device=device
            )
            out_cache_loc = torch.zeros(bs, dtype=torch.int64, device=device)

        assert seq_lens_cpu is not None
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        actual_max_seq_len = seq_lens_cpu.max().item()
        chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
        assert actual_max_seq_len <= chosen_max_seq_len

        if bucket == _GraphBucket.DECODE_OR_IDLE:
            assert out_cache_loc is not None
            assert len(out_cache_loc.shape) == 1, f"{out_cache_loc.shape=}"
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, bs - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_decode(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
            )
        elif bucket == _GraphBucket.TARGET_VERIFY:
            assert out_cache_loc is not None
            num_tokens = self.speculative_num_draft_tokens * bs
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, num_tokens - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_target_verify(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
                use_prefill_cuda_graph=True,
            )
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            num_tokens_per_bs = self.draft_extend_num_tokens_per_bs
            temp_metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
        else:
            raise NotImplementedError

        self.replay_cuda_graph_metadata_from(
            bs=bs, temp_metadata=temp_metadata, bucket=bucket
        )

    def replay_cuda_graph_metadata_from(
        self,
        bs: int,
        temp_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ],
        bucket: _GraphBucket,
    ) -> None:
        chosen_metadata = self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs]
        chosen_metadata.copy_(temp_metadata)
        self.forward_metadata = chosen_metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def on_after_cuda_graph_warmup(self):
        metadata = self.forward_metadata
        if isinstance(metadata, DSV4Metadata) and isinstance(
            metadata.core_attn_metadata, DSV4AttnMetadata
        ):
            core = metadata.core_attn_metadata
            core.c1_flashmla_metadata = _create_flashmla_metadata()
            core.c4_flashmla_metadata = _create_flashmla_metadata()
            core.c128_flashmla_metadata = _create_flashmla_metadata()

        # PREP_IN_CUDA_GRAPH=True: warmup upgraded raw->full on the host;
        # restore raw so capture re-runs the upgrade inside the graph.
        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        raw_loc = forward_batch.out_cache_loc
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            self.token_to_kv_pool.set_swa_key_buffer_radix_fused(
                layer_id=layer_id,
                raw_loc=raw_loc,
                cache_k=swa_k,
            )
        else:
            swa_k_pack = quant_to_nope_fp8_rope_bf16_pack_triton(swa_k)
            self.token_to_kv_pool.set_swa_key_buffer_radix(
                layer_id=layer_id,
                raw_loc=raw_loc,
                cache_nope_fp8_rope_bf16_pack=swa_k_pack,
            )

    def _maybe_upgrade_forward_metadata(self) -> None:
        # With SGLANG_PREP_IN_CUDA_GRAPH=1, init_forward_metadata_*
        # returns a Raw metadata that only carries a few tensors. The
        # full DSV4Metadata (including c4/c128 compress + core_attn +
        # indexer metadata) must be materialized before any caller that
        # touches those fields. For 1.6T the first two layers have
        # compress_ratio=128, so forward_core_compressor / forward_c4_indexer
        # can fire before attn_backend.forward(), and must trigger the
        # upgrade themselves.
        if isinstance(self.forward_metadata, DSV4RawVerifyMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_verify(
                raw_metadata=self.forward_metadata,
            )
        elif isinstance(self.forward_metadata, DSV4RawDecodeMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_decode(
                raw_metadata=self.forward_metadata,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 4, 128],
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        self._maybe_upgrade_forward_metadata()

        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        assert k is v, "DeepseekV4 shares k and v"
        swa_k = k

        layer_id = layer.layer_id
        metadata = self.forward_metadata
        core_attn_metadata = metadata.core_attn_metadata
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        if isinstance(core_attn_metadata, DSV4AttnMetadata):
            if save_kv_cache:
                self.store_cache(layer_id, swa_k, forward_batch)
            swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

            extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
            if compress_ratio == 4:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c4_sparse_page_indices
                extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths
            elif compress_ratio == 128:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c128_page_indices
                extra_topk_lengths = core_attn_metadata.c128_topk_lengths_clamp1

            swa_window_size = token_to_kv_pool.swa_window_size
            assert swa_k_cache.ndim == 2
            k_cache_total_dim = token_to_kv_pool.swa_kv_pool.kv_cache_total_dim
            swa_k_cache = swa_k_cache[:, : swa_window_size * k_cache_total_dim].view(
                swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
            )

            if extra_k_cache is not None:
                page_sizes = {
                    4: token_to_kv_pool.page_size // 4,
                    128: token_to_kv_pool.page_size // 128,
                }
                extra_k_cache = extra_k_cache[
                    :, : page_sizes[compress_ratio] * k_cache_total_dim
                ].view(
                    extra_k_cache.shape[0],
                    page_sizes[compress_ratio],
                    1,
                    k_cache_total_dim,
                )
            swa_page_indices = core_attn_metadata.swa_page_indices
            swa_topk_lengths = core_attn_metadata.swa_topk_lengths

            if self.mtp_enabled:
                if swa_page_indices.shape[0] != q.shape[0]:
                    swa_page_indices = _pad_tensor_to_size(
                        swa_page_indices, q.shape[0], value=0
                    )

                if swa_topk_lengths.shape[0] != q.shape[0]:
                    swa_topk_lengths = _pad_tensor_to_size(
                        swa_topk_lengths, q.shape[0], value=1
                    )

            if q.ndim == 3:
                q = q.unsqueeze(1)
            if swa_page_indices.ndim == 2:
                swa_page_indices = swa_page_indices.unsqueeze(1)
            if extra_indices is not None and extra_indices.ndim == 2:
                extra_indices = extra_indices.unsqueeze(1)

            assert attn_sink is not None

            flashmla_metadata = core_attn_metadata.get_flashmla_metadata(compress_ratio)

            assert (
                swa_page_indices.shape[-1] % 64 == 0
            ), f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
            if extra_indices is not None:
                assert (
                    extra_indices.shape[-1] % 64 == 0
                ), f"{extra_indices.shape=}'s last dimension is not aligned to 64"

            import flash_mla

            if (
                forward_batch.forward_mode.is_extend_without_speculative()
                and _dsv4_use_bf16_sparse_prefill()
            ):
                prefill_inputs = _dsv4_build_unified_prefill_inputs_from_real_decode(
                    q=q,
                    swa_k_cache=swa_k_cache,
                    extra_k_cache=extra_k_cache,
                    swa_page_indices=swa_page_indices,
                    extra_indices=extra_indices,
                    swa_topk_lengths=swa_topk_lengths,
                    extra_topk_lengths=extra_topk_lengths,
                )
                if prefill_inputs is not None:
                    q_prefill, kv_unified, indices_unified = prefill_inputs
                    o_sparse = _dsv4_run_bf16_sparse_prefill_attention(
                        q_prefill=q_prefill,
                        kv_unified=kv_unified,
                        indices_unified=indices_unified,
                        sm_scale=self.softmax_scale,
                        d_v=self.head_dim_v,
                        attn_sink=attn_sink,
                        compress_ratio=compress_ratio,
                    )
                    if o_sparse is not None:
                        return o_sparse

            o = flash_mla.flash_mla_with_kvcache(
                q=q,
                k_cache=swa_k_cache,
                head_dim_v=self.head_dim_v,
                block_table=None,
                cache_seqlens=None,
                tile_scheduler_metadata=flashmla_metadata,
                softmax_scale=self.softmax_scale,
                is_fp8_kvcache=True,
                indices=swa_page_indices,
                topk_length=swa_topk_lengths,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices,
                extra_topk_length=extra_topk_lengths,
            )[0]

            o = o.squeeze(1)
            return o

        raise NotImplementedError("ragged attention")

    def expand_prefill_casually(
        self,
        num_tokens: int,
        seq_lens: List[int],
        extend_seq_lens: List[int],
        req_pool_indices: torch.Tensor,
        padded_num_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_lens_casual = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        idx_to_req_repeated = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        offset = 0
        for i, (kv_len, qo_len) in enumerate(zip(seq_lens, extend_seq_lens)):
            out = seq_lens_casual[offset : offset + qo_len]
            offset += qo_len
            torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
            idx_to_req_repeated[offset - qo_len : offset].fill_(i)

        assert offset == num_tokens
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]

        if padded_num_tokens is not None and padded_num_tokens > num_tokens:
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                seq_lens_casual,
                (0, pad_size),
                value=1,
            )
            req_pool_indices_repeated = torch.nn.functional.pad(
                req_pool_indices_repeated,
                (0, pad_size),
                value=req_pool_indices_repeated[-1].item(),
            )

        return seq_lens_casual, req_pool_indices_repeated

    def expand_extend_with_same_length(
        self,
        bs: int,
        qo_len: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        seq_lens_casual = seq_lens[:, None] + torch.arange(
            -qo_len + 1, 1, **self.cuda_int32_kwargs
        )
        seq_lens_casual = seq_lens_casual.flatten()
        idx_to_req_repeated = torch.arange(
            bs, **self.cuda_int32_kwargs
        ).repeat_interleave(qo_len)
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]
        return seq_lens_casual, req_pool_indices_repeated

    def make_core_attn_metadata(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        out_loc: torch.Tensor,
        need_compress: bool = True,
        is_prefill: bool = False,
    ) -> DSV4AttnMetadata:
        assert self.swa_page_size == SWA_WINDOW

        swa_page_indices = self.get_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

        swa_page_indices = _pad_last_dim(
            swa_page_indices, multiples_of=PAGE_INDEX_ALIGNED_SIZE
        )

        raw_positions = seq_lens_casual - 1
        swa_topk_lengths = torch.clamp(seq_lens_casual, max=SWA_WINDOW)

        page_table = req_to_token[
            req_pool_indices_repeated, : max_seq_len : self.page_size
        ]
        page_table = (page_table // self.page_size).to(torch.int32)

        core_attn_metadata = DSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=raw_positions,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_sparse_topk=self.c4_topk,
        )

        if need_compress:
            core_attn_metadata.init_compression_metadata()
            core_attn_metadata.init_flashmla_related()
        else:
            core_attn_metadata.c4_sparse_topk_lengths = None
            core_attn_metadata.c4_sparse_page_indices = None
            core_attn_metadata.c1_flashmla_metadata = _create_flashmla_metadata()
            core_attn_metadata.c4_flashmla_metadata = None
            core_attn_metadata.c128_flashmla_metadata = None
        return core_attn_metadata

    def get_swa_page_indices(
        self,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
    ) -> torch.Tensor:
        pos_causal = seq_lens_casual - 1
        num_qo_tokens = seq_lens_casual.size(0)
        offsets = pos_causal.unsqueeze(1) - torch.arange(
            SWA_WINDOW, **self.cuda_int32_kwargs
        ).unsqueeze(0)
        invalid_offset_mask = offsets < 0
        offsets.masked_fill_(invalid_offset_mask, 0)
        raw_indices = self.req_to_token[req_pool_indices_repeated[:, None], offsets]
        assert raw_indices.shape == (num_qo_tokens, SWA_WINDOW)
        raw_indices.masked_fill_(invalid_offset_mask, -1)
        swa_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(raw_indices)
        return swa_indices


class DeepseekV4MultiStepBackend(DeepseekV4AttnBackend):
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner)
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[DeepseekV4AttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                DeepseekV4AttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def on_after_cuda_graph_warmup(self):
        for backend in self.attn_backends:
            backend.on_after_cuda_graph_warmup()

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        if self.speculative_num_steps == 1:
            return

        self.attn_backends[0]._replay_forward_batch = forward_batch
        self.attn_backends[0].init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )
        self.attn_backends[0]._replay_forward_batch = None
        temp_metadata = self.attn_backends[0].forward_metadata

        for i in range(1, self.speculative_num_steps - 1):
            self.attn_backends[i].replay_cuda_graph_metadata_from(
                bs=bs,
                temp_metadata=temp_metadata,
                bucket=_GraphBucket.DECODE_OR_IDLE,
            )


def _pad_tensor_to_size(tensor: torch.Tensor, size: int, *, value: int = 0):
    if value == 0:
        return torch.cat(
            [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:])],
            dim=0,
        )
    else:
        return torch.cat(
            [
                tensor,
                tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value),
            ],
            dim=0,
        )
