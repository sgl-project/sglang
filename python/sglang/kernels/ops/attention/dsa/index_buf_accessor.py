from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.attention.dsa.utils import (
    INDEXER_K_CACHE_PRESHUFFLE_TILE,
    aiter_can_use_preshuffle_paged_mqa,
)
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
# aiter cp_gather kernel with preshuffle=True is only valid when the indexer
# uses the page_size=64 preshuffle layout (i.e. when the matching MQA gluon path
# is also enabled).
_use_aiter_preshuffle = aiter_can_use_preshuffle_paged_mqa()

if _use_aiter_preshuffle:
    from aiter.ops.cache import cp_gather_indexer_k_quant_cache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

"""
k: data, 128 item per token, fp8
s: scale, 1 item per token, fp32
"""


class GetK:
    @classmethod
    def execute(cls, *args, **kwargs):
        return cls.triton(*args, **kwargs)

    @classmethod
    def slow(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        num_pages = (seq_len + pool.page_size - 1) // pool.page_size
        seq_len_ = num_pages * pool.page_size
        index_k_fp8 = torch.empty(
            (seq_len_, pool.index_head_dim),
            dtype=torch.uint8,
            device=pool.device,
        )
        for i in range(num_pages):
            page_index = page_indices[i]
            index_k_fp8[i * pool.page_size : (i + 1) * pool.page_size] = buf[
                page_index
            ][: pool.page_size * pool.index_head_dim].view(-1, pool.index_head_dim)

        return index_k_fp8[:seq_len]

    @classmethod
    def torch_fast(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        :param page_indices: (num_pages,), int32
        :return: (seq_len, index_head_dim), uint8
        """

        # can handle per 128B instead of per element

        # page_indices: (num_pages,), element := a page index
        buf_numel_per_page = buf.shape[1]

        num_k_bytes_per_page = pool.page_size * pool.index_head_dim
        num_k_bytes_per_token = pool.index_head_dim

        # buf: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4), uint8
        # flat_buf: (whatever,), uint8
        flat_buf = buf.flatten()

        # flat_indices: (num_pages, num_k_bytes_per_page), int32, element := an index into flat_buf that we want to access
        flat_indices = (page_indices * buf_numel_per_page)[:, None] + torch.arange(
            num_k_bytes_per_page, dtype=torch.int32, device="cuda"
        )[None, :]
        flat_indices = flat_indices.flatten()[: seq_len * num_k_bytes_per_token]

        out = flat_buf[flat_indices]
        return out.view(-1, 128)

    @classmethod
    def triton(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        Triton implementation for gathering K data from paged buffer.
        :param page_indices: (num_pages,), int32/int64
        :return: (seq_len, index_head_dim), uint8
        """
        return _get_k_triton(
            buf=buf,
            page_indices=page_indices,
            seq_len=seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )


class GetS:
    @classmethod
    def execute(cls, *args, **kwargs):
        return cls.triton(*args, **kwargs)

    @classmethod
    def slow(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        num_pages = (seq_len + pool.page_size - 1) // pool.page_size
        seq_len_ = num_pages * pool.page_size
        assert pool.index_head_dim // pool.quant_block_size == 1
        index_k_scale_fp8 = torch.empty(
            (seq_len_, 4),
            dtype=torch.uint8,
            device=pool.device,
        )
        for i in range(num_pages):
            page_index = page_indices[i]
            index_k_scale_fp8[i * pool.page_size : (i + 1) * pool.page_size] = buf[
                page_index
            ][pool.page_size * pool.index_head_dim :].view(-1, 4)
        return index_k_scale_fp8[:seq_len]

    @classmethod
    def torch_fast(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        :param page_indices: (num_pages,), int32
        :return: (seq_len, index_head_dim // quant_block_size), uint8
        """
        buf_numel_per_page = buf.shape[1]

        num_s_bytes_per_page = buf.shape[1] - pool.page_size * pool.index_head_dim
        num_s_bytes_per_token = pool.index_head_dim // pool.quant_block_size * 4
        s_offset_in_page = pool.page_size * pool.index_head_dim

        flat_buf = buf.flatten()
        flat_indices = (
            (page_indices * buf_numel_per_page)[:, None]
            + torch.arange(num_s_bytes_per_page, dtype=torch.int32, device="cuda")[
                None, :
            ]
            + s_offset_in_page
        )
        flat_indices = flat_indices.flatten()[: seq_len * num_s_bytes_per_token]

        out = flat_buf[flat_indices]
        return out.view(-1, 4)

    @classmethod
    def triton(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        Triton implementation for gathering S (scale) data from paged buffer.
        :param page_indices: (num_pages,), int32/int64
        :return: (seq_len, 4), uint8
        """
        return _get_s_triton(
            buf=buf,
            page_indices=page_indices,
            seq_len=seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )


class GetKAndS:
    @classmethod
    def execute(cls, *args, **kwargs):
        # The aiter path uses cp_gather_indexer_k_quant_cache(preshuffle=True),
        # which only matches the layout produced when the rest of the indexer
        # is on the page_size=64 preshuffle path. Otherwise fall back to the
        # triton implementation (which works on the page_size=1 legacy layout).
        if _use_aiter_preshuffle:
            return cls.aiter(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def aiter(
        cls,
        pool: "DSATokenToKVPool",
        buf: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_tensor: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

        page_size = pool.page_size
        index_head_dim = pool.index_head_dim
        quant_block_size = pool.quant_block_size
        scale_elems = index_head_dim // quant_block_size

        kv_cache = buf.view(-1, page_size, index_head_dim + scale_elems * 4).view(
            fp8_dtype
        )
        dst_k = torch.empty(
            (seq_len_sum, index_head_dim), dtype=torch.uint8, device=buf.device
        )
        dst_scale = torch.empty(
            (seq_len_sum, scale_elems * 4), dtype=torch.uint8, device=buf.device
        )

        cu_seq_lens = torch.zeros(
            seq_len_tensor.shape[0] + 1, dtype=torch.int32, device=buf.device
        )
        torch.cumsum(seq_len_tensor.to(torch.int32), dim=0, out=cu_seq_lens[1:])

        cp_gather_indexer_k_quant_cache(
            kv_cache,
            dst_k.view(fp8_dtype),
            dst_scale,
            page_indices.to(torch.int32),
            cu_seq_lens,
            preshuffle=True,
        )
        return dst_k, dst_scale

    @classmethod
    def triton(
        cls,
        pool: "DSATokenToKVPool",
        buf: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_tensor: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        """
        Triton implementation for gathering both K and S data from paged buffer in a single call.
        :param page_indices: (num_pages,), int32/int64
        :param seq_len_tensor: (num_pages,), int32/int64
        :param seq_len_sum: sum of all sequence len, int32
        :param max_seq_len: max of all sequence len, int32
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        return _get_k_and_s_triton(
            buf=buf,
            page_indices=page_indices,
            seq_lens=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )


class SetKAndS:
    @classmethod
    def execute(cls, *args, buf, **kwargs):
        cls.triton(*args, **kwargs, buf=buf)

    @classmethod
    def triton(cls, pool, buf, loc, index_k, index_k_scale):
        loc = loc.to(torch.int64)

        _set_k_and_s_triton(
            buf=buf,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
            page_size=pool.page_size,
        )


class MoveDSACache:
    """Relocate latent KV and paged index K/scale across all local layers."""

    @classmethod
    def execute(
        cls,
        pool,
        kv_ptrs,
        index_ptrs,
        tgt_loc,
        src_loc,
        kv_scratch,
        index_scratch,
        scratch_capacity,
    ):
        assert tgt_loc.numel() == src_loc.numel()
        if tgt_loc.numel() == 0:
            return

        num_layers = kv_ptrs.numel()
        assert num_layers == index_ptrs.numel()
        assert num_layers > 0
        assert kv_ptrs.dtype == index_ptrs.dtype == torch.uint64
        assert kv_ptrs.device == index_ptrs.device == tgt_loc.device
        assert tgt_loc.device == src_loc.device
        assert tgt_loc.dtype in (torch.int32, torch.int64)
        assert src_loc.dtype in (torch.int32, torch.int64)
        assert kv_scratch.dtype == index_scratch.dtype == torch.uint8
        assert kv_scratch.shape[0] == index_scratch.shape[0] == num_layers
        assert kv_scratch.shape[1] >= tgt_loc.numel()
        assert index_scratch.shape[1] >= tgt_loc.numel()

        tgt_loc = tgt_loc.reshape(-1).to(torch.int64).contiguous()
        src_loc = src_loc.reshape(-1).to(torch.int64).contiguous()
        num_tokens = src_loc.numel()
        kv_row_bytes = kv_scratch.shape[2]
        block_kv = triton.next_power_of_2(kv_row_bytes)
        preshuffle_tile = (
            INDEXER_K_CACHE_PRESHUFFLE_TILE if _use_aiter_preshuffle else 0
        )
        grid = (num_layers, num_tokens)
        _gather_dsa_cache_by_loc_kernel[grid](
            kv_ptrs,
            index_ptrs,
            src_loc,
            kv_scratch,
            index_scratch,
            scratch_capacity,
            PAGE_SIZE=pool.page_size,
            INDEX_BUF_NUMEL_PER_PAGE=pool.index_k_with_scale_buffer[0].shape[1],
            NUM_K_ELEMS_PER_TOKEN=pool.index_head_dim,
            KV_ROW_BYTES=kv_row_bytes,
            BLOCK_KV=block_kv,
            PRESHUFFLE_TILE=preshuffle_tile,
            HAS_KV=True,
        )
        _scatter_dsa_cache_by_loc_kernel[grid](
            kv_ptrs,
            index_ptrs,
            tgt_loc,
            kv_scratch,
            index_scratch,
            scratch_capacity,
            PAGE_SIZE=pool.page_size,
            INDEX_BUF_NUMEL_PER_PAGE=pool.index_k_with_scale_buffer[0].shape[1],
            NUM_K_ELEMS_PER_TOKEN=pool.index_head_dim,
            KV_ROW_BYTES=kv_row_bytes,
            BLOCK_KV=block_kv,
            PRESHUFFLE_TILE=preshuffle_tile,
            HAS_KV=True,
        )


@triton.jit
def _gather_dsa_cache_by_loc_kernel(
    kv_ptrs,
    index_ptrs,
    src_loc_ptr,
    kv_scratch_ptr,
    index_scratch_ptr,
    scratch_capacity,
    PAGE_SIZE: tl.constexpr,
    INDEX_BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    KV_ROW_BYTES: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
    HAS_KV: tl.constexpr,
):
    layer_id = tl.program_id(0)
    token_id = tl.program_id(1)
    loc = tl.load(src_loc_ptr + token_id)
    layer_token = layer_id * scratch_capacity + token_id

    if HAS_KV:
        kv_ptr = tl.load(kv_ptrs + layer_id).to(tl.pointer_type(tl.uint8))
        kv_range = tl.arange(0, BLOCK_KV)
        kv_mask = kv_range < KV_ROW_BYTES
        kv = tl.load(kv_ptr + loc * KV_ROW_BYTES + kv_range, mask=kv_mask)
        tl.store(
            kv_scratch_ptr + layer_token * KV_ROW_BYTES + kv_range,
            kv,
            mask=kv_mask,
        )

    index_ptr = tl.load(index_ptrs + layer_id).to(tl.pointer_type(tl.uint8))
    page = loc // PAGE_SIZE
    token_in_page = loc % PAGE_SIZE
    k_range = tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    if PRESHUFFLE_TILE:
        tile = PRESHUFFLE_TILE
        token_tile = token_in_page // tile
        token_in_tile = token_in_page % tile
        col_tile = k_range // tile
        col_in_tile = k_range % tile
        k_offsets = (
            page * INDEX_BUF_NUMEL_PER_PAGE
            + token_tile * (tile * NUM_K_ELEMS_PER_TOKEN)
            + col_tile * (tile * tile)
            + token_in_tile * tile
            + col_in_tile
        )
    else:
        k_offsets = (
            page * INDEX_BUF_NUMEL_PER_PAGE
            + token_in_page * NUM_K_ELEMS_PER_TOKEN
            + k_range
        )
    index_payload_bytes: tl.constexpr = NUM_K_ELEMS_PER_TOKEN + 4
    index_payload_base = layer_token * index_payload_bytes
    tl.store(
        index_scratch_ptr + index_payload_base + k_range,
        tl.load(index_ptr + k_offsets),
    )

    scale_range = tl.arange(0, 4)
    scale_offsets = (
        page * INDEX_BUF_NUMEL_PER_PAGE
        + PAGE_SIZE * NUM_K_ELEMS_PER_TOKEN
        + token_in_page * 4
        + scale_range
    )
    tl.store(
        index_scratch_ptr + index_payload_base + NUM_K_ELEMS_PER_TOKEN + scale_range,
        tl.load(index_ptr + scale_offsets),
    )


@triton.jit
def _scatter_dsa_cache_by_loc_kernel(
    kv_ptrs,
    index_ptrs,
    tgt_loc_ptr,
    kv_scratch_ptr,
    index_scratch_ptr,
    scratch_capacity,
    PAGE_SIZE: tl.constexpr,
    INDEX_BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    KV_ROW_BYTES: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
    HAS_KV: tl.constexpr,
):
    layer_id = tl.program_id(0)
    token_id = tl.program_id(1)
    loc = tl.load(tgt_loc_ptr + token_id)
    layer_token = layer_id * scratch_capacity + token_id

    if HAS_KV:
        kv_ptr = tl.load(kv_ptrs + layer_id).to(tl.pointer_type(tl.uint8))
        kv_range = tl.arange(0, BLOCK_KV)
        kv_mask = kv_range < KV_ROW_BYTES
        kv = tl.load(
            kv_scratch_ptr + layer_token * KV_ROW_BYTES + kv_range,
            mask=kv_mask,
        )
        tl.store(kv_ptr + loc * KV_ROW_BYTES + kv_range, kv, mask=kv_mask)

    index_ptr = tl.load(index_ptrs + layer_id).to(tl.pointer_type(tl.uint8))
    page = loc // PAGE_SIZE
    token_in_page = loc % PAGE_SIZE
    k_range = tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    if PRESHUFFLE_TILE:
        tile = PRESHUFFLE_TILE
        token_tile = token_in_page // tile
        token_in_tile = token_in_page % tile
        col_tile = k_range // tile
        col_in_tile = k_range % tile
        k_offsets = (
            page * INDEX_BUF_NUMEL_PER_PAGE
            + token_tile * (tile * NUM_K_ELEMS_PER_TOKEN)
            + col_tile * (tile * tile)
            + token_in_tile * tile
            + col_in_tile
        )
    else:
        k_offsets = (
            page * INDEX_BUF_NUMEL_PER_PAGE
            + token_in_page * NUM_K_ELEMS_PER_TOKEN
            + k_range
        )
    index_payload_bytes: tl.constexpr = NUM_K_ELEMS_PER_TOKEN + 4
    index_payload_base = layer_token * index_payload_bytes
    tl.store(
        index_ptr + k_offsets,
        tl.load(index_scratch_ptr + index_payload_base + k_range),
    )

    scale_range = tl.arange(0, 4)
    scale_offsets = (
        page * INDEX_BUF_NUMEL_PER_PAGE
        + PAGE_SIZE * NUM_K_ELEMS_PER_TOKEN
        + token_in_page * 4
        + scale_range
    )
    tl.store(
        index_ptr + scale_offsets,
        tl.load(
            index_scratch_ptr + index_payload_base + NUM_K_ELEMS_PER_TOKEN + scale_range
        ),
    )


class MoveKAndS:
    """Move logical DSA index-cache entries between physical token slots.

    ``buf`` is page-major and stores all K bytes before all scale bytes in each
    page.  On the ROCm AITer path, K is additionally preshuffled in 16x16 tiles.
    The slot arrays must have equal length and unique destinations.  Source and
    destination sets may overlap: gathering into ``scratch`` before scattering
    preserves assignment semantics.
    """

    @classmethod
    def execute(cls, pool, buf, tgt_loc, src_loc, scratch=None):
        assert tgt_loc.numel() == src_loc.numel()
        if tgt_loc.numel() == 0:
            return

        assert tgt_loc.device == src_loc.device == buf.device
        assert tgt_loc.dtype in (torch.int32, torch.int64)
        assert src_loc.dtype in (torch.int32, torch.int64)
        assert buf.dtype == torch.uint8
        assert buf.ndim == 2
        assert buf.shape[1] == pool.page_size * (
            pool.index_head_dim + pool.index_head_dim // pool.quant_block_size * 4
        )

        tgt_loc = tgt_loc.reshape(-1).to(torch.int64).contiguous()
        src_loc = src_loc.reshape(-1).to(torch.int64).contiguous()
        num_tokens = src_loc.numel()
        payload_bytes = pool.index_head_dim + 4
        if scratch is None:
            scratch = torch.empty(
                (num_tokens, payload_bytes), dtype=torch.uint8, device=buf.device
            )
        else:
            assert scratch.shape == (num_tokens, payload_bytes)
            assert scratch.dtype == torch.uint8
            assert scratch.device == buf.device
            assert scratch.is_contiguous()

        preshuffle_tile = (
            INDEXER_K_CACHE_PRESHUFFLE_TILE if _use_aiter_preshuffle else 0
        )
        _gather_k_and_s_by_loc_kernel[(num_tokens,)](
            buf,
            src_loc,
            scratch,
            PAGE_SIZE=pool.page_size,
            BUF_NUMEL_PER_PAGE=buf.shape[1],
            NUM_K_ELEMS_PER_TOKEN=pool.index_head_dim,
            PRESHUFFLE_TILE=preshuffle_tile,
        )
        _scatter_k_and_s_by_loc_kernel[(num_tokens,)](
            buf,
            tgt_loc,
            scratch,
            PAGE_SIZE=pool.page_size,
            BUF_NUMEL_PER_PAGE=buf.shape[1],
            NUM_K_ELEMS_PER_TOKEN=pool.index_head_dim,
            PRESHUFFLE_TILE=preshuffle_tile,
        )


@triton.jit
def _gather_k_and_s_by_loc_kernel(
    buf_ptr,
    src_loc_ptr,
    scratch_ptr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
):
    token_id = tl.program_id(0)
    loc = tl.load(src_loc_ptr + token_id)
    page = loc // PAGE_SIZE
    token_in_page = loc % PAGE_SIZE

    k_range = tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    if PRESHUFFLE_TILE:
        tile = PRESHUFFLE_TILE
        token_tile = token_in_page // tile
        token_in_tile = token_in_page % tile
        col_tile = k_range // tile
        col_in_tile = k_range % tile
        k_offsets = (
            page * BUF_NUMEL_PER_PAGE
            + token_tile * (tile * NUM_K_ELEMS_PER_TOKEN)
            + col_tile * (tile * tile)
            + token_in_tile * tile
            + col_in_tile
        )
    else:
        k_offsets = (
            page * BUF_NUMEL_PER_PAGE + token_in_page * NUM_K_ELEMS_PER_TOKEN + k_range
        )
    payload_base = token_id * (NUM_K_ELEMS_PER_TOKEN + 4)
    tl.store(scratch_ptr + payload_base + k_range, tl.load(buf_ptr + k_offsets))

    scale_range = tl.arange(0, 4)
    scale_offsets = (
        page * BUF_NUMEL_PER_PAGE
        + PAGE_SIZE * NUM_K_ELEMS_PER_TOKEN
        + token_in_page * 4
        + scale_range
    )
    tl.store(
        scratch_ptr + payload_base + NUM_K_ELEMS_PER_TOKEN + scale_range,
        tl.load(buf_ptr + scale_offsets),
    )


@triton.jit
def _scatter_k_and_s_by_loc_kernel(
    buf_ptr,
    tgt_loc_ptr,
    scratch_ptr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
):
    token_id = tl.program_id(0)
    loc = tl.load(tgt_loc_ptr + token_id)
    page = loc // PAGE_SIZE
    token_in_page = loc % PAGE_SIZE

    k_range = tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    if PRESHUFFLE_TILE:
        tile = PRESHUFFLE_TILE
        token_tile = token_in_page // tile
        token_in_tile = token_in_page % tile
        col_tile = k_range // tile
        col_in_tile = k_range % tile
        k_offsets = (
            page * BUF_NUMEL_PER_PAGE
            + token_tile * (tile * NUM_K_ELEMS_PER_TOKEN)
            + col_tile * (tile * tile)
            + token_in_tile * tile
            + col_in_tile
        )
    else:
        k_offsets = (
            page * BUF_NUMEL_PER_PAGE + token_in_page * NUM_K_ELEMS_PER_TOKEN + k_range
        )
    payload_base = token_id * (NUM_K_ELEMS_PER_TOKEN + 4)
    tl.store(k_offsets + buf_ptr, tl.load(scratch_ptr + payload_base + k_range))

    scale_range = tl.arange(0, 4)
    scale_offsets = (
        page * BUF_NUMEL_PER_PAGE
        + PAGE_SIZE * NUM_K_ELEMS_PER_TOKEN
        + token_in_page * 4
        + scale_range
    )
    tl.store(
        buf_ptr + scale_offsets,
        tl.load(scratch_ptr + payload_base + NUM_K_ELEMS_PER_TOKEN + scale_range),
    )


def _set_k_and_s_triton(
    buf: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    page_size: int,
):
    """
    :param buf: (num_pages, page_size 64 * (128B data + 4B scale)), uint8
    :param loc: (num_tokens_to_write,), int, element := the token index to write to
    :param index_k: (num_tokens_to_write, 128 elem), fp8
    :param index_k_scale: (num_tokens_to_write, 1 elem), fp32
    :return:
    """
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape
    num_tokens_to_write_, index_head_dim = index_k.shape

    # Handle both 1D (num_tokens,) and 2D (num_tokens, 1) shapes for index_k_scale
    if index_k_scale.ndim == 1:
        num_tokens_to_write__ = index_k_scale.shape[0]
        scale_dim = 1
    elif index_k_scale.ndim == 2:
        num_tokens_to_write__, scale_dim = index_k_scale.shape
    else:
        raise ValueError(
            f"index_k_scale must be 1D or 2D, got shape {index_k_scale.shape}"
        )
    assert buf_numel_per_page == page_size * (128 + 4)
    assert num_tokens_to_write == num_tokens_to_write_ == num_tokens_to_write__
    assert index_head_dim == 128
    assert scale_dim == 1
    if _is_hip:
        if _use_aiter_preshuffle:
            assert (
                page_size % 16 == 0
            ), f"HIP preshuffle requires page_size to be a multiple of 16, got {page_size}"
    else:
        assert page_size == 64

    assert buf.dtype == torch.uint8
    assert loc.dtype == torch.int64, f"{loc.dtype=}"  # can be int32
    if _is_fp8_fnuz:
        assert index_k.dtype == torch.float8_e4m3fnuz
    else:
        assert index_k.dtype == torch.float8_e4m3fn
    assert index_k_scale.dtype == torch.float32

    assert buf.is_contiguous()
    assert loc.is_contiguous()
    assert index_k.is_contiguous()
    assert index_k_scale.is_contiguous()

    if _is_fp8_fnuz:
        buf_fp8 = buf.view(torch.float8_e4m3fnuz)
    else:
        buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)

    _set_k_and_s_triton_kernel[(num_tokens_to_write,)](
        buf_fp8,
        buf_fp32,
        loc,
        index_k,
        index_k_scale,
        index_k.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        NUM_K_ELEMS_PER_TOKEN=index_head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_head_dim,
        PRESHUFFLE_TILE=INDEXER_K_CACHE_PRESHUFFLE_TILE if _use_aiter_preshuffle else 0,
    )


@triton.jit
def _set_k_and_s_triton_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    loc_ptr,
    index_k_ptr,
    index_k_scale_ptr,
    index_k_ptr_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
):
    token_id = tl.program_id(0)

    loc = tl.load(loc_ptr + token_id)

    in_k_offsets = token_id * index_k_ptr_stride_0 + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)

    # no need for `mask`, since we read 128B for k and 4B for scale, both pow of 2
    k = tl.load(index_k_ptr + in_k_offsets)
    k_scale = tl.load(index_k_scale_ptr + token_id)

    loc_page_index = loc // PAGE_SIZE
    loc_token_offset_in_page = loc % PAGE_SIZE

    k_range = tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    if PRESHUFFLE_TILE:
        tile = PRESHUFFLE_TILE
        token_tile_id = loc_token_offset_in_page // tile
        token_in_tile = loc_token_offset_in_page % tile
        col_tile_id = k_range // tile
        col_in_tile = k_range % tile
        out_k_offsets = (
            loc_page_index * BUF_NUMEL_PER_PAGE
            + token_tile_id * (tile * NUM_K_ELEMS_PER_TOKEN)
            + col_tile_id * (tile * tile)
            + token_in_tile * tile
            + col_in_tile
        )
    else:
        out_k_offsets = (
            loc_page_index * BUF_NUMEL_PER_PAGE
            + loc_token_offset_in_page * NUM_K_ELEMS_PER_TOKEN
            + k_range
        )

    # "//4" b/c it is fp32 instead of uint8
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )

    tl.store(buf_fp8_ptr + out_k_offsets, k)
    tl.store(buf_fp32_ptr + out_s_offset, k_scale)


def _get_k_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    page_size: int,
    index_head_dim: int,
):
    """
    Gather K (key) data from paged buffer using Triton.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_len: int, number of tokens to gather
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: (seq_len, index_head_dim), uint8
    """
    num_pages, buf_numel_per_page = buf.shape

    # Allocate output
    out = torch.empty((seq_len, index_head_dim), dtype=torch.uint8, device=buf.device)

    # Launch kernel with one thread per token
    grid = (seq_len,)
    _get_k_triton_kernel[grid](
        buf,
        page_indices,
        out,
        seq_len,
        page_size,
        buf_numel_per_page,
        index_head_dim,
        BLOCK_SIZE=128,
    )

    return out


@triton.jit
def _get_k_triton_kernel(
    buf_ptr,
    page_indices_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    index_head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one token (seq_len tokens total).
    Loads 128 bytes from the appropriate page.
    """
    token_id = tl.program_id(0)

    # Calculate which page and offset within page
    page_idx = token_id // page_size
    token_offset_in_page = token_id % page_size

    # Load the page index from page_indices
    page_index = tl.load(page_indices_ptr + page_idx)

    # Calculate source offset in buf
    # buf[page_index, token_offset_in_page * index_head_dim : ...]
    src_base_offset = (
        page_index * buf_numel_per_page + token_offset_in_page * index_head_dim
    )

    # Load 128 bytes (index_head_dim elements)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < index_head_dim
    data = tl.load(buf_ptr + src_base_offset + offsets, mask=mask)

    # Store to output
    dst_offset = token_id * index_head_dim
    tl.store(out_ptr + dst_offset + offsets, data, mask=mask)


def _get_s_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    page_size: int,
    index_head_dim: int,
):
    """
    Gather S (scale) data from paged buffer using Triton.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_len: int, number of tokens to gather
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: (seq_len, 4), uint8 (representing fp32 scale)
    """
    num_pages, buf_numel_per_page = buf.shape
    s_offset_in_page = page_size * index_head_dim  # Scales start after K data

    # Allocate output
    out = torch.empty((seq_len, 4), dtype=torch.uint8, device=buf.device)

    # Launch kernel with one thread per token
    grid = (seq_len,)
    _get_s_triton_kernel[grid](
        buf,
        page_indices,
        out,
        seq_len,
        page_size,
        buf_numel_per_page,
        s_offset_in_page,
    )

    return out


@triton.jit
def _get_s_triton_kernel(
    buf_ptr,
    page_indices_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    s_offset_in_page: tl.constexpr,
):
    """
    Each program handles one token (seq_len tokens total).
    Loads 4 bytes (fp32 scale) from the appropriate page.
    """
    token_id = tl.program_id(0)

    # Calculate which page and offset within page
    page_idx = token_id // page_size
    token_offset_in_page = token_id % page_size

    # Load the page index from page_indices
    page_index = tl.load(page_indices_ptr + page_idx)

    # Calculate source offset in buf
    # Scales are stored after K data: page_size * index_head_dim offset
    # buf[page_index, s_offset_in_page + token_offset_in_page * 4 : ...]
    src_base_offset = (
        page_index * buf_numel_per_page + s_offset_in_page + token_offset_in_page * 4
    )

    # Load 4 bytes (fp32 scale)
    offsets = tl.arange(0, 4)
    data = tl.load(buf_ptr + src_base_offset + offsets)

    # Store to output
    dst_offset = token_id * 4
    tl.store(out_ptr + dst_offset + offsets, data)


def _get_k_and_s_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_len_sum: int,
    max_seq_len: int,
    page_size: int,
    index_head_dim: int,
):
    """
    Fused gather of both K (key) and S (scale) data from paged buffer using Triton.
    This is more efficient than calling GetK and GetS separately.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_lens: tensor of sequence lens, int64
    :param seq_len_sum: sum of all sequence len, int32
    :param max_seq_len: max of sequence len, int32
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: tuple of (k_out, s_out) where
             k_out: (seq_len, index_head_dim), uint8
             s_out: (seq_len, 4), uint8
    """
    # Allocate outputs
    k_out = torch.empty(
        (seq_len_sum, index_head_dim), dtype=torch.uint8, device=buf.device
    )
    s_out = torch.empty((seq_len_sum, 4), dtype=torch.uint8, device=buf.device)

    _, buf_numel_per_page = buf.shape
    _, page_indice_batch_offset = page_indices.shape
    s_offset_in_page = page_size * index_head_dim

    # Launch kernel with one thread per token
    BLOCK_SIZE = 256
    BLOCK_SIZE_K = 128

    num_token_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_k_threads = (index_head_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    seq_num = seq_lens.shape[0]
    grid = (seq_num, num_token_blocks, num_k_threads)
    seq_num_pow2 = 1
    while seq_num_pow2 < seq_num:
        seq_num_pow2 *= 2

    _get_k_and_s_triton_kernel[grid](
        buf_ptr=buf,
        page_indices_ptr=page_indices,
        k_out_ptr=k_out,
        s_out_ptr=s_out,
        seq_len_ptr=seq_lens,
        seq_len_num_pow=seq_num_pow2,
        page_size=page_size,
        buf_numel_per_page=buf_numel_per_page,
        index_head_dim=index_head_dim,
        s_offset_in_page=s_offset_in_page,
        page_indice_batch_offset=page_indice_batch_offset,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return k_out, s_out


@triton.jit
def _get_k_and_s_triton_kernel(
    buf_ptr,
    page_indices_ptr,
    k_out_ptr,
    s_out_ptr,
    seq_len_ptr,
    seq_len_num_pow: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    index_head_dim: tl.constexpr,
    s_offset_in_page: tl.constexpr,
    page_indice_batch_offset,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel that gathers both K and S data in a single pass.
    Each program handles one token (seq_len tokens total).
    Loads 128 bytes (K) + 4 bytes (S) from the appropriate page.
    """
    batch_id = tl.program_id(0)
    block_token_start = tl.program_id(1) * BLOCK_SIZE
    thread_idx = tl.program_id(2)

    # Define the token range within the block and the K dimension range handled by the thread.
    token_ids_in_block = tl.arange(0, BLOCK_SIZE)
    token_ids = block_token_start + token_ids_in_block
    k_offsets = thread_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    seq_len = tl.load(seq_len_ptr + batch_id)
    # Grid axis 1 spans the batch-max seq len; fully-masked blocks store nothing.
    if block_token_start >= seq_len:
        return
    token_valid_mask = token_ids < seq_len

    pre_batch_idx = tl.arange(0, seq_len_num_pow)
    mask_pre_batch_idx = pre_batch_idx < batch_id
    prev_seq_lens = tl.load(seq_len_ptr + pre_batch_idx, mask=mask_pre_batch_idx)
    batch_token_offset = tl.sum(prev_seq_lens)

    # Batch calculate the page index and in-page offset of each token.
    page_idx = token_ids // page_size
    token_offset_in_page = token_ids % page_size
    page_indices_base = batch_id * page_indice_batch_offset
    page_idx_valid_mask = page_idx < page_indice_batch_offset
    page_index = tl.load(
        page_indices_ptr + page_idx + page_indices_base,
        mask=token_valid_mask & page_idx_valid_mask,
    )

    # ===== Load K data =====
    # The address calculation logic for K: page_index * total number of elements in a single page + K offset of the token within the page.
    k_src_token_offset = token_offset_in_page * index_head_dim
    k_src_base_offset = page_index * buf_numel_per_page + k_src_token_offset

    k_load_addr = buf_ptr + k_src_base_offset[:, None] + k_offsets[None, :]
    k_dim_mask = k_offsets[None, :] < index_head_dim
    k_mask = token_valid_mask[:, None] & k_dim_mask

    k_data = tl.load(k_load_addr, mask=k_mask, other=0)

    # Store K to output
    k_dst_token_offset = batch_token_offset + token_ids
    k_dst_base_offset = k_dst_token_offset * index_head_dim
    k_store_addr = k_out_ptr + k_dst_base_offset[:, None] + k_offsets[None, :]
    tl.store(k_store_addr, k_data, mask=k_mask)

    # ===== Load S data =====
    # The address calculation logic for S: page_index * total number of elements in a single page + starting offset of S within the page + offset of token within S in the page
    s_src_token_offset = s_offset_in_page + token_offset_in_page * 4
    s_src_base_offset = page_index * buf_numel_per_page + s_src_token_offset

    s_offsets = tl.arange(0, 4)
    s_load_addr = buf_ptr + s_src_base_offset[:, None] + s_offsets[None, :]
    s_mask = token_valid_mask[:, None] & (s_offsets[None, :] < 4)
    s_data = tl.load(s_load_addr, mask=s_mask, other=0)

    # Store S to output
    s_dst_token_offset = batch_token_offset + token_ids
    s_dst_base_offset = s_dst_token_offset * 4
    s_store_addr = s_out_ptr + s_dst_base_offset[:, None] + s_offsets[None, :]
    tl.store(s_store_addr, s_data, mask=s_mask)
