from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
    is_dsv4_layout: bool = False,
) -> Module:
    template_args = make_cpp_args(
        block_size, num_top_k, hot_buffer_size, is_mla, is_dsv4_layout
    )
    cache_args = make_cpp_args(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla, is_dsv4_layout
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_dsv4_transfer_module(block_size: int) -> Module:
    template_args = make_cpp_args(block_size)
    return load_jit(
        "sparse_cache_dsv4_transfer",
        block_size,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "transfer_cache_dsv4_mla",
                f"transfer_cache_dsv4_mla<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_copy_planned_module(
    block_size: int, is_mla: bool, is_dsv4_layout: bool
) -> Module:
    template_args = make_cpp_args(block_size, is_mla, is_dsv4_layout)
    return load_jit(
        "sparse_copy_planned",
        block_size,
        is_mla,
        is_dsv4_layout,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "copy_cache_planned",
                f"copy_cache_planned<{template_args}>",
            )
        ],
    )


def transfer_cache_dsv4_mla(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    block_size: int = 1024,
) -> None:
    """Transfer DSv4 C4 tokens between page-padded C4 buffers."""
    module = _jit_dsv4_transfer_module(block_size)
    module.transfer_cache_dsv4_mla(
        src_ptrs,
        dst_ptrs,
        src_indices,
        dst_indices,
    )


def _load_cache_to_device_buffer_mla(
    *,
    is_dsv4_layout: bool,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int,
    block_size: int,
    num_real_reqs: torch.Tensor | None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
) -> None:
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_sparse_module(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla=True,
        is_dsv4_layout=is_dsv4_layout,
    )

    empty = torch.empty(0)

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )

    # Optional miss-plan capture: 0-dim sentinels mean "don't record" in the kernel.
    if miss_src is None:
        none_i64 = torch.empty((), dtype=torch.int64, device=top_k_tokens.device)
        none_i32 = torch.empty((), dtype=torch.int32, device=top_k_tokens.device)
        miss_src, miss_dst, miss_count, plan_stride = none_i64, none_i64, none_i32, 0
    else:
        plan_stride = miss_src.stride(0)

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
        miss_src,
        miss_dst,
        miss_count,
        plan_stride,
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
) -> None:
    """Generic MLA hisparse swap-in: device + host both linear (stride=item_size_bytes).

    When miss_src/miss_dst/miss_count are given, the kernel also records this step's
    host->device miss plan into them for later replay by copy_cache_planned_mla.
    """
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=False,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
    )


def copy_cache_planned_mla(
    miss_src: torch.Tensor,
    miss_dst: torch.Tensor,
    miss_count: torch.Tensor,
    num_real_reqs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    item_size_bytes: int,
    num_blocks: int = 4,
    block_size: int = 1024,
    is_dsv4_layout: bool = False,
) -> None:
    """Replay a recorded miss plan (host_cache -> device_buffer) for a skip layer.

    IO-only: no planning / hit detection / LRU. Uses a small fixed grid
    (num_blocks) so the copies keep a low SM footprint while overlapping compute
    on a side stream. The slot table is shared with the anchor (lockstep layout).
    """
    module = _jit_copy_planned_module(block_size, True, is_dsv4_layout)
    empty = torch.empty(0)
    module.copy_cache_planned(
        miss_src,
        miss_dst,
        miss_count,
        num_real_reqs,
        host_cache,
        empty,
        device_buffer,
        empty,
        num_blocks,
        item_size_bytes,
    )


def load_cache_to_device_buffer_dsv4_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
) -> None:
    """DSv4 hisparse swap-in: page-padded device + page-padded host C4 layout."""
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=True,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
    )
