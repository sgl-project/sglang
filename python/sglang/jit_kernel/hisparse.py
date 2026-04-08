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
) -> Module:
    template_args = make_cpp_args(block_size, num_top_k, hot_buffer_size, is_mla)
    h2d_template_args = make_cpp_args(block_size, is_mla)
    cache_args = make_cpp_args(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "prepare_swap",
                f"prepare_swap<{template_args}>",
            ),
            (
                "execute_h2d_copy",
                f"execute_h2d_copy<{h2d_template_args}>",
            ),
        ],
    )


def prepare_swap_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    hit_device_locs: torch.Tensor,
    miss_device_locs: torch.Tensor,
    hit_count: torch.Tensor,
    miss_src_locs: torch.Tensor,
    miss_dst_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=True
    )

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )

    module.prepare_swap(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        top_k_device_locs,
        hit_device_locs,
        miss_device_locs,
        hit_count,
        miss_src_locs,
        miss_dst_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
    )


def execute_h2d_copy_mla(
    miss_src_locs: torch.Tensor,
    miss_dst_locs: torch.Tensor,
    hit_count: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    module = _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=True
    )

    empty = torch.empty(0)

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [miss_src_locs.size(0)], dtype=torch.int32, device=miss_src_locs.device
        )

    module.execute_h2d_copy(
        miss_src_locs,
        miss_dst_locs,
        hit_count,
        host_cache,
        empty,
        device_buffer,
        empty,
        num_real_reqs,
        num_top_k,
        item_size_bytes,
    )
