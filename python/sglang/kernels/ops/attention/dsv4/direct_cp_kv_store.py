"""Producer-direct CP stores into replicated packed DSV4 KV caches."""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    _blockwise_barrier,
)


@triton.jit
def _direct_store_barrier(signal_ptrs, RANK: tl.constexpr, WORLD: tl.constexpr):
    _blockwise_barrier(signal_ptrs, RANK, WORLD, sem="acq_rel")


@cache_once
def _jit_direct_store(dtype: torch.dtype, index_dtype: torch.dtype, page_size: int):
    args = make_cpp_args(dtype, index_dtype, page_size)
    return load_jit(
        "direct_cp_kv_store",
        *args,
        cuda_files=["deepseek_v4/direct_cp_kv_store.cuh"],
        cuda_wrappers=[("run", f"DirectCPKVStoreKernel<{args}>::run")],
    )


def direct_cp_kv_store(
    *,
    cache: torch.Tensor,
    handle: Any,
    cache_multicast: int,
    local_kv: torch.Tensor,
    local_indices: torch.Tensor,
    rank: int,
    world_size: int,
    page_size: int,
) -> None:
    if local_kv.shape != (local_indices.numel(), 512):
        raise ValueError(
            f"local KV shape {tuple(local_kv.shape)} does not match "
            f"indices {local_indices.numel()} x 512"
        )
    _jit_direct_store(local_kv.dtype, local_indices.dtype, page_size).run(
        local_kv,
        cache,
        int(cache_multicast),
        local_indices,
    )
    _direct_store_barrier[(1,)](
        handle.signal_pad_ptrs_dev,
        RANK=rank,
        WORLD=world_size,
        num_warps=1,
    )
