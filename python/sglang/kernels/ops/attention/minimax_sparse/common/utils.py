# Copyright 2025 XunhaoLai. All rights reserved.

import functools
from collections import deque
from typing import Any, Callable, List, Optional, Tuple

import torch
import triton
import triton.language as tl

_tma_keep_alive_buf = deque(maxlen=200)

# Q is always bf16/fp16. The paged main K/V cache may be fp8 (unit-scaled) under
# --kv-cache-dtype fp8_*; the kernel widens it to the Q dtype on load (IS_FP8
# branch). Accepted on both HIP and CUDA (the bf16->fp8 cache write is unit-scaled,
# so the widening cast is the exact inverse dequant). The bf16/fp16-only MSA
# (fmha_sm100) kernel is excluded for fp8 KV by the backend use_msa gate.
SPARSE_KV_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
)


def check_sparse_kv_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    label: str,
) -> bool:
    """Validate the sparse-attention KV cache dtype contract.

    Returns True iff the K cache is fp8 (then widened to Q dtype in the kernel).
    Raises AssertionError otherwise, mirroring the contract the decode and prefill
    topk kernels both enforce. fp8 is accepted on both HIP and CUDA.
    """
    assert q.dtype in (torch.bfloat16, torch.float16)
    is_fp8 = k_cache.dtype in SPARSE_KV_FP8_DTYPES
    assert k_cache.dtype == q.dtype or is_fp8, (
        f"sparse {label} expects K cache dtype == Q dtype ({q.dtype}) "
        f"or fp8, got {k_cache.dtype}"
    )
    assert v_cache.dtype == k_cache.dtype
    return is_fp8


try:
    make_tensor_descriptor = tl.make_tensor_descriptor
except Exception:
    make_tensor_descriptor = tl._experimental_make_tensor_descriptor


def robust_allocator(size: int, alignment: int, stream: int = None):
    """Allocator for Triton TMA descriptors.

    We keep reference in deque to prevent GC from collecting the buffer.
    """
    tensor = torch.empty(size, device="cuda", dtype=torch.uint8)
    _tma_keep_alive_buf.append(tensor)
    return tensor


def tensor_cache(maxsize: int = 8):
    """
    Cache function results using identity comparison.
    Zero-overhead cache hit: no hash, no DtoH, just pointer comparison.

    Args:
        maxsize: Maximum number of cached entries. Supports multi-GPU scenarios
                 where different devices have different tensor arguments.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # LRU-style cache: list of (args, kwargs, result) tuples
        # Most recently used at the end
        _cache: list = []

        def _args_match(args: tuple, cached_args: tuple) -> bool:
            if len(args) != len(cached_args):
                return False
            for i in range(len(args)):
                if args[i] is not cached_args[i]:
                    return False
            return True

        def _kwargs_match(kwargs: dict, cached_kwargs: dict) -> bool:
            if not kwargs and not cached_kwargs:
                return True
            if kwargs.keys() != cached_kwargs.keys():
                return False
            for k, v in kwargs.items():
                if v is not cached_kwargs[k]:
                    return False
            return True

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Search cache (most recent first for better hit rate)
            for i in range(len(_cache) - 1, -1, -1):
                cached_args, cached_kwargs, cached_result = _cache[i]
                if _args_match(args, cached_args) and _kwargs_match(
                    kwargs, cached_kwargs
                ):
                    # Move to end (most recently used)
                    if i != len(_cache) - 1:
                        _cache.append(_cache.pop(i))
                    return cached_result

            # Cache miss
            result = fn(*args, **kwargs)

            # Add to cache
            if len(_cache) >= maxsize:
                _cache.pop(0)  # Remove oldest
            _cache.append((args, kwargs, result))
            return result

        # Expose cache for manual clearing if needed
        wrapper.cache_clear = lambda: _cache.clear()
        wrapper.cache_info = lambda: {"size": len(_cache), "maxsize": maxsize}

        return wrapper

    return decorator


@tensor_cache(maxsize=8)
def get_cu_seqblocks(
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    block_size_q: int,
    block_size_k: int,
    seqlens_cpu: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
    """Compute cumulative sequence block indices for blocked sparse attention.

    Converts token-level cumulative sequence lengths to block-level indices,
    which are needed for block-sparse attention kernels.

    Note:
        Results are cached (maxsize=8) based on input arguments. Repeated calls
        with the same cu_seqlens, max_seqlen, and block sizes will return cached
        results without recomputation.

    Args:
        cu_seqlens: Cumulative sequence lengths. Shape: [batch_size + 1], dtype: int32.
        max_seqlen: Maximum sequence length in the batch.
        block_size_q: Query block size.
        block_size_k: Key-value block size.
        seqlens_cpu: Optional host copy of ``torch.diff(cu_seqlens)``; when given,
            ``all_seqblock_q/k`` are summed on the host to avoid a per-layer sync.

    Returns:
        A tuple of 6 values:
            - cu_seqblocks_q: Cumulative query block indices. Shape: [batch_size + 1]
            - max_seqblock_q: Maximum number of query blocks per sequence.
            - all_seqblock_q: Total number of query blocks across all sequences.
            - cu_seqblocks_k: Cumulative key block indices. Shape: [batch_size + 1]
            - max_seqblock_k: Maximum number of key blocks per sequence.
            - all_seqblock_k: Total number of key blocks across all sequences.
    """
    cu_seqblocks_q = torch.zeros_like(cu_seqlens)
    cu_seqblocks_k = torch.zeros_like(cu_seqlens)
    seq_lens = torch.diff(cu_seqlens)
    seqblocks_q = (seq_lens + block_size_q - 1) // block_size_q
    seqblocks_k = (seq_lens + block_size_k - 1) // block_size_k
    max_seqblock_q = (max_seqlen + block_size_q - 1) // block_size_q
    max_seqblock_k = (max_seqlen + block_size_k - 1) // block_size_k
    cu_seqblocks_q[1:] = seqblocks_q
    cu_seqblocks_k[1:] = seqblocks_k
    cu_seqblocks_q.cumsum_(0)
    cu_seqblocks_k.cumsum_(0)
    if seqlens_cpu is not None:
        # Bit-identical to seqblocks.sum().item() but no device->host sync.
        all_seqblock_q = sum(
            (s + block_size_q - 1) // block_size_q for s in seqlens_cpu
        )
        all_seqblock_k = sum(
            (s + block_size_k - 1) // block_size_k for s in seqlens_cpu
        )
    else:
        all_seqblock_q = seqblocks_q.sum().item()
        all_seqblock_k = seqblocks_k.sum().item()
    return (
        cu_seqblocks_q,
        max_seqblock_q,
        all_seqblock_q,
        cu_seqblocks_k,
        max_seqblock_k,
        all_seqblock_k,
    )


# Bitonic-sort compare-and-swap primitives shared by the decode and prefill
# topk-index kernels. Identical copies previously lived in both
# decode/flash_with_topk_idx.py and prefill/flash_with_topk_idx.py.
@triton.jit
def _compare_and_swap(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [
            n_outer * 2 ** (n_dims - 1 - stage),
            2,
            2**stage,
        ]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids
