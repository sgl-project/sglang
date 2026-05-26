# Copyright 2025 XunhaoLai. All rights reserved.

import functools
from collections import deque
from typing import Any, Callable, Tuple

import torch
import triton.language as tl

_tma_keep_alive_buf = deque(maxlen=200)

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
