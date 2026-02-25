"""
Correctness tests for the store JIT kernel.

Tests the JIT-compiled store_kv_cache against direct tensor indexing.
"""

import itertools

import pytest
import torch

from sglang.jit_kernel.store import store_kv_cache

CACHE_SIZE = 1024
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
INDEX_DTYPES = [torch.int32, torch.int64]
BATCH_SIZES = [1, 4, 16, 64, 128]
HEAD_DIMS = [64, 128, 256, 512]
DEVICE = "cuda"


@pytest.mark.parametrize(
    "batch_size,head_dim,dtype,index_dtype",
    list(itertools.product(BATCH_SIZES, HEAD_DIMS, DTYPES, INDEX_DTYPES)),
)
def test_store_kv_cache(
    batch_size: int,
    head_dim: int,
    dtype: torch.dtype,
    index_dtype: torch.dtype,
) -> None:
    k = torch.randn((batch_size, head_dim), dtype=dtype, device=DEVICE)
    v = torch.randn((batch_size, head_dim), dtype=dtype, device=DEVICE)
    k_cache = torch.zeros((CACHE_SIZE, head_dim), dtype=dtype, device=DEVICE)
    v_cache = torch.zeros((CACHE_SIZE, head_dim), dtype=dtype, device=DEVICE)
    indices = torch.randperm(CACHE_SIZE, device=DEVICE)[:batch_size].to(index_dtype)

    store_kv_cache(k_cache, v_cache, indices, k, v)

    assert torch.all(k_cache[indices] == k), "k mismatch"
    assert torch.all(v_cache[indices] == v), "v mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
