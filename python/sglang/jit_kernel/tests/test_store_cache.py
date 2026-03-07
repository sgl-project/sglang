import itertools

import pytest
import torch

from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache

BS_LIST = [2**n for n in range(0, 15)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
HIDDEN_DIMS = [64, 128, 256, 512, 1024, 96, 98, 100]
CACHE_SIZE = 1024 * 1024
DTYPE = torch.bfloat16
DEVICE = "cuda"


@pytest.mark.parametrize(
    "batch_size,element_dim",
    list(itertools.product(BS_LIST, HIDDEN_DIMS)),
)
def test_store_cache(batch_size: int, element_dim: int) -> None:
    k = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE)
    v = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn((CACHE_SIZE, element_dim), dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn((CACHE_SIZE, element_dim), dtype=DTYPE, device=DEVICE)
    indices = torch.randperm(CACHE_SIZE, device=DEVICE)[:batch_size]

    # AOT store cache
    store_cache(k, v, k_cache, v_cache, indices)

    assert torch.all(k_cache[indices] == k)
    assert torch.all(v_cache[indices] == v)


# Smaller subset for targeted tests below
REPR_BS = [1, 7, 128]
REPR_DIMS = [64, 128, 512, 1024, 96]
SMALL_CACHE = 4096


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "batch_size,element_dim",
    list(itertools.product(REPR_BS, REPR_DIMS)),
)
def test_store_cache_dtypes(
    batch_size: int, element_dim: int, dtype: torch.dtype
) -> None:
    k = torch.randn((batch_size, element_dim), dtype=dtype, device=DEVICE)
    v = torch.randn((batch_size, element_dim), dtype=dtype, device=DEVICE)
    k_cache = torch.randn((SMALL_CACHE, element_dim), dtype=dtype, device=DEVICE)
    v_cache = torch.randn((SMALL_CACHE, element_dim), dtype=dtype, device=DEVICE)
    indices = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]

    store_cache(k, v, k_cache, v_cache, indices)

    assert torch.all(k_cache[indices] == k)
    assert torch.all(v_cache[indices] == v)


@pytest.mark.parametrize(
    "batch_size,element_dim",
    list(itertools.product(REPR_BS, REPR_DIMS)),
)
def test_store_cache_int32_indices(batch_size: int, element_dim: int) -> None:
    k = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE)
    v = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn((SMALL_CACHE, element_dim), dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn((SMALL_CACHE, element_dim), dtype=DTYPE, device=DEVICE)
    # int32 indices exercise a different CUDA template instantiation than default int64
    indices = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size].to(torch.int32)

    store_cache(k, v, k_cache, v_cache, indices)

    assert torch.all(k_cache[indices.long()] == k)
    assert torch.all(v_cache[indices.long()] == v)


def _valid_num_splits(element_dim: int, dtype: torch.dtype) -> list:
    """Return the list of valid num_split values for a given element_dim/dtype."""
    row_bytes = element_dim * dtype.itemsize
    splits = [1]
    if row_bytes % (2 * 128) == 0:
        splits.append(2)
    if row_bytes % (4 * 128) == 0:
        splits.append(4)
    return splits


_NUM_SPLIT_CASES = [
    (_dim, _ns, _dtype)
    for _dtype in [torch.float16, torch.bfloat16, torch.float32]
    for _dim in REPR_DIMS
    for _ns in _valid_num_splits(_dim, _dtype)
]


@pytest.mark.parametrize("element_dim,num_split,dtype", _NUM_SPLIT_CASES)
def test_store_cache_num_split(
    element_dim: int, num_split: int, dtype: torch.dtype
) -> None:
    batch_size = 128
    k = torch.randn((batch_size, element_dim), dtype=dtype, device=DEVICE)
    v = torch.randn((batch_size, element_dim), dtype=dtype, device=DEVICE)
    k_cache = torch.randn((SMALL_CACHE, element_dim), dtype=dtype, device=DEVICE)
    v_cache = torch.randn((SMALL_CACHE, element_dim), dtype=dtype, device=DEVICE)
    indices = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]

    # Verify each num_split kernel path (1, 2, 4) produces correct results
    store_cache(k, v, k_cache, v_cache, indices, num_split=num_split)

    assert torch.all(k_cache[indices] == k)
    assert torch.all(v_cache[indices] == v)


def test_can_use_store_cache() -> None:
    assert can_use_store_cache(128)
    assert can_use_store_cache(256)
    assert can_use_store_cache(1024)
    assert can_use_store_cache(2048)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
