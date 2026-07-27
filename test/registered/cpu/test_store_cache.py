import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-b-test-cpu")

torch.manual_seed(42)

DEVICE = "cpu"
CACHE_SIZE = 4096

# for fp8 KV stored as uint8, e.g. float8_e4m3fn and float8_e5m2
DTYPES = [torch.float16, torch.bfloat16, torch.uint8]
DTYPE_IDS = ["float16", "bfloat16", "uint8"]


def _store_cache_cpu(k, v, k_cache, v_cache, indices):
    row_dim = k.size(1) * k.size(2)
    torch.ops.sgl_kernel.store_cache_cpu(k, v, k_cache, v_cache, indices, row_dim)


def _random_tensor(shape, dtype):
    """FP8 KV is stored as uint8; randn is not implemented for Byte."""
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=DEVICE)
    return torch.randn(shape, dtype=dtype, device=DEVICE)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 8, 16, 32])
@pytest.mark.parametrize("batch_size", [1, 7, 133])
def test_store_cache(batch_size, num_heads, head_dim, dtype):
    shape = (batch_size, num_heads, head_dim)
    cache_shape = (CACHE_SIZE, num_heads, head_dim)
    k = _random_tensor(shape, dtype)
    v = _random_tensor(shape, dtype)
    k_cache = _random_tensor(cache_shape, dtype)
    v_cache = _random_tensor(cache_shape, dtype)
    indices = torch.randperm(CACHE_SIZE, device=DEVICE, dtype=torch.int64)[:batch_size]

    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    k_cache_ref[indices] = k
    v_cache_ref[indices] = v

    _store_cache_cpu(k, v, k_cache, v_cache, indices)

    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("batch_size", [11])
def test_store_cache_int32_indices(batch_size, num_heads, head_dim, dtype):
    shape = (batch_size, num_heads, head_dim)
    cache_shape = (CACHE_SIZE, num_heads, head_dim)
    k = _random_tensor(shape, dtype)
    v = _random_tensor(shape, dtype)
    k_cache = _random_tensor(cache_shape, dtype)
    v_cache = _random_tensor(cache_shape, dtype)
    indices = torch.randperm(CACHE_SIZE, device=DEVICE, dtype=torch.int64)[
        :batch_size
    ].to(torch.int32)

    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    k_cache_ref[indices.long()] = k
    v_cache_ref[indices.long()] = v

    _store_cache_cpu(k, v, k_cache, v_cache, indices)

    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
