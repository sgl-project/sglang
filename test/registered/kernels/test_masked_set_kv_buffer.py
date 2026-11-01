import sys

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import masked_set_kv_buffer_kernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


def _cache_entries() -> int:
    return sum(
        len(device_cache[0])
        for device_cache in masked_set_kv_buffer_kernel.device_caches.values()
    )


def _run_masked_set_kv_buffer(n: int) -> None:
    torch.manual_seed(n)
    num_heads, head_dim, chunk_size = 2, 8, 16
    capacity = 64

    key = torch.randn((n, num_heads, head_dim), device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    key_buffer = torch.full(
        (capacity, num_heads, head_dim), -1, device="cuda", dtype=key.dtype
    )
    value_buffer = torch.full_like(key_buffer, -1)
    locations = torch.arange(n - 1, -1, -1, device="cuda", dtype=torch.int64)
    write_mask = (torch.arange(n, device="cuda") % 2 == 0).to(torch.int32)

    masked_set_kv_buffer_kernel[(n,)](
        key,
        value,
        key_buffer,
        value_buffer,
        locations,
        write_mask,
        num_heads,
        head_dim,
        chunk_size,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
    )
    torch.cuda.synchronize()

    selected = write_mask.bool()
    torch.testing.assert_close(
        key_buffer[locations[selected]], key[selected], rtol=0, atol=0
    )
    torch.testing.assert_close(
        value_buffer[locations[selected]], value[selected], rtol=0, atol=0
    )
    assert torch.all(key_buffer[locations[~selected]] == -1)
    assert torch.all(value_buffer[locations[~selected]] == -1)


def test_batch_size_does_not_create_extra_specializations() -> None:
    masked_set_kv_buffer_kernel.device_caches.clear()
    try:
        _run_masked_set_kv_buffer(17)
        assert _cache_entries() == 1

        # Both launches have the same pointer/scalar types and integer
        # divisibility properties. Only the grid size differs.
        _run_masked_set_kv_buffer(33)
        assert _cache_entries() == 1
    finally:
        masked_set_kv_buffer_kernel.device_caches.clear()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
