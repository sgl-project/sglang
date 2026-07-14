from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.attention.dsv4.nvfp4_k_cache import (
    DSV4_NVFP4_BYTES_PER_TOKEN,
    DSV4_NVFP4_NOPE_DIM,
    DSV4_NVFP4_ROPE_DIM,
    dequantize_dsv4_nvfp4_k_cache_paged,
    quantize_dsv4_nvfp4_k_cache_into,
)


def _sm90_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.mark.skipif(not _sm90_available(), reason="requires an SM90 CUDA device")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_dsv4_nvfp4_cuda_matches_cpu_reference(index_dtype: torch.dtype) -> None:
    torch.manual_seed(19)
    page_size = 4
    num_pages = 3
    num_inputs = 5
    global_scale = 0.5
    cache_k_cpu = torch.randn(
        (num_inputs, DSV4_NVFP4_NOPE_DIM + DSV4_NVFP4_ROPE_DIM),
        dtype=torch.bfloat16,
    )
    loc_cpu = torch.tensor([5, 1, -1, 9, num_pages * page_size], dtype=index_dtype)
    cache_cpu = torch.full(
        (num_pages, page_size * DSV4_NVFP4_BYTES_PER_TOKEN),
        0x5A,
        dtype=torch.uint8,
    )
    quantize_dsv4_nvfp4_k_cache_into(
        cache_k_cpu, cache_cpu, loc_cpu, page_size, global_scale
    )

    cache_k_cuda = cache_k_cpu.cuda()
    loc_cuda = loc_cpu.cuda()
    cache_cuda = torch.full_like(cache_cpu, 0x5A, device="cuda")
    quantize_dsv4_nvfp4_k_cache_into(
        cache_k_cuda, cache_cuda, loc_cuda, page_size, global_scale
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(cache_cuda.cpu(), cache_cpu, rtol=0, atol=0)

    gather_cpu = torch.tensor(
        [9, -1, 1, 7, 5, num_pages * page_size], dtype=index_dtype
    )
    expected = dequantize_dsv4_nvfp4_k_cache_paged(
        cache_cpu, gather_cpu, page_size, global_scale
    )
    actual = dequantize_dsv4_nvfp4_k_cache_paged(
        cache_cuda, gather_cpu.cuda(), page_size, global_scale
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[1]).item() == 0
    assert torch.count_nonzero(actual[-1]).item() == 0
