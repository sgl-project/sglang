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


@pytest.mark.skipif(not _sm90_available(), reason="requires an SM90 CUDA device")
def test_dsv4_nvfp4_dequant_large_shuffled_pages() -> None:
    """Exercise the shared one-CTA kernel with DSV4's 28-block row."""

    torch.manual_seed(23)
    page_size = 64
    num_pages = 2
    capacity = num_pages * page_size
    global_scale = 0.5
    cache_k = torch.randn(
        (capacity, DSV4_NVFP4_NOPE_DIM + DSV4_NVFP4_ROPE_DIM),
        dtype=torch.bfloat16,
    )
    cache_cpu = torch.zeros(
        (num_pages, page_size * DSV4_NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8
    )
    loc = torch.arange(capacity, dtype=torch.int32)
    quantize_dsv4_nvfp4_k_cache_into(
        cache_k, cache_cpu, loc, page_size, global_scale
    )

    pattern = torch.cat(
        (
            torch.randperm(capacity, dtype=torch.int32),
            torch.tensor([-1, capacity, 11, 11], dtype=torch.int32),
        )
    )
    indices_cpu = pattern.repeat(8)[:1024]
    expected = dequantize_dsv4_nvfp4_k_cache_paged(
        cache_cpu, indices_cpu, page_size, global_scale
    )
    actual = dequantize_dsv4_nvfp4_k_cache_paged(
        cache_cpu.cuda(), indices_cpu.cuda(), page_size, global_scale
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    invalid = torch.tensor([-1, capacity] * 33, dtype=torch.int64, device="cuda")
    invalid_output = dequantize_dsv4_nvfp4_k_cache_paged(
        cache_cpu.cuda(), invalid, page_size, global_scale
    )
    assert torch.count_nonzero(invalid_output).item() == 0


@pytest.mark.skipif(not _sm90_available(), reason="requires an SM90 CUDA device")
@pytest.mark.parametrize("page_size", [2, 64, 256])
def test_dsv4_nvfp4_quant_grouped_matches_exact_paged_reference(
    page_size: int,
) -> None:
    num_inputs = 65
    num_pages = max(2, (num_inputs + 15 + page_size - 1) // page_size)
    capacity = num_pages * page_size
    cache_k = torch.zeros(
        (num_inputs, DSV4_NVFP4_NOPE_DIM + DSV4_NVFP4_ROPE_DIM),
        dtype=torch.bfloat16,
    )
    nope = cache_k[:, :DSV4_NVFP4_NOPE_DIM].view(num_inputs, -1, 16)
    nope[:, :, 0] = 6.0
    nope[:, :, 1] = -3.0
    nope[:, :, 2] = 1.5
    cache_k[:, DSV4_NVFP4_NOPE_DIM:] = (
        torch.arange(num_inputs * DSV4_NVFP4_ROPE_DIM)
        .reshape(num_inputs, DSV4_NVFP4_ROPE_DIM)
        .remainder(67)
        .sub(33)
        .div(16)
        .to(torch.bfloat16)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(53 + page_size)
    loc = torch.randperm(capacity, generator=generator, dtype=torch.int32)[:num_inputs]
    loc[-2] = -1
    loc[-1] = capacity
    expected = torch.full(
        (num_pages, page_size * DSV4_NVFP4_BYTES_PER_TOKEN),
        0xA5,
        dtype=torch.uint8,
    )
    quantize_dsv4_nvfp4_k_cache_into(
        cache_k, expected, loc, page_size, global_scale=0.5
    )
    actual = torch.full_like(expected, 0xA5, device="cuda")
    global_scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    quantize_dsv4_nvfp4_k_cache_into(
        cache_k.cuda(), actual, loc.cuda(), page_size, global_scale
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
