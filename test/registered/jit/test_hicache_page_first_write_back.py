"""Unit tests for the page_first + ``kernel`` JIT HiCache write-back / load path.

This file specifically exercises the JIT staged write-back and load kernels that
accept a CPU-resident destination index and stage through device memory
(``staged_write_back.cuh`` / ``hicache.cuh``). Unlike ``test_hicache.py`` (which
is registered CUDA-only), this file is also registered for the AMD PR-CI kernel
suite so the ROCm/HIP build and execution of those kernels are validated on AMD
hardware, not just CUDA.
"""

import sys

import pytest
import torch

from sglang.jit_kernel.hicache import can_use_write_back_jit_kernel
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or is_npu()
    or is_xpu()
    or not (is_cuda() or is_hip()),
    reason="HiCache JIT write-back tests require CUDA/ROCm.",
)

DEVICE = "cuda"
PAGE_SIZE = 1 if is_hip() else 16
NUM_LAYERS = 2
MHA_ELEMENT_DIMS = [128, 512]
MLA_ELEMENT_DIMS = [576]
# Include counts around and above the staging capacity so both the single-pass
# and the multi-chunk staged relayout branches are exercised.
PAGE_COUNTS = [1, 64, 65, 129]


def _token_indices_for_pages(
    pages: torch.Tensor,
    device: str = DEVICE,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    parts = [
        torch.arange(
            int(page) * PAGE_SIZE,
            (int(page) + 1) * PAGE_SIZE,
            device=device,
            dtype=dtype,
        )
        for page in pages.tolist()
    ]
    return torch.cat(parts, dim=0)


def _pinned_host_pool(host_pool_cls, **kwargs):
    original_alloc = ALLOC_MEMORY_FUNCS[DEVICE]
    ALLOC_MEMORY_FUNCS[DEVICE] = alloc_with_pin_memory
    try:
        return host_pool_cls(
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            pin_memory=True,
            device="cpu",
            **kwargs,
        )
    finally:
        ALLOC_MEMORY_FUNCS[DEVICE] = original_alloc


def _fill_with_offset(tensor: torch.Tensor, offset: int) -> None:
    data = torch.arange(
        tensor.numel(), device=tensor.device, dtype=tensor.dtype
    ).view_as(tensor)
    tensor.copy_(data + offset)


def _assert_pages_equal(host_ref, device_ref, host_pages, device_pages) -> None:
    for host_page, device_page in zip(host_pages.tolist(), device_pages.tolist()):
        host_start = host_page * PAGE_SIZE
        device_start = device_page * PAGE_SIZE
        assert torch.equal(
            host_ref[host_start : host_start + PAGE_SIZE].cpu(),
            device_ref[device_start : device_start + PAGE_SIZE].cpu(),
        )


def _run_mha(element_dim: int, page_count: int) -> None:
    pool_size = PAGE_SIZE * (page_count + 8)
    device_pool = MHATokenToKVPool(
        size=pool_size,
        page_size=PAGE_SIZE,
        head_num=element_dim // 128,
        head_dim=128,
        dtype=torch.bfloat16,
        layer_num=NUM_LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
    )
    host_pool = _pinned_host_pool(
        MHATokenToKVPoolHost, device_pool=device_pool, layout="page_first"
    )
    assert can_use_write_back_jit_kernel(
        element_size=element_dim * host_pool.dtype.itemsize,
    )
    # page_first + kernel staged write-back JIT path must be enabled.
    assert host_pool.can_use_write_back_jit

    for layer_id in range(NUM_LAYERS):
        _fill_with_offset(device_pool.k_buffer[layer_id], layer_id)
        _fill_with_offset(device_pool.v_buffer[layer_id], layer_id + 100)

    device_pages = torch.arange(2, 2 + page_count, device=DEVICE, dtype=torch.int64)
    host_pages = torch.arange(page_count, 0, -1, dtype=torch.int64)
    device_indices = _token_indices_for_pages(device_pages)
    # host_indices stay on the CPU: this is the case the staged JIT kernel must
    # accept (kDLCPU / kDLGPUHost destination indices).
    host_indices = _token_indices_for_pages(host_pages, device="cpu")
    assert not host_indices.is_cuda

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        _assert_pages_equal(
            host_pool.k_data_refs[layer_id],
            device_pool.k_buffer[layer_id],
            host_pages,
            device_pages,
        )
        _assert_pages_equal(
            host_pool.v_data_refs[layer_id],
            device_pool.v_buffer[layer_id],
            host_pages,
            device_pages,
        )

    # Load path (prefix-cache hit): exercises the hicache.cuh load matchers.
    if not host_pool.can_use_jit:
        return
    for layer_id in range(NUM_LAYERS):
        device_pool.k_buffer[layer_id].zero_()
        device_pool.v_buffer[layer_id].zero_()

    load_pages = torch.arange(1, 1 + page_count, device=DEVICE, dtype=torch.int64)
    load_indices = _token_indices_for_pages(load_pages)
    host_indices_device = host_indices.to(DEVICE)
    for layer_id in range(NUM_LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices_device, load_indices, layer_id, "kernel"
        )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        _assert_pages_equal(
            host_pool.k_data_refs[layer_id],
            device_pool.k_buffer[layer_id],
            host_pages,
            load_pages,
        )
        _assert_pages_equal(
            host_pool.v_data_refs[layer_id],
            device_pool.v_buffer[layer_id],
            host_pages,
            load_pages,
        )


def _run_mla(element_dim: int, page_count: int) -> None:
    pool_size = PAGE_SIZE * (page_count + 8)
    device_pool = MLATokenToKVPool(
        size=pool_size,
        page_size=PAGE_SIZE,
        kv_lora_rank=element_dim - 64,
        qk_rope_head_dim=64,
        dtype=torch.bfloat16,
        layer_num=NUM_LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
    )
    host_pool = _pinned_host_pool(
        MLATokenToKVPoolHost, device_pool=device_pool, layout="page_first"
    )
    assert can_use_write_back_jit_kernel(
        element_size=element_dim * host_pool.dtype.itemsize,
    )
    assert host_pool.can_use_write_back_jit

    for layer_id in range(NUM_LAYERS):
        _fill_with_offset(device_pool.kv_buffer[layer_id], layer_id)

    device_pages = torch.arange(2, 2 + page_count, device=DEVICE, dtype=torch.int64)
    host_pages = torch.arange(page_count, 0, -1, dtype=torch.int64)
    device_indices = _token_indices_for_pages(device_pages)
    host_indices = _token_indices_for_pages(host_pages, device="cpu")
    assert not host_indices.is_cuda

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        _assert_pages_equal(
            host_pool.data_refs[layer_id],
            device_pool.kv_buffer[layer_id],
            host_pages,
            device_pages,
        )

    if not host_pool.can_use_jit:
        return
    for layer_id in range(NUM_LAYERS):
        device_pool.kv_buffer[layer_id].zero_()

    load_pages = torch.arange(1, 1 + page_count, device=DEVICE, dtype=torch.int64)
    load_indices = _token_indices_for_pages(load_pages)
    host_indices_device = host_indices.to(DEVICE)
    for layer_id in range(NUM_LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices_device, load_indices, layer_id, "kernel"
        )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        _assert_pages_equal(
            host_pool.data_refs[layer_id],
            device_pool.kv_buffer[layer_id],
            host_pages,
            load_pages,
        )


@pytest.mark.parametrize("element_dim", MHA_ELEMENT_DIMS)
@pytest.mark.parametrize("page_count", PAGE_COUNTS)
def test_page_first_staged_write_back_mha(element_dim: int, page_count: int) -> None:
    _run_mha(element_dim, page_count)


@pytest.mark.parametrize("element_dim", MLA_ELEMENT_DIMS)
@pytest.mark.parametrize("page_count", PAGE_COUNTS)
def test_page_first_staged_write_back_mla(element_dim: int, page_count: int) -> None:
    _run_mla(element_dim, page_count)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
