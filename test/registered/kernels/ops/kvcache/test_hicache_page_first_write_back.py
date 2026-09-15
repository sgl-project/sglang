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

from sglang.kernels.ops.kvcache.hicache import (
    can_use_write_back_jit_kernel,
    transfer_hicache_all_layer_mla_staged_lf_page_unified,
    transfer_hicache_all_layer_staged_lf_page_unified,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
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
    from sglang.srt.mem_cache.pool_host.common import (
        ALLOC_MEMORY_FUNCS,
        alloc_with_pin_memory,
    )

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
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

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
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

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


@pytest.mark.parametrize(
    "groups,layers,page_size,heads_per_group,dim",
    [(1, 1, 1, 1, 16), (3, 5, 3, 2, 24), (8, 3, 64, 1, 128)],
)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("page_count", [0, 1, 2, 5])
def test_page_unified_staged_write_back(
    groups, layers, page_size, heads_per_group, dim, dtype, index_dtype, page_count
):
    # Capacity 2 covers an exact chunk, repeated reuse, and a short final chunk.
    # Small and >=128 KiB pages exercise the fallback/batch-copy size branches
    # (the batch API additionally requires a supported CUDA runtime/driver).
    shape = (groups, layers, 2, page_size, heads_per_group, dim)
    generator = torch.Generator().manual_seed(1234)
    source_cpu = [
        [
            torch.randint(
                0,
                128,
                (6 * page_size, groups * heads_per_group, dim),
                generator=generator,
                dtype=torch.int32,
            ).to(dtype)
            for _ in range(layers)
        ]
        for _ in range(2)
    ]
    src_ids = [4, 0, 3, 4, 1][:page_count]  # Includes a repeated source page.
    dst_ids = [6, 2, 0, 5, 3][:page_count]
    dst = torch.full((8, *shape), 255, dtype=dtype, pin_memory=True)
    expected = dst.clone()
    for src_page, dst_page in zip(src_ids, dst_ids):
        for group in range(groups):
            for layer in range(layers):
                for kv in range(2):
                    expected[dst_page, group, layer, kv].copy_(
                        source_cpu[kv][layer][
                            src_page * page_size : (src_page + 1) * page_size,
                            group * heads_per_group : (group + 1) * heads_per_group,
                        ]
                    )

    # All GPU work uses a non-default stream; only synchronize after the call.
    # This also checks ordering when the same staging allocation is reused.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source_gpu = [[t.to(DEVICE) for t in component] for component in source_cpu]
        ptrs = [
            torch.tensor(
                [t.data_ptr() for t in component], dtype=torch.uint64, device=DEVICE
            )
            for component in source_gpu
        ]
        src_pages = torch.tensor(src_ids, dtype=index_dtype, device=DEVICE)
        dst_pages = torch.tensor(dst_ids, dtype=torch.int64)
        staging = torch.empty((2, *shape), dtype=dtype, device=DEVICE)
        transfer_hicache_all_layer_staged_lf_page_unified(
            *ptrs, src_pages, dst_pages, staging, dst
        )
    stream.synchronize()
    # Compare every element, including pages that should retain the sentinel.
    assert torch.equal(dst, expected)


@pytest.mark.parametrize(
    "invalid,match",
    [
        ("kv_axis", "K/V dimension"),
        ("group_alignment", "16-byte aligned"),
        ("staging_capacity", "at least one page"),
        ("page_shape", "matching page shapes"),
        ("noncontiguous", "must be contiguous"),
        ("page_count", "equal-length vectors"),
        ("dst_page", "destination page out of range"),
        ("layer_count", "page byte size mismatch"),
    ],
)
def test_page_unified_write_back_invalid_input(invalid, match):
    shape = (2, 3, 2, 4, 1, 16)
    if invalid == "kv_axis":
        shape = (2, 3, 3, 4, 1, 16)
    elif invalid == "group_alignment":
        shape = (2, 3, 2, 4, 1, 7)
    staging = torch.empty((1, *shape), dtype=torch.float16, device=DEVICE)
    dst = torch.empty((2, *shape), dtype=torch.float16, pin_memory=True)
    # Validation must reject the inputs before any pointer is dereferenced.
    ptrs = torch.zeros(3, dtype=torch.uint64, device=DEVICE)
    src_pages = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    dst_pages = torch.zeros(1, dtype=torch.int64)
    if invalid == "staging_capacity":
        staging = staging[:0]
    elif invalid == "page_shape":
        staging = staging[:, :1]
    elif invalid == "noncontiguous":
        staging = staging.transpose(1, 2).contiguous().transpose(1, 2)
    elif invalid == "page_count":
        dst_pages = dst_pages[:0]
    elif invalid == "dst_page":
        dst_pages.fill_(dst.shape[0])
    elif invalid == "layer_count":
        ptrs = ptrs[:2]
    # TVM FFI uses its own exception type for C++ RuntimeCheck failures.
    with pytest.raises(Exception, match=match):
        transfer_hicache_all_layer_staged_lf_page_unified(
            ptrs, ptrs, src_pages, dst_pages, staging, dst
        )


@pytest.mark.parametrize(
    "layers,page_size,dim", [(1, 1, 16), (5, 3, 512), (3, 64, 576)]
)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("page_count", [0, 1, 2, 5])
def test_page_unified_mla_staged_write_back(
    layers, page_size, dim, dtype, index_dtype, page_count
):
    generator = torch.Generator().manual_seed(5678)
    source_cpu = [
        torch.randint(0, 128, (6 * page_size, dim), generator=generator).to(dtype)
        for _ in range(layers)
    ]
    src_ids = [4, 0, 3, 4, 1][:page_count]
    dst_ids = [6, 2, 0, 5, 3][:page_count]
    shape = (layers, page_size, dim)
    dst = torch.full((8, *shape), 255, dtype=dtype, pin_memory=True)
    expected = dst.clone()
    for src_page, dst_page in zip(src_ids, dst_ids):
        for layer in range(layers):
            expected[dst_page, layer].copy_(
                source_cpu[layer][src_page * page_size : (src_page + 1) * page_size]
            )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source_gpu = [t.to(DEVICE) for t in source_cpu]
        ptrs = torch.tensor(
            [t.data_ptr() for t in source_gpu], dtype=torch.uint64, device=DEVICE
        )
        src_pages = torch.tensor(src_ids, dtype=index_dtype, device=DEVICE)
        dst_pages = torch.tensor(dst_ids, dtype=torch.int64)
        staging = torch.empty((2, *shape), dtype=dtype, device=DEVICE)
        transfer_hicache_all_layer_mla_staged_lf_page_unified(
            ptrs, src_pages, dst_pages, staging, dst
        )
    stream.synchronize()
    assert torch.equal(dst, expected)


@pytest.mark.parametrize(
    "invalid,match",
    [
        ("rank", "Expected MLA"),
        ("dimension", "must be positive"),
        ("alignment", "16-byte aligned"),
        ("staging_capacity", "at least one page"),
        ("page_shape", "matching page shapes"),
        ("noncontiguous", "must be contiguous"),
        ("dst_page", "destination page out of range"),
        ("layer_count", "page byte size mismatch"),
    ],
)
def test_page_unified_mla_write_back_invalid_input(invalid, match):
    shape = (3, 4, 16)
    if invalid == "dimension":
        shape = (3, 4, 0)
    elif invalid == "alignment":
        shape = (3, 4, 7)
    staging = torch.empty((1, *shape), dtype=torch.float16, device=DEVICE)
    dst = torch.empty((2, *shape), dtype=torch.float16, pin_memory=True)
    ptrs = torch.zeros(3, dtype=torch.uint64, device=DEVICE)
    src_pages = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    dst_pages = torch.zeros(1, dtype=torch.int64)
    if invalid == "rank":
        staging = staging.unsqueeze(1)
    elif invalid == "staging_capacity":
        staging = staging[:0]
    elif invalid == "page_shape":
        staging = staging[:, :1]
    elif invalid == "noncontiguous":
        staging = staging.transpose(1, 2).contiguous().transpose(1, 2)
    elif invalid == "dst_page":
        dst_pages.fill_(-1)
    elif invalid == "layer_count":
        ptrs = ptrs[:2]
    with pytest.raises(Exception, match=match):
        transfer_hicache_all_layer_mla_staged_lf_page_unified(
            ptrs, src_pages, dst_pages, staging, dst
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
