import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.hicache import can_use_write_back_jit_kernel
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or is_npu()
    or is_xpu()
    or not (is_cuda() or is_hip()),
    reason="HiCache JIT tests require CUDA/ROCm.",
)

DEVICE = "cuda"
PAGE_SIZE = 1 if is_hip() else 16
NUM_LAYERS = 2
POOL_SIZE = PAGE_SIZE * 8
MHA_ELEMENT_DIMS = [128, 256, 512, 1024]
MLA_ELEMENT_DIMS = [576]
LAYOUTS = ["layer_first", "page_first"]
STAGED_WRITE_BACK_PAGE_COUNTS = [1, 63, 64, 65, 67, 128, 129]


@pytest.mark.parametrize("direction", ["d2h", "h2d"])
@pytest.mark.parametrize("backend", ["direct", "kernel"])
@pytest.mark.skipif(not is_cuda(), reason="Uses a delayed CUDA producer stream")
def test_unified_l2_waits_for_index_producer(direction, backend):
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    bundle = init_unified_swa_pools(
        device=DEVICE,
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=4,
        v_head_dim=4,
        swa_head_num=1,
        swa_head_dim=4,
        swa_v_head_dim=4,
        page_size=1,
        start_layer=0,
        end_layer=2,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        total_bytes=4096,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=True,
    )
    device_pool = bundle.token_to_kv_pool.full_kv_pool
    host_pool = UnifiedPageEnvelopeHostPool(
        device_pool,
        1.0,
        0,
        1,
        "layer_first",
        pin_memory=True,
        allocator_type="default",
    )
    engine = L2TransferEngine(backend)
    producer = torch.cuda.Stream()
    device_indices = torch.tensor([1], device=DEVICE)
    host_indices = host_pool.alloc(1)
    transfer = L2Transfer(host_pool, device_pool, host_indices, device_indices)
    device_pages = device_pool.get_page_envelope_buffer()
    try:
        # Warm both copy paths before creating the delayed producer dependency.
        engine.submit_device_to_host([transfer]).finish_event.synchronize()
        engine.submit_host_to_device([transfer], layer_num=1).finish_event.synchronize()
        device_pages.zero_()
        device_pages[5].fill_(77)
        host_pool.get_data_page(int(host_indices[0])).fill_(33)
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            torch.cuda._sleep(400_000_000)
            device_indices.fill_(5)
            if direction == "d2h":
                completion = engine.submit_device_to_host([transfer])
            else:
                ready = torch.cuda.Event()
                ready.record()
                completion = engine.submit_host_to_device(
                    [transfer], layer_num=1, start_event=ready
                )
        completion.finish_event.synchronize()
        if direction == "d2h":
            assert torch.all(host_pool.get_data_page(int(host_indices[0])) == 77)
        else:
            assert torch.all(device_pages[5] == 33)
            assert torch.all(device_pages[1] == 0)
    finally:
        producer.synchronize()
        host_pool.destroy()
        reset_context()


def _token_indices_for_pages(
    pages: torch.Tensor,
    page_size: int = PAGE_SIZE,
    device: str = DEVICE,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    parts = [
        torch.arange(
            int(page) * page_size,
            (int(page) + 1) * page_size,
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


def _copy_tensor_with_offset(tensor: torch.Tensor, offset: int) -> None:
    data = torch.arange(
        tensor.numel(), device=tensor.device, dtype=tensor.dtype
    ).view_as(tensor)
    tensor.copy_(data + offset)


def _assert_page_filled(tensor: torch.Tensor, page: int, value: float) -> None:
    page_slice = tensor[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
    expected = torch.full_like(page_slice, value)
    assert torch.equal(page_slice.cpu(), expected.cpu())


def _run_transfer_roundtrip_mha(layout: str, element_dim: int) -> None:
    device_pool = MHATokenToKVPool(
        size=POOL_SIZE,
        page_size=PAGE_SIZE,
        head_num=element_dim // 128,
        head_dim=128,
        dtype=torch.bfloat16,
        layer_num=NUM_LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
    )
    host_pool = _pinned_host_pool(
        MHATokenToKVPoolHost,
        device_pool=device_pool,
        layout=layout,
    )
    assert host_pool.can_use_jit, (
        f"Expected JIT HiCache kernel for MHA dim={element_dim}"
    )

    for layer_id in range(NUM_LAYERS):
        _copy_tensor_with_offset(device_pool.k_buffer[layer_id], layer_id)
        _copy_tensor_with_offset(device_pool.v_buffer[layer_id], layer_id + 100)

    device_pages = torch.tensor([1, 2, 3], device=DEVICE, dtype=torch.int64)
    host_pages = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int64)
    device_indices = _token_indices_for_pages(device_pages)
    host_indices = _token_indices_for_pages(host_pages)
    host_indices_backup = (
        _token_indices_for_pages(host_pages, device="cpu")
        if layout == "page_first"
        else host_indices
    )

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices_backup, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), device_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                host_pool.k_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.k_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )
            assert torch.equal(
                host_pool.v_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.v_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )

    for layer_id in range(NUM_LAYERS):
        device_pool.k_buffer[layer_id].zero_()
        device_pool.v_buffer[layer_id].zero_()

    load_pages = torch.tensor([4, 5, 6], device=DEVICE, dtype=torch.int64)
    load_indices = _token_indices_for_pages(load_pages)
    for layer_id in range(NUM_LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices, load_indices, layer_id, "kernel"
        )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), load_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                device_pool.k_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
                host_pool.k_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
            )
            assert torch.equal(
                device_pool.v_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
                host_pool.v_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
            )


def _run_transfer_roundtrip_mla(layout: str, element_dim: int) -> None:
    device_pool = MLATokenToKVPool(
        size=POOL_SIZE,
        page_size=PAGE_SIZE,
        kv_lora_rank=element_dim - 64,
        qk_rope_head_dim=64,
        dtype=torch.bfloat16,
        layer_num=NUM_LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
    )
    host_pool = _pinned_host_pool(
        MLATokenToKVPoolHost,
        device_pool=device_pool,
        layout=layout,
    )
    assert host_pool.can_use_jit, (
        f"Expected JIT HiCache kernel for MLA dim={element_dim}"
    )

    for layer_id in range(NUM_LAYERS):
        _copy_tensor_with_offset(device_pool.kv_buffer[layer_id], layer_id)

    device_pages = torch.tensor([1, 2, 3], device=DEVICE, dtype=torch.int64)
    host_pages = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int64)
    device_indices = _token_indices_for_pages(device_pages)
    host_indices = _token_indices_for_pages(host_pages)
    host_indices_backup = (
        _token_indices_for_pages(host_pages, device="cpu")
        if layout == "page_first"
        else host_indices
    )

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices_backup, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), device_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                host_pool.data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.kv_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )

    for layer_id in range(NUM_LAYERS):
        device_pool.kv_buffer[layer_id].zero_()

    load_pages = torch.tensor([4, 5, 6], device=DEVICE, dtype=torch.int64)
    load_indices = _token_indices_for_pages(load_pages)
    for layer_id in range(NUM_LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices, load_indices, layer_id, "kernel"
        )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), load_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                device_pool.kv_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
                host_pool.data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
            )


def _run_page_first_staged_write_back_mha(
    layout: str, element_dim: int, page_count: int
) -> None:
    pool_size = PAGE_SIZE * (page_count + 8)
    head_num = (
        element_dim // 128 if element_dim >= 128 and element_dim % 128 == 0 else 1
    )
    head_dim = element_dim // head_num
    device_pool = MHATokenToKVPool(
        size=pool_size,
        page_size=PAGE_SIZE,
        head_num=head_num,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        layer_num=NUM_LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
    )
    host_pool = _pinned_host_pool(
        MHATokenToKVPoolHost,
        device_pool=device_pool,
        layout=layout,
    )
    assert can_use_write_back_jit_kernel(
        element_size=element_dim * host_pool.dtype.itemsize,
    )
    assert host_pool.can_use_write_back_jit
    if element_dim * host_pool.dtype.itemsize % 128 != 0:
        assert not host_pool.can_use_jit
    assert host_pool.staging_page_capacity > 0
    if page_count > 64:
        assert host_pool.staging_page_capacity < page_count

    for layer_id in range(NUM_LAYERS):
        _copy_tensor_with_offset(device_pool.k_buffer[layer_id], layer_id)
        _copy_tensor_with_offset(device_pool.v_buffer[layer_id], layer_id + 100)
    host_pool.k_buffer.fill_(-7)
    host_pool.v_buffer.fill_(-11)

    device_pages = torch.arange(
        2,
        2 + page_count,
        device=DEVICE,
        dtype=torch.int64,
    )
    host_pages = torch.arange(
        page_count,
        0,
        -1,
        dtype=torch.int64,
    )
    src_index_dtype = torch.int32 if page_count == 64 else torch.int64
    device_indices = _token_indices_for_pages(device_pages, dtype=src_index_dtype)
    host_indices = _token_indices_for_pages(host_pages, device="cpu")
    assert not host_indices.is_cuda
    expected_k = [
        device_pool.k_buffer[layer_id][device_indices.to(dtype=torch.int64)].cpu()
        for layer_id in range(NUM_LAYERS)
    ]
    expected_v = [
        device_pool.v_buffer[layer_id][device_indices.to(dtype=torch.int64)].cpu()
        for layer_id in range(NUM_LAYERS)
    ]

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), device_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                host_pool.k_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.k_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )
            assert torch.equal(
                host_pool.v_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.v_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )

    for layer_id in range(NUM_LAYERS):
        for untouched_page in [0, page_count + 1]:
            _assert_page_filled(host_pool.k_data_refs[layer_id], untouched_page, -7)
            _assert_page_filled(host_pool.v_data_refs[layer_id], untouched_page, -11)

    for layer_id in range(NUM_LAYERS):
        device_pool.k_buffer[layer_id].zero_()
        device_pool.v_buffer[layer_id].zero_()
    load_indices = device_indices.to(dtype=torch.int64)
    host_indices_load = _token_indices_for_pages(host_pages)
    for layer_id in range(NUM_LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices_load, load_indices, layer_id, "kernel"
        )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        assert torch.equal(
            device_pool.k_buffer[layer_id][load_indices].cpu(), expected_k[layer_id]
        )
        assert torch.equal(
            device_pool.v_buffer[layer_id][load_indices].cpu(), expected_v[layer_id]
        )


def _run_page_first_staged_write_back_mla(
    layout: str, element_dim: int, page_count: int
) -> None:
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
        MLATokenToKVPoolHost,
        device_pool=device_pool,
        layout=layout,
    )
    assert can_use_write_back_jit_kernel(
        element_size=element_dim * host_pool.dtype.itemsize,
    )
    assert host_pool.can_use_write_back_jit
    if element_dim * host_pool.dtype.itemsize % 128 != 0:
        assert not host_pool.can_use_jit
    assert host_pool.staging_page_capacity > 0
    if page_count > 64:
        assert host_pool.staging_page_capacity < page_count

    for layer_id in range(NUM_LAYERS):
        _copy_tensor_with_offset(device_pool.kv_buffer[layer_id], layer_id)
    host_pool.kv_buffer.fill_(-13)

    device_pages = torch.arange(
        2,
        2 + page_count,
        device=DEVICE,
        dtype=torch.int64,
    )
    host_pages = torch.arange(
        page_count,
        0,
        -1,
        dtype=torch.int64,
    )
    src_index_dtype = torch.int32 if page_count == 64 else torch.int64
    device_indices = _token_indices_for_pages(device_pages, dtype=src_index_dtype)
    host_indices = _token_indices_for_pages(host_pages, device="cpu")
    assert not host_indices.is_cuda
    expected = [
        device_pool.kv_buffer[layer_id][device_indices.to(dtype=torch.int64)].cpu()
        for layer_id in range(NUM_LAYERS)
    ]

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), device_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                host_pool.data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.kv_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )

    for layer_id in range(NUM_LAYERS):
        for untouched_page in [0, page_count + 1]:
            _assert_page_filled(host_pool.data_refs[layer_id], untouched_page, -13)

    for layer_id in range(NUM_LAYERS):
        device_pool.kv_buffer[layer_id].zero_()
    load_indices = device_indices.to(dtype=torch.int64)
    host_indices_load = _token_indices_for_pages(host_pages)
    for layer_id in range(NUM_LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices_load, load_indices, layer_id, "kernel"
        )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        assert torch.equal(
            device_pool.kv_buffer[layer_id][load_indices].cpu(), expected[layer_id]
        )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("element_dim", MHA_ELEMENT_DIMS)
def test_hicache_transfer_mha(layout: str, element_dim: int) -> None:
    _run_transfer_roundtrip_mha(layout, element_dim)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("element_dim", MLA_ELEMENT_DIMS)
def test_hicache_transfer_mla(layout: str, element_dim: int) -> None:
    _run_transfer_roundtrip_mla(layout, element_dim)


@pytest.mark.parametrize("layout", ["page_first"])
@pytest.mark.parametrize("element_dim", MHA_ELEMENT_DIMS)
@pytest.mark.parametrize("page_count", STAGED_WRITE_BACK_PAGE_COUNTS)
def test_hicache_page_first_staged_write_back_mha(
    layout: str, element_dim: int, page_count: int
) -> None:
    _run_page_first_staged_write_back_mha(layout, element_dim, page_count)


@pytest.mark.parametrize("layout", ["page_first"])
@pytest.mark.parametrize("element_dim", MLA_ELEMENT_DIMS)
@pytest.mark.parametrize("page_count", STAGED_WRITE_BACK_PAGE_COUNTS)
def test_hicache_page_first_staged_write_back_mla(
    layout: str, element_dim: int, page_count: int
) -> None:
    _run_page_first_staged_write_back_mla(layout, element_dim, page_count)


def test_hicache_page_first_staged_write_back_mha_staged_only_alignment() -> None:
    _run_page_first_staged_write_back_mha("page_first", 72, 65)


def test_hicache_page_first_staged_write_back_mla_staged_only_alignment() -> None:
    _run_page_first_staged_write_back_mla("page_first", 72, 65)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
