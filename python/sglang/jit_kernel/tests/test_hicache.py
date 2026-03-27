import sys

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    ALLOC_MEMORY_FUNCS,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    alloc_with_pin_memory,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

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


def _token_indices_for_pages(
    pages: torch.Tensor, page_size: int = PAGE_SIZE, device: str = DEVICE
) -> torch.Tensor:
    parts = [
        torch.arange(
            int(page) * page_size,
            (int(page) + 1) * page_size,
            device=device,
            dtype=torch.int64,
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
    assert (
        host_pool.can_use_jit
    ), f"Expected JIT HiCache kernel for MHA dim={element_dim}"

    for layer_id in range(NUM_LAYERS):
        _copy_tensor_with_offset(device_pool.k_buffer[layer_id], layer_id)
        _copy_tensor_with_offset(device_pool.v_buffer[layer_id], layer_id + 100)

    device_pages = torch.tensor([1, 2, 3], device=DEVICE, dtype=torch.int64)
    host_pages = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int64)
    device_indices = _token_indices_for_pages(device_pages)
    host_indices = _token_indices_for_pages(host_pages)

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
    assert (
        host_pool.can_use_jit
    ), f"Expected JIT HiCache kernel for MLA dim={element_dim}"

    for layer_id in range(NUM_LAYERS):
        _copy_tensor_with_offset(device_pool.kv_buffer[layer_id], layer_id)

    device_pages = torch.tensor([1, 2, 3], device=DEVICE, dtype=torch.int64)
    host_pages = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int64)
    device_indices = _token_indices_for_pages(device_pages)
    host_indices = _token_indices_for_pages(host_pages)

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


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("element_dim", MHA_ELEMENT_DIMS)
def test_hicache_transfer_mha(layout: str, element_dim: int) -> None:
    _run_transfer_roundtrip_mha(layout, element_dim)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("element_dim", MLA_ELEMENT_DIMS)
def test_hicache_transfer_mla(layout: str, element_dim: int) -> None:
    _run_transfer_roundtrip_mla(layout, element_dim)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
