import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.memory_pool_host import AsymmetricMHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="base-b-kernel-unit-1-gpu-large")

# These tests use AsymmetricMHATokenToKVPoolHost methods and let that class call
# the real sgl-kernel transfer ops. The asymmetric host pool is kernel-only;
# direct/page_first_direct is intentionally rejected in the CPU dispatch tests.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="asymmetric host-pool tests require CUDA."
)

DEVICE = "cuda"
PAGE_SIZE = 16
NUM_LAYERS = 3
TOTAL_ITEMS = PAGE_SIZE * 8
HEAD_NUM = 4
K_HEAD_DIM = 192
V_HEAD_DIM = 128
DTYPES = [torch.float16, torch.bfloat16]


def token_indices_for_pages(pages, page_size=PAGE_SIZE, device=None):
    indices = torch.cat(
        [
            torch.arange(
                int(page) * page_size,
                (int(page) + 1) * page_size,
                dtype=torch.int64,
            )
            for page in pages.tolist()
        ]
    )
    return indices if device is None else indices.to(device)


def fill_with_offset(tensor, offset):
    data = torch.arange(tensor.numel(), device=tensor.device, dtype=tensor.dtype)
    tensor.copy_((data + offset).view_as(tensor))


def make_host_pool(dtype):
    host = AsymmetricMHATokenToKVPoolHost.__new__(AsymmetricMHATokenToKVPoolHost)
    host.layout = "page_first"
    host.page_size = PAGE_SIZE
    host.layer_num = NUM_LAYERS
    host.head_num = HEAD_NUM
    host.head_dim = K_HEAD_DIM
    host.v_head_dim = V_HEAD_DIM
    host.dtype = dtype
    host.kv_buffer = (
        torch.zeros(
            TOTAL_ITEMS, NUM_LAYERS, HEAD_NUM, K_HEAD_DIM, dtype=dtype
        ).pin_memory(),
        torch.zeros(
            TOTAL_ITEMS, NUM_LAYERS, HEAD_NUM, V_HEAD_DIM, dtype=dtype
        ).pin_memory(),
    )
    return host


def make_device_pool(dtype):
    k_buffer = [
        torch.empty(TOTAL_ITEMS, HEAD_NUM, K_HEAD_DIM, dtype=dtype, device=DEVICE)
        for _ in range(NUM_LAYERS)
    ]
    v_buffer = [
        torch.empty(TOTAL_ITEMS, HEAD_NUM, V_HEAD_DIM, dtype=dtype, device=DEVICE)
        for _ in range(NUM_LAYERS)
    ]
    for layer_id in range(NUM_LAYERS):
        fill_with_offset(k_buffer[layer_id], layer_id * 1000)
        fill_with_offset(v_buffer[layer_id], layer_id * 1000 + 100)

    return SimpleNamespace(
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_data_ptrs=torch.tensor(
            [x.data_ptr() for x in k_buffer], dtype=torch.uint64, device=DEVICE
        ),
        v_data_ptrs=torch.tensor(
            [x.data_ptr() for x in v_buffer], dtype=torch.uint64, device=DEVICE
        ),
    )


def assert_backup_matches_device(host, device_pool, host_indices_host, device_indices):
    for layer_id in range(NUM_LAYERS):
        torch.testing.assert_close(
            host.k_buffer[host_indices_host, layer_id],
            device_pool.k_buffer[layer_id][device_indices].cpu(),
        )
        torch.testing.assert_close(
            host.v_buffer[host_indices_host, layer_id],
            device_pool.v_buffer[layer_id][device_indices].cpu(),
        )


def assert_load_matches_host(host, device_pool, host_indices_host, load_indices):
    for layer_id in range(NUM_LAYERS):
        torch.testing.assert_close(
            device_pool.k_buffer[layer_id][load_indices],
            host.k_buffer[host_indices_host, layer_id].to(DEVICE),
        )
        torch.testing.assert_close(
            device_pool.v_buffer[layer_id][load_indices],
            host.v_buffer[host_indices_host, layer_id].to(DEVICE),
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_asymmetric_mha_kernel_page_first_roundtrip(dtype):
    # Covers D2H backup + H2D load through AsymmetricMHATokenToKVPoolHost using
    # MiMoV2's real K/V head dims and the real MLA single-buffer kernels.
    host = make_host_pool(dtype)
    device_pool = make_device_pool(dtype)

    device_pages = torch.tensor([1, 2, 3], dtype=torch.int64)
    host_pages = torch.tensor([0, 1, 2], dtype=torch.int64)
    load_pages = torch.tensor([4, 5, 6], dtype=torch.int64)
    device_indices_host = token_indices_for_pages(device_pages)
    host_indices_host = token_indices_for_pages(host_pages)
    load_indices_host = token_indices_for_pages(load_pages)
    device_indices = device_indices_host.to(DEVICE)
    host_indices = host_indices_host.to(DEVICE)
    load_indices = load_indices_host.to(DEVICE)

    host.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()
    assert_backup_matches_device(
        host, device_pool, host_indices_host, device_indices_host
    )

    for layer_id in range(NUM_LAYERS):
        device_pool.k_buffer[layer_id].zero_()
        device_pool.v_buffer[layer_id].zero_()
        host.load_to_device_per_layer(
            device_pool, host_indices, load_indices, layer_id, io_backend="kernel"
        )
    torch.cuda.synchronize()
    assert_load_matches_host(host, device_pool, host_indices_host, load_indices_host)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
