"""Tests for HiCache JIT kernel page_first layout support.

Tests transfer operations between page_first (host) and layer_first (device) layouts.
"""

import pytest
import torch

from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
    transfer_hicache_all_layer_lf_pf,
    transfer_hicache_one_layer_pf_lf,
)


@pytest.fixture
def setup_params():
    """Common test parameters."""
    return {
        "batch_size": 16,
        "num_layers": 8,
        "head_num": 8,
        "head_dim": 128,
        "gpu_cache_size": 1024,
        "host_cache_size": 4096,
        "dtype": torch.float16,
    }


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else pytest.skip("CUDA not available")


def test_can_use_hicache_jit_kernel(setup_params):
    """Test that JIT kernel can be loaded for typical element sizes."""
    element_size = setup_params["head_num"] * setup_params["head_dim"] * 2  # fp16
    assert can_use_hicache_jit_kernel(element_size=element_size)


def test_transfer_one_layer_pf_lf(setup_params, device):
    """Test single layer transfer from page_first (host) to layer_first (device)."""
    batch_size = setup_params["batch_size"]
    num_layers = setup_params["num_layers"]
    head_num = setup_params["head_num"]
    head_dim = setup_params["head_dim"]
    gpu_cache_size = setup_params["gpu_cache_size"]
    host_cache_size = setup_params["host_cache_size"]
    dtype = setup_params["dtype"]

    element_size = head_num * head_dim * dtype.itemsize
    if not can_use_hicache_jit_kernel(element_size=element_size):
        pytest.skip("JIT kernel not available for this element size")

    # Create page_first layout host cache: [token, layer, head, dim]
    k_cache_host = torch.randn(
        (host_cache_size, num_layers, head_num, head_dim),
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    v_cache_host = torch.randn(
        (host_cache_size, num_layers, head_num, head_dim),
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )

    # Create layer_first layout device cache for one layer: [token, head, dim]
    k_cache_device = torch.zeros(
        (gpu_cache_size, head_num, head_dim), dtype=dtype, device=device
    )
    v_cache_device = torch.zeros(
        (gpu_cache_size, head_num, head_dim), dtype=dtype, device=device
    )

    # Create indices
    host_indices = torch.randperm(host_cache_size, device=device)[:batch_size]
    device_indices = torch.randperm(gpu_cache_size, device=device)[:batch_size]

    # Test for each layer
    for layer_id in range(num_layers):
        k_cache_device.zero_()
        v_cache_device.zero_()

        src_layout_dim = num_layers * head_num * head_dim * dtype.itemsize

        transfer_hicache_one_layer_pf_lf(
            k_cache_dst=k_cache_device,
            v_cache_dst=v_cache_device,
            indices_dst=device_indices,
            k_cache_src=k_cache_host,
            v_cache_src=v_cache_host,
            indices_src=host_indices,
            layer_id=layer_id,
            src_layout_dim=src_layout_dim,
            element_dim=head_num * head_dim,
        )

        torch.cuda.synchronize()

        # Verify transfer correctness
        host_indices_cpu = host_indices.cpu()
        device_indices_cpu = device_indices.cpu()

        for i in range(batch_size):
            src_idx = host_indices_cpu[i].item()
            dst_idx = device_indices_cpu[i].item()

            expected_k = k_cache_host[src_idx, layer_id, :, :]
            actual_k = k_cache_device[dst_idx, :, :].cpu()
            assert torch.allclose(
                expected_k, actual_k, atol=1e-5
            ), f"K mismatch at layer {layer_id}, batch {i}"

            expected_v = v_cache_host[src_idx, layer_id, :, :]
            actual_v = v_cache_device[dst_idx, :, :].cpu()
            assert torch.allclose(
                expected_v, actual_v, atol=1e-5
            ), f"V mismatch at layer {layer_id}, batch {i}"


def test_transfer_all_layer_lf_pf(setup_params, device):
    """Test all layer transfer from layer_first (device) to page_first (host)."""
    batch_size = setup_params["batch_size"]
    num_layers = setup_params["num_layers"]
    head_num = setup_params["head_num"]
    head_dim = setup_params["head_dim"]
    gpu_cache_size = setup_params["gpu_cache_size"]
    host_cache_size = setup_params["host_cache_size"]
    dtype = setup_params["dtype"]

    element_size = head_num * head_dim * dtype.itemsize
    if not can_use_hicache_jit_kernel(element_size=element_size):
        pytest.skip("JIT kernel not available for this element size")

    # Create layer_first layout device cache: [layer, token, head, dim]
    k_caches_device = torch.randn(
        (num_layers, gpu_cache_size, head_num, head_dim), dtype=dtype, device=device
    )
    v_caches_device = torch.randn(
        (num_layers, gpu_cache_size, head_num, head_dim), dtype=dtype, device=device
    )

    # Create page_first layout host cache: [token, layer, head, dim]
    k_cache_host = torch.zeros(
        (host_cache_size, num_layers, head_num, head_dim),
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    v_cache_host = torch.zeros(
        (host_cache_size, num_layers, head_num, head_dim),
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )

    # Create pointer tensors
    k_ptrs = torch.tensor(
        [k_caches_device[i].data_ptr() for i in range(num_layers)],
        dtype=torch.uint64,
        device=device,
    )
    v_ptrs = torch.tensor(
        [v_caches_device[i].data_ptr() for i in range(num_layers)],
        dtype=torch.uint64,
        device=device,
    )

    # Create indices
    device_indices = torch.randperm(gpu_cache_size, device=device)[:batch_size]
    host_indices = torch.randperm(host_cache_size, device=device)[:batch_size]

    dst_layout_dim = num_layers * head_num * head_dim * dtype.itemsize

    transfer_hicache_all_layer_lf_pf(
        k_cache_dst=k_cache_host,
        v_cache_dst=v_cache_host,
        indices_dst=host_indices,
        k_ptr_src=k_ptrs,
        v_ptr_src=v_ptrs,
        indices_src=device_indices,
        kv_cache_src_stride_bytes=head_num * head_dim * dtype.itemsize,
        dst_layout_dim=dst_layout_dim,
        element_size=element_size,
    )

    torch.cuda.synchronize()

    # Verify transfer correctness
    device_indices_cpu = device_indices.cpu()
    host_indices_cpu = host_indices.cpu()

    for i in range(batch_size):
        src_idx = device_indices_cpu[i].item()
        dst_idx = host_indices_cpu[i].item()

        for layer_id in range(num_layers):
            expected_k = k_caches_device[layer_id, src_idx, :, :].cpu()
            actual_k = k_cache_host[dst_idx, layer_id, :, :]
            assert torch.allclose(
                expected_k, actual_k, atol=1e-5
            ), f"K mismatch at layer {layer_id}, batch {i}"

            expected_v = v_caches_device[layer_id, src_idx, :, :].cpu()
            actual_v = v_cache_host[dst_idx, layer_id, :, :]
            assert torch.allclose(
                expected_v, actual_v, atol=1e-5
            ), f"V mismatch at layer {layer_id}, batch {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
