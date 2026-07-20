"""Unit tests for the Mamba JIT transfer kernel.

Verifies kernel backup (D2H) and load (H2D) correctness for
``MambaPoolHost`` via the ``io_backend='kernel'`` path, across both
supported layouts and multiple index scenarios.
"""

import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba transfer kernel tests require CUDA."
)

DEVICE = "cuda"
NUM_LAYERS = 3
SIZE = 16
TEMPORAL_SHAPE = (16, 128)  # 16*128*2 = 4096 bytes (fp16), 16-byte aligned
CONV_SHAPE = (4, 128)  # 4*128*2 = 1024 bytes (fp16), 16-byte aligned
DTYPES = [torch.float16, torch.bfloat16]
LAYOUTS = ["page_first", "page_first_direct"]


def make_device_pool(dtype, device=DEVICE):
    """Create a minimal mock device pool that MambaPoolHost can use."""
    temporal = torch.zeros(
        (NUM_LAYERS, SIZE) + TEMPORAL_SHAPE, dtype=dtype, device=device
    )
    conv = [torch.zeros((NUM_LAYERS, SIZE) + CONV_SHAPE, dtype=dtype, device=device)]

    mamba_cache = SimpleNamespace(temporal=temporal, conv=conv)
    return SimpleNamespace(
        mamba_cache=mamba_cache,
        size=SIZE,
        device=device,
    )


def make_host_pool(dtype, layout):
    """Create a MambaPoolHost bypassing __init__, manually setting attributes.

    NOTE: If MambaPoolHost adds/renames attributes accessed by
    backup_from_device_all_layer or load_to_device_per_layer, this mock
    must be updated to match. See assert_host_mock_complete() below.
    """
    host = MambaPoolHost.__new__(MambaPoolHost)
    host.layout = layout
    host.page_size = 1
    host.page_num = SIZE
    host.size = SIZE
    host.pin_memory = True
    host.device = "cpu"
    host.num_mamba_layers = NUM_LAYERS
    host.conv_state_shapes = [CONV_SHAPE]
    host.temporal_state_shape = TEMPORAL_SHAPE
    host.temporal_state_elem_size = int(torch.prod(torch.tensor(TEMPORAL_SHAPE)).item())
    host.conv_state_elem_sizes = [int(torch.prod(torch.tensor(CONV_SHAPE)).item())]
    host.conv_dtype = dtype
    host.temporal_dtype = dtype
    host.dtype = dtype
    host.size_per_token = host.get_size_per_token()

    # Allocate host buffers (page_first layout)
    temporal_dims = (SIZE, NUM_LAYERS, 1) + TEMPORAL_SHAPE
    host.temporal_buffer = torch.zeros(temporal_dims, dtype=dtype).pin_memory()

    host.conv_buffer = []
    conv_dims = (SIZE, NUM_LAYERS, 1) + CONV_SHAPE
    host.conv_buffer.append(torch.zeros(conv_dims, dtype=dtype).pin_memory())

    # Staging buffers and JIT flags
    host.temporal_staging_buffer = None
    host.conv_staging_buffers = [None]
    host.can_use_write_back_jit = True
    host._temporal_can_use_jit = False
    host._conv_can_use_jit = [False]

    # Device pointers (needed for backup kernel path)
    device_pool = make_device_pool(dtype)
    host.device_pool = device_pool
    host.temporal_device_ptrs = torch.tensor(
        [device_pool.mamba_cache.temporal[i].data_ptr() for i in range(NUM_LAYERS)],
        dtype=torch.uint64,
        device=DEVICE,
    )
    host.conv_device_ptrs = [
        torch.tensor(
            [conv_state[i].data_ptr() for i in range(NUM_LAYERS)],
            dtype=torch.uint64,
            device=DEVICE,
        )
        for conv_state in device_pool.mamba_cache.conv
    ]

    host.lock = threading.RLock()
    host.clear()
    return host


def assert_host_mock_complete(host):
    """Sanity check: ensure mock covers attributes used by backup/load paths."""
    required = [
        "layout",
        "page_size",
        "page_num",
        "size",
        "pin_memory",
        "device",
        "num_mamba_layers",
        "conv_state_shapes",
        "temporal_state_shape",
        "temporal_state_elem_size",
        "conv_state_elem_sizes",
        "conv_dtype",
        "temporal_dtype",
        "dtype",
        "size_per_token",
        "temporal_buffer",
        "conv_buffer",
        "temporal_staging_buffer",
        "conv_staging_buffers",
        "can_use_write_back_jit",
        "_temporal_can_use_jit",
        "_conv_can_use_jit",
        "device_pool",
        "temporal_device_ptrs",
        "conv_device_ptrs",
        "lock",
    ]
    missing = [attr for attr in required if not hasattr(host, attr)]
    assert not missing, f"Mock MambaPoolHost missing attributes: {missing}"


def fill_device_data(device_pool, dtype):
    """Fill device temporal and conv states with deterministic data."""
    for layer_id in range(NUM_LAYERS):
        offset = layer_id * 1000
        data = torch.arange(
            device_pool.mamba_cache.temporal[layer_id].numel(),
            device=DEVICE,
            dtype=dtype,
        )
        device_pool.mamba_cache.temporal[layer_id].copy_(
            (data + offset).view_as(device_pool.mamba_cache.temporal[layer_id])
        )
        for conv_idx in range(len(device_pool.mamba_cache.conv)):
            conv_data = torch.arange(
                device_pool.mamba_cache.conv[conv_idx][layer_id].numel(),
                device=DEVICE,
                dtype=dtype,
            )
            device_pool.mamba_cache.conv[conv_idx][layer_id].copy_(
                (conv_data + offset + conv_idx * 500).view_as(
                    device_pool.mamba_cache.conv[conv_idx][layer_id]
                )
            )


def assert_host_matches_device(host, device_pool, host_indices, device_indices):
    """Verify host backup data matches device source data."""
    for layer_id in range(NUM_LAYERS):
        # Temporal
        host_temporal = host.temporal_buffer[host_indices, layer_id, 0].cpu()
        dev_temporal = device_pool.mamba_cache.temporal[layer_id][device_indices].cpu()
        torch.testing.assert_close(host_temporal, dev_temporal)

        # Conv
        for conv_idx in range(len(host.conv_buffer)):
            host_conv = host.conv_buffer[conv_idx][host_indices, layer_id, 0].cpu()
            dev_conv = device_pool.mamba_cache.conv[conv_idx][layer_id][
                device_indices
            ].cpu()
            torch.testing.assert_close(host_conv, dev_conv)


def assert_device_matches_host(host, device_pool, host_indices, device_indices):
    """Verify device load data matches host source data."""
    for layer_id in range(NUM_LAYERS):
        # Temporal
        host_temporal = host.temporal_buffer[host_indices, layer_id, 0].to(DEVICE)
        dev_temporal = device_pool.mamba_cache.temporal[layer_id][device_indices]
        torch.testing.assert_close(dev_temporal, host_temporal)

        # Conv
        for conv_idx in range(len(host.conv_buffer)):
            host_conv = host.conv_buffer[conv_idx][host_indices, layer_id, 0].to(DEVICE)
            dev_conv = device_pool.mamba_cache.conv[conv_idx][layer_id][device_indices]
            torch.testing.assert_close(dev_conv, host_conv)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mamba_kernel_backup_load_roundtrip(dtype, layout):
    """Test D2H backup + H2D load roundtrip with io_backend='kernel'."""
    host = make_host_pool(dtype, layout)
    assert_host_mock_complete(host)
    device_pool = host.device_pool

    # Fill device with known data
    fill_device_data(device_pool, dtype)

    # Use a few indices for the test
    device_indices = torch.tensor([1, 5, 10], dtype=torch.int64, device=DEVICE)
    host_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
    load_indices = torch.tensor([3, 7, 12], dtype=torch.int64, device=DEVICE)

    # --- Backup: device -> host (kernel) ---
    host.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()
    assert_host_matches_device(host, device_pool, host_indices, device_indices)

    # --- Clear device buffers ---
    for layer_id in range(NUM_LAYERS):
        device_pool.mamba_cache.temporal[layer_id].zero_()
        for conv_idx in range(len(device_pool.mamba_cache.conv)):
            device_pool.mamba_cache.conv[conv_idx][layer_id].zero_()

    # --- Load: host -> device (kernel), per layer ---
    for layer_id in range(NUM_LAYERS):
        host.load_to_device_per_layer(
            device_pool,
            host_indices,
            load_indices,
            layer_id,
            io_backend="kernel",
        )
    torch.cuda.synchronize()
    assert_device_matches_host(host, device_pool, host_indices, load_indices)

    # Verify non-target positions remain zero (catch kernel writing wrong indices)
    all_indices = set(range(SIZE))
    target_set = set(load_indices.tolist())
    untouched = sorted(all_indices - target_set)
    if untouched:
        untouched_t = torch.tensor(untouched, dtype=torch.int64, device=DEVICE)
        for layer_id in range(NUM_LAYERS):
            assert (
                device_pool.mamba_cache.temporal[layer_id][untouched_t].abs().max() == 0
            )
            for conv_idx in range(len(device_pool.mamba_cache.conv)):
                assert (
                    device_pool.mamba_cache.conv[conv_idx][layer_id][untouched_t]
                    .abs()
                    .max()
                    == 0
                )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mamba_kernel_empty_indices(dtype, layout):
    """Test that empty indices are handled gracefully (no crash)."""
    host = make_host_pool(dtype, layout)
    device_pool = host.device_pool
    fill_device_data(device_pool, dtype)

    empty_device = torch.tensor([], dtype=torch.int64, device=DEVICE)
    empty_host = torch.tensor([], dtype=torch.int64)

    host.backup_from_device_all_layer(
        device_pool, empty_host, empty_device, io_backend="kernel"
    )
    torch.cuda.synchronize()
    # Host buffers should remain all zeros
    assert host.temporal_buffer.abs().max() == 0

    for layer_id in range(NUM_LAYERS):
        host.load_to_device_per_layer(
            device_pool, empty_host, empty_device, layer_id, io_backend="kernel"
        )
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mamba_kernel_single_item(dtype, layout):
    """Test single item backup + load."""
    host = make_host_pool(dtype, layout)
    device_pool = host.device_pool
    fill_device_data(device_pool, dtype)

    device_indices = torch.tensor([7], dtype=torch.int64, device=DEVICE)
    host_indices = torch.tensor([3], dtype=torch.int64)
    load_indices = torch.tensor([9], dtype=torch.int64, device=DEVICE)

    host.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()
    assert_host_matches_device(host, device_pool, host_indices, device_indices)

    for layer_id in range(NUM_LAYERS):
        device_pool.mamba_cache.temporal[layer_id].zero_()
        for conv_idx in range(len(device_pool.mamba_cache.conv)):
            device_pool.mamba_cache.conv[conv_idx][layer_id].zero_()

    for layer_id in range(NUM_LAYERS):
        host.load_to_device_per_layer(
            device_pool, host_indices, load_indices, layer_id, io_backend="kernel"
        )
    torch.cuda.synchronize()
    assert_device_matches_host(host, device_pool, host_indices, load_indices)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mamba_kernel_full_indices(dtype, layout):
    """Test full-size backup + load (all SIZE items)."""
    host = make_host_pool(dtype, layout)
    device_pool = host.device_pool
    fill_device_data(device_pool, dtype)

    device_indices = torch.arange(SIZE, dtype=torch.int64, device=DEVICE)
    host_indices = torch.arange(SIZE, dtype=torch.int64)
    load_indices = torch.arange(SIZE, dtype=torch.int64, device=DEVICE)

    host.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()
    assert_host_matches_device(host, device_pool, host_indices, device_indices)

    for layer_id in range(NUM_LAYERS):
        device_pool.mamba_cache.temporal[layer_id].zero_()
        for conv_idx in range(len(device_pool.mamba_cache.conv)):
            device_pool.mamba_cache.conv[conv_idx][layer_id].zero_()

    for layer_id in range(NUM_LAYERS):
        host.load_to_device_per_layer(
            device_pool, host_indices, load_indices, layer_id, io_backend="kernel"
        )
    torch.cuda.synchronize()
    assert_device_matches_host(host, device_pool, host_indices, load_indices)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
