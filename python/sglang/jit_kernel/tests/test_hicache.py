import itertools

import pytest
import torch

from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
    transfer_hicache_all_layer,
    transfer_hicache_one_layer,
)

# element_dim must satisfy: element_dim * dtype_size % 128 == 0
# For bf16 (2 bytes), element_dim must be a multiple of 64
ELEMENT_DIMS = [64, 128, 256, 512]
BS_LIST = [1, 2, 7, 16, 128, 1024]
GPU_CACHE_SIZE = 4096
HOST_CACHE_SIZE = 8192
NUM_LAYERS_LIST = [1, 4]
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _gen_indices(size: int, pool_size: int, device="cuda") -> torch.Tensor:
    return torch.randperm(pool_size, device="cpu")[:size].to(device)


def _create_ptr_tensor(tensors) -> torch.Tensor:
    return torch.tensor(
        [t.data_ptr() for t in tensors],
        dtype=torch.uint64,
        device="cuda",
    )


# =============================================================================
# transfer_hicache_one_layer tests
# =============================================================================


@pytest.mark.parametrize(
    "batch_size,element_dim",
    list(itertools.product(BS_LIST, ELEMENT_DIMS)),
)
def test_one_layer_gpu_to_gpu(batch_size: int, element_dim: int) -> None:
    """GPU -> GPU single-layer KV cache transfer."""
    k_cache_src = torch.randn(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
    v_cache_src = torch.randn(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
    k_cache_dst = torch.zeros(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
    v_cache_dst = torch.zeros(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)

    indices_src = _gen_indices(batch_size, GPU_CACHE_SIZE)
    indices_dst = _gen_indices(batch_size, GPU_CACHE_SIZE)

    transfer_hicache_one_layer(
        k_cache_dst, v_cache_dst, indices_dst,
        k_cache_src, v_cache_src, indices_src,
        element_dim=element_dim,
    )
    torch.cuda.synchronize()

    assert torch.equal(k_cache_dst[indices_dst], k_cache_src[indices_src])
    assert torch.equal(v_cache_dst[indices_dst], v_cache_src[indices_src])


@pytest.mark.parametrize(
    "batch_size,element_dim",
    list(itertools.product(BS_LIST, ELEMENT_DIMS)),
)
def test_one_layer_host_to_gpu(batch_size: int, element_dim: int) -> None:
    """Host (pinned CPU) -> GPU single-layer KV cache transfer."""
    k_cache_src = torch.randn(HOST_CACHE_SIZE, element_dim, dtype=DTYPE).pin_memory()
    v_cache_src = torch.randn(HOST_CACHE_SIZE, element_dim, dtype=DTYPE).pin_memory()
    k_cache_dst = torch.zeros(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
    v_cache_dst = torch.zeros(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)

    indices_src = _gen_indices(batch_size, HOST_CACHE_SIZE)
    indices_dst = _gen_indices(batch_size, GPU_CACHE_SIZE)

    transfer_hicache_one_layer(
        k_cache_dst, v_cache_dst, indices_dst,
        k_cache_src, v_cache_src, indices_src,
        element_dim=element_dim,
    )
    torch.cuda.synchronize()

    expected_k = k_cache_src[indices_src.cpu()].to(DEVICE)
    expected_v = v_cache_src[indices_src.cpu()].to(DEVICE)
    assert torch.equal(k_cache_dst[indices_dst], expected_k)
    assert torch.equal(v_cache_dst[indices_dst], expected_v)


@pytest.mark.parametrize(
    "batch_size,element_dim",
    list(itertools.product(BS_LIST, ELEMENT_DIMS)),
)
def test_one_layer_gpu_to_host(batch_size: int, element_dim: int) -> None:
    """GPU -> Host (pinned CPU) single-layer KV cache transfer."""
    k_cache_src = torch.randn(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
    v_cache_src = torch.randn(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
    k_cache_dst = torch.zeros(HOST_CACHE_SIZE, element_dim, dtype=DTYPE).pin_memory()
    v_cache_dst = torch.zeros(HOST_CACHE_SIZE, element_dim, dtype=DTYPE).pin_memory()

    indices_src = _gen_indices(batch_size, GPU_CACHE_SIZE)
    indices_dst = _gen_indices(batch_size, HOST_CACHE_SIZE)

    transfer_hicache_one_layer(
        k_cache_dst, v_cache_dst, indices_dst,
        k_cache_src, v_cache_src, indices_src,
        element_dim=element_dim,
    )
    torch.cuda.synchronize()

    expected_k = k_cache_src[indices_src].cpu()
    expected_v = v_cache_src[indices_src].cpu()
    assert torch.equal(k_cache_dst[indices_dst.cpu()], expected_k)
    assert torch.equal(v_cache_dst[indices_dst.cpu()], expected_v)


# =============================================================================
# transfer_hicache_all_layer tests
# =============================================================================


@pytest.mark.parametrize(
    "batch_size,element_dim,num_layers",
    list(itertools.product(BS_LIST, ELEMENT_DIMS, NUM_LAYERS_LIST)),
)
def test_all_layer_gpu_to_host(
    batch_size: int, element_dim: int, num_layers: int
) -> None:
    """GPU -> Host (pinned CPU) all-layer KV cache transfer."""
    k_caches_src = [
        torch.randn(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]
    v_caches_src = [
        torch.randn(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]
    k_caches_dst = [
        torch.zeros(HOST_CACHE_SIZE, element_dim, dtype=DTYPE).pin_memory()
        for _ in range(num_layers)
    ]
    v_caches_dst = [
        torch.zeros(HOST_CACHE_SIZE, element_dim, dtype=DTYPE).pin_memory()
        for _ in range(num_layers)
    ]

    indices_src = _gen_indices(batch_size, GPU_CACHE_SIZE)
    indices_dst = _gen_indices(batch_size, HOST_CACHE_SIZE)

    stride_bytes = element_dim * DTYPE.itemsize

    transfer_hicache_all_layer(
        _create_ptr_tensor(k_caches_dst),
        _create_ptr_tensor(v_caches_dst),
        indices_dst,
        _create_ptr_tensor(k_caches_src),
        _create_ptr_tensor(v_caches_src),
        indices_src,
        kv_cache_src_stride_bytes=stride_bytes,
        kv_cache_dst_stride_bytes=stride_bytes,
        element_size=stride_bytes,
    )
    torch.cuda.synchronize()

    for layer in range(num_layers):
        expected_k = k_caches_src[layer][indices_src].cpu()
        expected_v = v_caches_src[layer][indices_src].cpu()
        assert torch.equal(k_caches_dst[layer][indices_dst.cpu()], expected_k), (
            f"K mismatch at layer {layer}"
        )
        assert torch.equal(v_caches_dst[layer][indices_dst.cpu()], expected_v), (
            f"V mismatch at layer {layer}"
        )


@pytest.mark.parametrize(
    "batch_size,element_dim,num_layers",
    list(itertools.product(BS_LIST, ELEMENT_DIMS, NUM_LAYERS_LIST)),
)
def test_all_layer_host_to_gpu(
    batch_size: int, element_dim: int, num_layers: int
) -> None:
    """Host (pinned CPU) -> GPU all-layer KV cache transfer."""
    k_caches_src = [
        torch.randn(HOST_CACHE_SIZE, element_dim, dtype=DTYPE).pin_memory()
        for _ in range(num_layers)
    ]
    v_caches_src = [
        torch.randn(HOST_CACHE_SIZE, element_dim, dtype=DTYPE).pin_memory()
        for _ in range(num_layers)
    ]
    k_caches_dst = [
        torch.zeros(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]
    v_caches_dst = [
        torch.zeros(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]

    indices_src = _gen_indices(batch_size, HOST_CACHE_SIZE)
    indices_dst = _gen_indices(batch_size, GPU_CACHE_SIZE)

    stride_bytes = element_dim * DTYPE.itemsize

    transfer_hicache_all_layer(
        _create_ptr_tensor(k_caches_dst),
        _create_ptr_tensor(v_caches_dst),
        indices_dst,
        _create_ptr_tensor(k_caches_src),
        _create_ptr_tensor(v_caches_src),
        indices_src,
        kv_cache_src_stride_bytes=stride_bytes,
        kv_cache_dst_stride_bytes=stride_bytes,
        element_size=stride_bytes,
    )
    torch.cuda.synchronize()

    for layer in range(num_layers):
        expected_k = k_caches_src[layer][indices_src.cpu()].to(DEVICE)
        expected_v = v_caches_src[layer][indices_src.cpu()].to(DEVICE)
        assert torch.equal(k_caches_dst[layer][indices_dst], expected_k), (
            f"K mismatch at layer {layer}"
        )
        assert torch.equal(v_caches_dst[layer][indices_dst], expected_v), (
            f"V mismatch at layer {layer}"
        )


# =============================================================================
# Index dtype tests (int32 vs int64)
# =============================================================================


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_one_layer_index_dtypes(index_dtype: torch.dtype) -> None:
    """Ensure both int32 and int64 index tensors work correctly."""
    batch_size, element_dim = 64, 128
    k_cache_src = torch.randn(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
    v_cache_src = torch.randn(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
    k_cache_dst = torch.zeros(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
    v_cache_dst = torch.zeros(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)

    indices_src = _gen_indices(batch_size, GPU_CACHE_SIZE).to(index_dtype)
    indices_dst = _gen_indices(batch_size, GPU_CACHE_SIZE).to(index_dtype)

    transfer_hicache_one_layer(
        k_cache_dst, v_cache_dst, indices_dst,
        k_cache_src, v_cache_src, indices_src,
        element_dim=element_dim,
    )
    torch.cuda.synchronize()

    assert torch.equal(k_cache_dst[indices_dst.long()], k_cache_src[indices_src.long()])
    assert torch.equal(v_cache_dst[indices_dst.long()], v_cache_src[indices_src.long()])


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_all_layer_index_dtypes(index_dtype: torch.dtype) -> None:
    """Ensure both int32 and int64 index tensors work with all-layer transfer."""
    batch_size, element_dim, num_layers = 32, 128, 2
    k_caches_src = [
        torch.randn(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]
    v_caches_src = [
        torch.randn(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]
    k_caches_dst = [
        torch.zeros(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]
    v_caches_dst = [
        torch.zeros(GPU_CACHE_SIZE, element_dim, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]

    indices_src = _gen_indices(batch_size, GPU_CACHE_SIZE).to(index_dtype)
    indices_dst = _gen_indices(batch_size, GPU_CACHE_SIZE).to(index_dtype)
    stride_bytes = element_dim * DTYPE.itemsize

    transfer_hicache_all_layer(
        _create_ptr_tensor(k_caches_dst),
        _create_ptr_tensor(v_caches_dst),
        indices_dst,
        _create_ptr_tensor(k_caches_src),
        _create_ptr_tensor(v_caches_src),
        indices_src,
        kv_cache_src_stride_bytes=stride_bytes,
        kv_cache_dst_stride_bytes=stride_bytes,
        element_size=stride_bytes,
    )
    torch.cuda.synchronize()

    for layer in range(num_layers):
        assert torch.equal(
            k_caches_dst[layer][indices_dst.long()],
            k_caches_src[layer][indices_src.long()],
        ), f"K mismatch at layer {layer}"
        assert torch.equal(
            v_caches_dst[layer][indices_dst.long()],
            v_caches_src[layer][indices_src.long()],
        ), f"V mismatch at layer {layer}"


# =============================================================================
# Different src/dst stride tests
# =============================================================================


def test_all_layer_different_strides() -> None:
    """Test all-layer transfer where src and dst have different strides."""
    batch_size, num_layers = 32, 2
    element_dim = 128
    # src uses a wider allocation (extra padding columns)
    src_total_cols = element_dim + 64  # stride differs from element_dim
    dst_total_cols = element_dim

    k_caches_src = [
        torch.randn(GPU_CACHE_SIZE, src_total_cols, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]
    v_caches_src = [
        torch.randn(GPU_CACHE_SIZE, src_total_cols, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]
    k_caches_dst = [
        torch.zeros(GPU_CACHE_SIZE, dst_total_cols, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]
    v_caches_dst = [
        torch.zeros(GPU_CACHE_SIZE, dst_total_cols, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]

    indices_src = _gen_indices(batch_size, GPU_CACHE_SIZE)
    indices_dst = _gen_indices(batch_size, GPU_CACHE_SIZE)

    src_stride_bytes = src_total_cols * DTYPE.itemsize
    dst_stride_bytes = dst_total_cols * DTYPE.itemsize
    element_bytes = element_dim * DTYPE.itemsize

    transfer_hicache_all_layer(
        _create_ptr_tensor(k_caches_dst),
        _create_ptr_tensor(v_caches_dst),
        indices_dst,
        _create_ptr_tensor(k_caches_src),
        _create_ptr_tensor(v_caches_src),
        indices_src,
        kv_cache_src_stride_bytes=src_stride_bytes,
        kv_cache_dst_stride_bytes=dst_stride_bytes,
        element_size=element_bytes,
    )
    torch.cuda.synchronize()

    for layer in range(num_layers):
        # Only the first element_dim columns should be copied
        expected_k = k_caches_src[layer][indices_src, :element_dim]
        expected_v = v_caches_src[layer][indices_src, :element_dim]
        assert torch.equal(k_caches_dst[layer][indices_dst], expected_k), (
            f"K mismatch at layer {layer}"
        )
        assert torch.equal(v_caches_dst[layer][indices_dst], expected_v), (
            f"V mismatch at layer {layer}"
        )


# =============================================================================
# can_use_hicache_jit_kernel tests
# =============================================================================


def test_can_use_valid_sizes() -> None:
    """Verify can_use_hicache_jit_kernel accepts valid element sizes."""
    for dim in ELEMENT_DIMS:
        element_size = dim * DTYPE.itemsize
        assert can_use_hicache_jit_kernel(element_size=element_size)


def test_can_use_rejects_invalid_sizes() -> None:
    """Verify can_use_hicache_jit_kernel rejects non-128-byte-aligned sizes."""
    assert not can_use_hicache_jit_kernel(element_size=100)
    assert not can_use_hicache_jit_kernel(element_size=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
