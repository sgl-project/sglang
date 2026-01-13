"""
Tests for the sparse KV cache manager kernel.

The sparse cache manager handles:
1. Identifying cache hits (tokens already in GPU hot buffer)
2. Identifying cache misses and eviction candidates
3. Assigning GPU locations for top-k tokens
4. Preparing copy operations for CPU→GPU transfers
"""

import pytest
import torch


def reference_sparse_cache_manager(
    top_k_indices: torch.Tensor,
    hot_buffer_token_indices: torch.Tensor,
    hot_buffer_device_locations: torch.Tensor,
    cache_cpu_locations: torch.Tensor,
):
    """
    Reference implementation of sparse cache manager.

    Returns:
        top_k_device_locations: GPU locations for top_k tokens
        copy_src_cpu_locations: CPU locations to copy from
        copy_dst_gpu_locations: GPU locations to copy to
        copy_count: Number of copies needed
        updated_hot_buffer_token_indices: Updated hot buffer token indices
    """
    top_k_size = top_k_indices.numel()
    hot_buffer_size = hot_buffer_token_indices.numel()

    # Convert to CPU for reference computation
    top_k = top_k_indices.cpu().numpy()
    hot_tokens = hot_buffer_token_indices.cpu().numpy().copy()
    hot_locs = hot_buffer_device_locations.cpu().numpy()
    cpu_locs = cache_cpu_locations.cpu().numpy()

    top_k_device_locations = [-1] * top_k_size
    copy_src_cpu_locations = []
    copy_dst_gpu_locations = []

    # Build set of top_k tokens for fast lookup
    top_k_set = set(top_k)

    # Find hits and mark evictable slots
    hot_token_to_idx = {tok: i for i, tok in enumerate(hot_tokens)}
    evictable = []

    for i, hot_tok in enumerate(hot_tokens):
        if hot_tok not in top_k_set:
            evictable.append(i)

    # Process each top_k token
    eviction_ptr = 0
    for k_idx, token in enumerate(top_k):
        if token in hot_token_to_idx:
            # Hit - token is already in hot buffer
            hot_idx = hot_token_to_idx[token]
            top_k_device_locations[k_idx] = hot_locs[hot_idx]
        else:
            # Miss - need to evict and load
            if eviction_ptr < len(evictable):
                evict_idx = evictable[eviction_ptr]
                eviction_ptr += 1

                gpu_loc = hot_locs[evict_idx]
                cpu_loc = cpu_locs[token]

                top_k_device_locations[k_idx] = gpu_loc
                copy_src_cpu_locations.append(cpu_loc)
                copy_dst_gpu_locations.append(gpu_loc)

                # Update hot buffer
                hot_tokens[evict_idx] = token
                hot_token_to_idx[token] = evict_idx

    copy_count = len(copy_src_cpu_locations)

    return (
        torch.tensor(top_k_device_locations, dtype=torch.int64),
        torch.tensor(copy_src_cpu_locations, dtype=torch.int64),
        torch.tensor(copy_dst_gpu_locations, dtype=torch.int64),
        copy_count,
        torch.tensor(hot_tokens, dtype=torch.int64),
    )


@pytest.mark.parametrize("top_k_size", [4, 16, 64, 128, 256])
@pytest.mark.parametrize("hot_buffer_size", [32, 128, 256, 512, 1024])
@pytest.mark.parametrize("total_tokens", [1024, 4096])
@pytest.mark.parametrize("hit_ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_sparse_cache_manager_fused(
    top_k_size: int,
    hot_buffer_size: int,
    total_tokens: int,
    hit_ratio: float,
):
    """Test the fused sparse cache manager kernel."""
    if top_k_size > hot_buffer_size:
        pytest.skip("top_k_size cannot exceed hot_buffer_size")
    if top_k_size > 256 or hot_buffer_size > 1024:
        pytest.skip("Fused kernel only supports top_k <= 256, hot_buffer <= 1024")

    torch.cuda.manual_seed(42)
    device = "cuda"

    # Generate random cache CPU locations
    cache_cpu_locations = torch.randperm(total_tokens * 2, dtype=torch.int64)[:total_tokens]

    # Generate hot buffer with random tokens from [0, total_tokens)
    all_tokens = torch.randperm(total_tokens, dtype=torch.int64)
    hot_buffer_token_indices = all_tokens[:hot_buffer_size].clone()
    hot_buffer_device_locations = torch.arange(hot_buffer_size, dtype=torch.int64)

    # Generate top_k with controlled hit ratio
    num_hits = int(top_k_size * hit_ratio)
    num_misses = top_k_size - num_hits

    # Select some tokens from hot buffer (hits) and some from outside (misses)
    hit_tokens = hot_buffer_token_indices[torch.randperm(hot_buffer_size)[:num_hits]]
    miss_candidates = all_tokens[hot_buffer_size:]
    miss_tokens = miss_candidates[torch.randperm(len(miss_candidates))[:num_misses]]
    top_k_indices = torch.cat([hit_tokens, miss_tokens])
    top_k_indices = top_k_indices[torch.randperm(top_k_size)]

    # Move to GPU
    top_k_indices = top_k_indices.to(device)
    hot_buffer_token_indices_kernel = hot_buffer_token_indices.clone().to(device)
    hot_buffer_device_locations = hot_buffer_device_locations.to(device)
    cache_cpu_locations = cache_cpu_locations.to(device)

    # Allocate output tensors
    top_k_device_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_src_cpu_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_dst_gpu_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_count = torch.zeros(1, dtype=torch.int32, device=device)

    # Run reference implementation
    (
        ref_top_k_locs,
        ref_copy_src,
        ref_copy_dst,
        ref_copy_count,
        ref_hot_tokens,
    ) = reference_sparse_cache_manager(
        top_k_indices,
        hot_buffer_token_indices,
        hot_buffer_device_locations,
        cache_cpu_locations,
    )

    # Run kernel
    from sgl_kernel.kvcacheio import sparse_cache_manager_fused

    sparse_cache_manager_fused(
        top_k_indices,
        hot_buffer_token_indices_kernel,
        hot_buffer_device_locations,
        cache_cpu_locations,
        top_k_device_locations,
        copy_src_cpu_locations,
        copy_dst_gpu_locations,
        copy_count,
    )

    torch.cuda.synchronize()

    # Verify copy count
    kernel_copy_count = copy_count.item()
    assert kernel_copy_count == ref_copy_count, (
        f"Copy count mismatch: kernel={kernel_copy_count}, ref={ref_copy_count}"
    )

    # Verify top_k device locations
    torch.testing.assert_close(
        top_k_device_locations.cpu(),
        ref_top_k_locs,
        msg="top_k_device_locations mismatch",
    )

    # Verify copy operations (order may differ, so check as sets)
    if kernel_copy_count > 0:
        kernel_copies = set(
            zip(
                copy_src_cpu_locations[:kernel_copy_count].cpu().tolist(),
                copy_dst_gpu_locations[:kernel_copy_count].cpu().tolist(),
            )
        )
        ref_copies = set(zip(ref_copy_src.tolist(), ref_copy_dst.tolist()))
        assert kernel_copies == ref_copies, "Copy operations mismatch"


@pytest.mark.parametrize("top_k_size", [16, 64, 128])
@pytest.mark.parametrize("hot_buffer_size", [64, 256, 512])
@pytest.mark.parametrize("total_tokens", [1024])
def test_sparse_cache_manager_multi_phase(
    top_k_size: int,
    hot_buffer_size: int,
    total_tokens: int,
):
    """Test the multi-phase sparse cache manager kernel."""
    if top_k_size > hot_buffer_size:
        pytest.skip("top_k_size cannot exceed hot_buffer_size")

    torch.cuda.manual_seed(42)
    device = "cuda"
    hit_ratio = 0.5

    # Generate random cache CPU locations
    cache_cpu_locations = torch.randperm(total_tokens * 2, dtype=torch.int64)[:total_tokens]

    # Generate hot buffer
    all_tokens = torch.randperm(total_tokens, dtype=torch.int64)
    hot_buffer_token_indices = all_tokens[:hot_buffer_size].clone()
    hot_buffer_device_locations = torch.arange(hot_buffer_size, dtype=torch.int64)

    # Generate top_k
    num_hits = int(top_k_size * hit_ratio)
    num_misses = top_k_size - num_hits
    hit_tokens = hot_buffer_token_indices[torch.randperm(hot_buffer_size)[:num_hits]]
    miss_candidates = all_tokens[hot_buffer_size:]
    miss_tokens = miss_candidates[torch.randperm(len(miss_candidates))[:num_misses]]
    top_k_indices = torch.cat([hit_tokens, miss_tokens])
    top_k_indices = top_k_indices[torch.randperm(top_k_size)]

    # Move to GPU
    top_k_indices = top_k_indices.to(device)
    hot_buffer_token_indices_kernel = hot_buffer_token_indices.clone().to(device)
    hot_buffer_device_locations = hot_buffer_device_locations.to(device)
    cache_cpu_locations = cache_cpu_locations.to(device)

    # Allocate output tensors
    top_k_device_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_src_cpu_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_dst_gpu_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_count = torch.zeros(1, dtype=torch.int32, device=device)

    # Run reference
    (
        ref_top_k_locs,
        ref_copy_src,
        ref_copy_dst,
        ref_copy_count,
        ref_hot_tokens,
    ) = reference_sparse_cache_manager(
        top_k_indices,
        hot_buffer_token_indices,
        hot_buffer_device_locations,
        cache_cpu_locations,
    )

    # Run kernel
    from sgl_kernel.kvcacheio import sparse_cache_manager

    sparse_cache_manager(
        top_k_indices,
        hot_buffer_token_indices_kernel,
        hot_buffer_device_locations,
        cache_cpu_locations,
        top_k_device_locations,
        copy_src_cpu_locations,
        copy_dst_gpu_locations,
        copy_count,
    )

    torch.cuda.synchronize()

    # Verify
    kernel_copy_count = copy_count.item()
    assert kernel_copy_count == ref_copy_count, (
        f"Copy count mismatch: kernel={kernel_copy_count}, ref={ref_copy_count}"
    )

    torch.testing.assert_close(
        top_k_device_locations.cpu(),
        ref_top_k_locs,
        msg="top_k_device_locations mismatch",
    )


@pytest.mark.parametrize("top_k_size", [16, 32])
@pytest.mark.parametrize("hot_buffer_size", [64, 128])
def test_all_hits(top_k_size: int, hot_buffer_size: int):
    """Test case where all top_k tokens are cache hits."""
    if top_k_size > hot_buffer_size:
        pytest.skip("top_k_size cannot exceed hot_buffer_size")

    torch.cuda.manual_seed(42)
    device = "cuda"
    total_tokens = 512

    cache_cpu_locations = torch.arange(total_tokens, dtype=torch.int64)
    all_tokens = torch.randperm(total_tokens, dtype=torch.int64)
    hot_buffer_token_indices = all_tokens[:hot_buffer_size].clone()
    hot_buffer_device_locations = torch.arange(hot_buffer_size, dtype=torch.int64)

    # All top_k from hot buffer
    top_k_indices = hot_buffer_token_indices[torch.randperm(hot_buffer_size)[:top_k_size]]

    # Move to GPU
    top_k_indices = top_k_indices.to(device)
    hot_buffer_token_indices_kernel = hot_buffer_token_indices.clone().to(device)
    hot_buffer_device_locations = hot_buffer_device_locations.to(device)
    cache_cpu_locations = cache_cpu_locations.to(device)

    top_k_device_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_src_cpu_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_dst_gpu_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_count = torch.zeros(1, dtype=torch.int32, device=device)

    from sgl_kernel.kvcacheio import sparse_cache_manager_fused

    sparse_cache_manager_fused(
        top_k_indices,
        hot_buffer_token_indices_kernel,
        hot_buffer_device_locations,
        cache_cpu_locations,
        top_k_device_locations,
        copy_src_cpu_locations,
        copy_dst_gpu_locations,
        copy_count,
    )

    torch.cuda.synchronize()

    # No copies needed
    assert copy_count.item() == 0, "Expected 0 copies for all-hits case"

    # All device locations should be valid (non-negative)
    assert (top_k_device_locations >= 0).all(), "All device locations should be valid"


@pytest.mark.parametrize("top_k_size", [16, 32])
@pytest.mark.parametrize("hot_buffer_size", [64, 128])
def test_all_misses(top_k_size: int, hot_buffer_size: int):
    """Test case where all top_k tokens are cache misses."""
    if top_k_size > hot_buffer_size:
        pytest.skip("top_k_size cannot exceed hot_buffer_size")

    torch.cuda.manual_seed(42)
    device = "cuda"
    total_tokens = 512

    cache_cpu_locations = torch.arange(total_tokens, dtype=torch.int64)
    all_tokens = torch.arange(total_tokens, dtype=torch.int64)
    hot_buffer_token_indices = all_tokens[:hot_buffer_size].clone()
    hot_buffer_device_locations = torch.arange(hot_buffer_size, dtype=torch.int64)

    # All top_k from outside hot buffer
    top_k_indices = all_tokens[hot_buffer_size : hot_buffer_size + top_k_size].clone()

    # Move to GPU
    top_k_indices = top_k_indices.to(device)
    hot_buffer_token_indices_kernel = hot_buffer_token_indices.clone().to(device)
    hot_buffer_device_locations = hot_buffer_device_locations.to(device)
    cache_cpu_locations = cache_cpu_locations.to(device)

    top_k_device_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_src_cpu_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_dst_gpu_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_count = torch.zeros(1, dtype=torch.int32, device=device)

    from sgl_kernel.kvcacheio import sparse_cache_manager_fused

    sparse_cache_manager_fused(
        top_k_indices,
        hot_buffer_token_indices_kernel,
        hot_buffer_device_locations,
        cache_cpu_locations,
        top_k_device_locations,
        copy_src_cpu_locations,
        copy_dst_gpu_locations,
        copy_count,
    )

    torch.cuda.synchronize()

    # All misses, so copy_count should equal top_k_size
    assert copy_count.item() == top_k_size, (
        f"Expected {top_k_size} copies, got {copy_count.item()}"
    )

    # All device locations should be valid
    assert (top_k_device_locations >= 0).all(), "All device locations should be valid"


def test_hot_buffer_update():
    """Test that hot buffer is correctly updated after evictions."""
    torch.cuda.manual_seed(42)
    device = "cuda"

    top_k_size = 4
    hot_buffer_size = 8
    total_tokens = 16

    cache_cpu_locations = torch.arange(total_tokens, dtype=torch.int64)

    # Hot buffer contains tokens 0-7
    hot_buffer_token_indices = torch.arange(hot_buffer_size, dtype=torch.int64)
    hot_buffer_device_locations = torch.arange(hot_buffer_size, dtype=torch.int64)

    # Request tokens 0, 1, 8, 9 (2 hits, 2 misses)
    top_k_indices = torch.tensor([0, 1, 8, 9], dtype=torch.int64)

    # Move to GPU
    top_k_indices = top_k_indices.to(device)
    hot_buffer_token_indices_kernel = hot_buffer_token_indices.clone().to(device)
    hot_buffer_device_locations = hot_buffer_device_locations.to(device)
    cache_cpu_locations = cache_cpu_locations.to(device)

    top_k_device_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_src_cpu_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_dst_gpu_locations = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_count = torch.zeros(1, dtype=torch.int32, device=device)

    from sgl_kernel.kvcacheio import sparse_cache_manager_fused

    sparse_cache_manager_fused(
        top_k_indices,
        hot_buffer_token_indices_kernel,
        hot_buffer_device_locations,
        cache_cpu_locations,
        top_k_device_locations,
        copy_src_cpu_locations,
        copy_dst_gpu_locations,
        copy_count,
    )

    torch.cuda.synchronize()

    # Check that hot buffer was updated
    updated_hot_buffer = hot_buffer_token_indices_kernel.cpu()

    # Tokens 8 and 9 should now be in hot buffer
    assert 8 in updated_hot_buffer.tolist(), "Token 8 should be in hot buffer"
    assert 9 in updated_hot_buffer.tolist(), "Token 9 should be in hot buffer"

    # Tokens 0 and 1 should still be there (they were hits)
    assert 0 in updated_hot_buffer.tolist(), "Token 0 should still be in hot buffer"
    assert 1 in updated_hot_buffer.tolist(), "Token 1 should still be in hot buffer"

    # Two evictions happened
    assert copy_count.item() == 2, f"Expected 2 copies, got {copy_count.item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
