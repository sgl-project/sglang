"""
Tests for the bitmap-based sparse KV cache manager.

Tests the Triton JIT kernels against reference implementation.
"""

import pytest
import torch

# Skip if no GPU or Triton not available
try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="CUDA and Triton required"
)


class TestSparseCacheManager:
    """Test the SparseCacheManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh manager for each test."""
        from sgl_kernel.sparse_kv_cache import SparseCacheManager
        return SparseCacheManager(
            max_tokens=1024,
            hot_buffer_size=128,
            device="cuda",
        )
    
    def test_init(self, manager):
        """Test manager initialization."""
        assert manager.max_tokens == 1024
        assert manager.hot_buffer_size == 128
        assert manager.residence_bitmap.sum() == 0
        assert (manager.token_to_gpu_loc == -1).all()
        assert (manager.gpu_loc_to_token == -1).all()
    
    def test_all_misses_first_request(self, manager):
        """First request should be all misses."""
        top_k_indices = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device="cuda")
        cache_cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        top_k_locs, copy_info = manager.process_topk(top_k_indices, cache_cpu_locs)
        
        # All should require copies
        assert copy_info.num_copies == 4
        # All GPU locations should be valid
        assert (top_k_locs >= 0).all()
        assert (top_k_locs < manager.hot_buffer_size).all()
        # GPU locations should be unique
        assert len(set(top_k_locs.tolist())) == 4
    
    def test_all_hits_repeated_request(self, manager):
        """Repeated request should be all hits."""
        top_k_indices = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device="cuda")
        cache_cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        # First request
        locs1, copy1 = manager.process_topk(top_k_indices, cache_cpu_locs)
        
        # Same request again
        locs2, copy2 = manager.process_topk(top_k_indices, cache_cpu_locs)
        
        # Should be all hits
        assert copy2.num_copies == 0
        # Same locations
        torch.testing.assert_close(locs1, locs2)
    
    def test_partial_hits(self, manager):
        """Test mix of hits and misses."""
        cache_cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        # First request
        top_k1 = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device="cuda")
        locs1, copy1 = manager.process_topk(top_k1, cache_cpu_locs)
        
        # Second request with partial overlap
        top_k2 = torch.tensor([0, 5, 20, 25], dtype=torch.int64, device="cuda")
        locs2, copy2 = manager.process_topk(top_k2, cache_cpu_locs)
        
        # 2 hits (0, 5), 2 misses (20, 25)
        assert copy2.num_copies == 2
        # Tokens 0 and 5 should have same locations
        assert locs2[0] == locs1[0]  # token 0
        assert locs2[1] == locs1[1]  # token 5
    
    def test_bitmap_consistency(self, manager):
        """Test that bitmap stays consistent."""
        cache_cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        top_k = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device="cuda")
        locs, _ = manager.process_topk(top_k, cache_cpu_locs)
        
        # Check bitmap
        for token in top_k.tolist():
            assert manager.residence_bitmap[token] == 1
            assert manager.token_to_gpu_loc[token] >= 0
        
        # Check reverse mapping
        for i, token in enumerate(top_k.tolist()):
            gpu_loc = locs[i].item()
            assert manager.gpu_loc_to_token[gpu_loc] == token
    
    def test_eviction_updates_bitmap(self, manager):
        """Test that evicted tokens are cleared from bitmap."""
        cache_cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        # Fill hot buffer
        for i in range(0, 128, 4):
            top_k = torch.tensor([i, i+1, i+2, i+3], dtype=torch.int64, device="cuda")
            manager.process_topk(top_k, cache_cpu_locs)
        
        # Now request tokens that will cause evictions
        new_tokens = torch.tensor([500, 501, 502, 503], dtype=torch.int64, device="cuda")
        manager.process_topk(new_tokens, cache_cpu_locs)
        
        # New tokens should be in bitmap
        for token in new_tokens.tolist():
            assert manager.residence_bitmap[token] == 1
        
        # Some old tokens should be evicted
        assert manager.residence_bitmap.sum() <= 128
    
    def test_get_stats(self, manager):
        """Test statistics reporting."""
        stats = manager.get_stats()
        assert stats["num_resident"] == 0
        assert stats["hot_buffer_size"] == 128
        assert stats["occupancy"] == 0.0
        
        # Add some tokens
        top_k = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device="cuda")
        cache_cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        manager.process_topk(top_k, cache_cpu_locs)
        
        stats = manager.get_stats()
        assert stats["num_resident"] == 4
        assert stats["occupancy"] == 4 / 128
    
    def test_reset(self, manager):
        """Test reset functionality."""
        cache_cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        # Add some tokens
        top_k = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device="cuda")
        manager.process_topk(top_k, cache_cpu_locs)
        
        assert manager.residence_bitmap.sum() > 0
        
        # Reset
        manager.reset()
        
        assert manager.residence_bitmap.sum() == 0
        assert (manager.token_to_gpu_loc == -1).all()
        assert (manager.gpu_loc_to_token == -1).all()


class TestReferenceImplementation:
    """Test that Triton kernels match reference implementation."""
    
    @pytest.mark.parametrize("top_k_size", [4, 16, 32])
    @pytest.mark.parametrize("hot_buffer_size", [32, 64, 128])
    @pytest.mark.parametrize("hit_ratio", [0.0, 0.5, 1.0])
    def test_against_reference(self, top_k_size, hot_buffer_size, hit_ratio):
        """Compare Triton kernel against reference."""
        from sgl_kernel.sparse_kv_cache import (
            SparseCacheManager,
            reference_sparse_cache_manager,
        )
        
        if top_k_size > hot_buffer_size:
            pytest.skip("top_k cannot exceed hot_buffer")
        
        torch.manual_seed(42)
        max_tokens = 512
        device = "cuda"
        
        cache_cpu_locs = torch.arange(max_tokens, dtype=torch.int64, device=device)
        
        # Create initial state
        all_tokens = torch.randperm(max_tokens, dtype=torch.int64, device=device)
        
        # Pre-populate some tokens for partial hits
        num_prepopulate = int(hot_buffer_size * 0.5)
        prepopulated = all_tokens[:num_prepopulate]
        
        # Initialize manager
        manager = SparseCacheManager(
            max_tokens=max_tokens,
            hot_buffer_size=hot_buffer_size,
            device=device,
        )
        
        # Pre-populate using reference (to set up consistent state)
        for i in range(0, num_prepopulate, 4):
            chunk = prepopulated[i:min(i+4, num_prepopulate)]
            manager.process_topk(chunk, cache_cpu_locs)
        
        # Now create top_k with controlled hit ratio
        num_hits = int(top_k_size * hit_ratio)
        num_misses = top_k_size - num_hits
        
        # Select hits from prepopulated tokens
        resident_tokens = prepopulated[:min(num_hits, len(prepopulated))]
        if len(resident_tokens) < num_hits:
            num_hits = len(resident_tokens)
            num_misses = top_k_size - num_hits
        
        # Select misses from non-resident tokens
        non_resident = all_tokens[num_prepopulate:]
        miss_tokens = non_resident[:num_misses]
        
        top_k = torch.cat([resident_tokens[:num_hits], miss_tokens])
        top_k = top_k[torch.randperm(len(top_k))][:top_k_size]
        
        # Clone state for reference
        ref_bitmap = manager.residence_bitmap.clone()
        ref_tok_to_loc = manager.token_to_gpu_loc.clone()
        ref_loc_to_tok = manager.gpu_loc_to_token.clone()
        
        # Run Triton version
        triton_locs, triton_copy = manager.process_topk(top_k, cache_cpu_locs)
        
        # Run reference version (on cloned state)
        ref_locs, ref_src, ref_dst, ref_num = reference_sparse_cache_manager(
            top_k, ref_bitmap, ref_tok_to_loc, ref_loc_to_tok, cache_cpu_locs
        )
        
        # Compare results
        # Note: Order may differ, so compare as sets for copies
        assert triton_copy.num_copies == ref_num, (
            f"Copy count mismatch: triton={triton_copy.num_copies}, ref={ref_num}"
        )
        
        # Check that all top_k got valid GPU locations
        assert (triton_locs >= 0).all()
        assert (ref_locs >= 0).all()
        
        # The actual locations may differ (different eviction order),
        # but the number of copies should match


class TestExecuteCopies:
    """Test the copy execution function."""
    
    def test_execute_copies(self):
        """Test that copies work correctly."""
        from sgl_kernel.sparse_kv_cache import execute_copies, CopyInfo
        
        # Create test data
        item_size = 128
        num_tokens = 64
        hot_buffer = 32
        
        cpu_cache = torch.randn(num_tokens, item_size).pin_memory()
        gpu_cache = torch.zeros(hot_buffer, item_size, device="cuda")
        
        # Copy tokens 5, 10, 15 to GPU locations 0, 1, 2
        copy_info = CopyInfo(
            src_cpu_locs=torch.tensor([5, 10, 15], dtype=torch.int64, device="cuda"),
            dst_gpu_locs=torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda"),
            num_copies=3,
        )
        
        execute_copies(cpu_cache, gpu_cache, copy_info, item_size)
        
        torch.cuda.synchronize()
        
        # Verify copies
        torch.testing.assert_close(gpu_cache[0].cpu(), cpu_cache[5])
        torch.testing.assert_close(gpu_cache[1].cpu(), cpu_cache[10])
        torch.testing.assert_close(gpu_cache[2].cpu(), cpu_cache[15])
    
    def test_empty_copies(self):
        """Test with no copies."""
        from sgl_kernel.sparse_kv_cache import execute_copies, CopyInfo
        
        cpu_cache = torch.randn(64, 128).pin_memory()
        gpu_cache = torch.zeros(32, 128, device="cuda")
        
        copy_info = CopyInfo(
            src_cpu_locs=torch.tensor([], dtype=torch.int64, device="cuda"),
            dst_gpu_locs=torch.tensor([], dtype=torch.int64, device="cuda"),
            num_copies=0,
        )
        
        # Should not crash
        execute_copies(cpu_cache, gpu_cache, copy_info, 128)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_token(self):
        """Test with single token."""
        from sgl_kernel.sparse_kv_cache import SparseCacheManager
        
        manager = SparseCacheManager(max_tokens=256, hot_buffer_size=32)
        
        top_k = torch.tensor([42], dtype=torch.int64, device="cuda")
        cache_cpu_locs = torch.arange(256, dtype=torch.int64, device="cuda")
        
        locs, copy_info = manager.process_topk(top_k, cache_cpu_locs)
        
        assert locs.numel() == 1
        assert copy_info.num_copies == 1
    
    def test_full_buffer_with_all_hits(self):
        """Test when buffer is full and request is all hits."""
        from sgl_kernel.sparse_kv_cache import SparseCacheManager
        
        manager = SparseCacheManager(max_tokens=256, hot_buffer_size=32)
        cache_cpu_locs = torch.arange(256, dtype=torch.int64, device="cuda")
        
        # Fill buffer
        for i in range(0, 32, 4):
            top_k = torch.tensor([i, i+1, i+2, i+3], dtype=torch.int64, device="cuda")
            manager.process_topk(top_k, cache_cpu_locs)
        
        # Request subset of what's in buffer
        top_k = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda")
        locs, copy_info = manager.process_topk(top_k, cache_cpu_locs)
        
        assert copy_info.num_copies == 0
    
    def test_sequential_requests(self):
        """Test many sequential requests."""
        from sgl_kernel.sparse_kv_cache import SparseCacheManager
        
        manager = SparseCacheManager(max_tokens=1024, hot_buffer_size=64)
        cache_cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        for i in range(50):
            base = (i * 4) % 900
            top_k = torch.tensor(
                [base, base+1, base+2, base+3],
                dtype=torch.int64, device="cuda"
            )
            locs, copy_info = manager.process_topk(top_k, cache_cpu_locs)
            
            # Verify all locations are valid
            assert (locs >= 0).all()
            assert (locs < 64).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
