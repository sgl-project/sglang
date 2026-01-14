"""
Tests for the JIT-compiled CUDA sparse KV cache manager.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required"
)


class TestSparseCacheManagerCUDA:
    """Test the SparseCacheManagerCUDA class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh manager for each test."""
        from sgl_kernel.sparse_kv_cache_cuda import SparseCacheManagerCUDA
        return SparseCacheManagerCUDA(
            max_tokens=1024,
            hot_buffer_size=128,
            device="cuda",
        )
    
    def test_init(self, manager):
        """Test initialization."""
        assert manager.max_tokens == 1024
        assert manager.hot_buffer_size == 128
        assert manager.residence_bitmap.sum() == 0
        assert (manager.token_to_gpu_loc == -1).all()
        assert (manager.gpu_loc_to_token == -1).all()
    
    def test_first_request_all_misses(self, manager):
        """First request should have all misses."""
        top_k = torch.tensor([0, 5, 10, 15], dtype=torch.int32, device="cuda")
        cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        gpu_locs, copy_info = manager.process_topk(top_k, cpu_locs)
        
        # All should be misses
        assert copy_info.num_copies == 4
        # All locations should be valid
        assert (gpu_locs >= 0).all()
        assert (gpu_locs < manager.hot_buffer_size).all()
        # Locations should be unique
        assert len(set(gpu_locs.tolist())) == 4
    
    def test_repeated_request_all_hits(self, manager):
        """Same request twice should be all hits second time."""
        top_k = torch.tensor([0, 5, 10, 15], dtype=torch.int32, device="cuda")
        cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        # First request
        locs1, copy1 = manager.process_topk(top_k, cpu_locs)
        
        # Same request again
        locs2, copy2 = manager.process_topk(top_k, cpu_locs)
        
        # Should be all hits
        assert copy2.num_copies == 0
        # Same locations
        assert (locs1 == locs2).all()
    
    def test_partial_overlap(self, manager):
        """Test with partial hits and misses."""
        cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        # First request: tokens 0, 5, 10, 15
        top_k1 = torch.tensor([0, 5, 10, 15], dtype=torch.int32, device="cuda")
        locs1, _ = manager.process_topk(top_k1, cpu_locs)
        
        # Second request: tokens 0, 5, 20, 25 (2 hits, 2 misses)
        top_k2 = torch.tensor([0, 5, 20, 25], dtype=torch.int32, device="cuda")
        locs2, copy2 = manager.process_topk(top_k2, cpu_locs)
        
        # 2 misses
        assert copy2.num_copies == 2
        # Hits should have same locations
        assert locs2[0] == locs1[0]  # token 0
        assert locs2[1] == locs1[1]  # token 5
    
    def test_bitmap_updated_correctly(self, manager):
        """Test that bitmap state is correct after operations."""
        cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        top_k = torch.tensor([100, 200, 300], dtype=torch.int32, device="cuda")
        locs, _ = manager.process_topk(top_k, cpu_locs)
        
        # Check bitmap
        assert manager.residence_bitmap[100].item() == 1
        assert manager.residence_bitmap[200].item() == 1
        assert manager.residence_bitmap[300].item() == 1
        assert manager.residence_bitmap[0].item() == 0  # Not added
        
        # Check token -> GPU loc
        assert manager.token_to_gpu_loc[100].item() == locs[0].item()
        assert manager.token_to_gpu_loc[200].item() == locs[1].item()
        assert manager.token_to_gpu_loc[300].item() == locs[2].item()
        
        # Check GPU loc -> token (reverse mapping)
        for i, token in enumerate([100, 200, 300]):
            gpu_loc = locs[i].item()
            assert manager.gpu_loc_to_token[gpu_loc].item() == token
    
    def test_eviction_clears_bitmap(self, manager):
        """Test that evicted tokens are cleared from bitmap."""
        cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        # Fill buffer with tokens 0-127
        for i in range(0, 128, 4):
            top_k = torch.tensor([i, i+1, i+2, i+3], dtype=torch.int32, device="cuda")
            manager.process_topk(top_k, cpu_locs)
        
        # All 128 slots used
        assert manager.residence_bitmap[:128].sum().item() == 128
        
        # Request new tokens that will cause evictions
        new_tokens = torch.tensor([500, 501, 502, 503], dtype=torch.int32, device="cuda")
        manager.process_topk(new_tokens, cpu_locs)
        
        # New tokens should be resident
        for tok in [500, 501, 502, 503]:
            assert manager.residence_bitmap[tok].item() == 1
        
        # Some old tokens should be evicted (total still 128)
        assert manager.residence_bitmap.sum().item() == 128
    
    def test_reset(self, manager):
        """Test reset clears all state."""
        cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        
        top_k = torch.tensor([0, 5, 10], dtype=torch.int32, device="cuda")
        manager.process_topk(top_k, cpu_locs)
        
        assert manager.residence_bitmap.sum() > 0
        
        manager.reset()
        
        assert manager.residence_bitmap.sum() == 0
        assert (manager.token_to_gpu_loc == -1).all()
        assert (manager.gpu_loc_to_token == -1).all()
    
    def test_get_stats(self, manager):
        """Test statistics reporting."""
        stats = manager.get_stats()
        assert stats["num_resident"] == 0
        assert stats["occupancy"] == 0.0
        
        cpu_locs = torch.arange(1024, dtype=torch.int64, device="cuda")
        top_k = torch.tensor([0, 5, 10, 15], dtype=torch.int32, device="cuda")
        manager.process_topk(top_k, cpu_locs)
        
        stats = manager.get_stats()
        assert stats["num_resident"] == 4
        assert stats["occupancy"] == 4 / 128


class TestCopyExecution:
    """Test copy execution."""
    
    def test_execute_copies(self):
        """Test that copies transfer data correctly."""
        from sgl_kernel.sparse_kv_cache_cuda import SparseCacheManagerCUDA, CopyInfo
        
        manager = SparseCacheManagerCUDA(max_tokens=256, hot_buffer_size=64)
        
        # Create test data
        item_size = 128  # elements per cache entry
        cpu_cache = torch.randn(256, item_size).pin_memory()
        gpu_cache = torch.zeros(64, item_size, device="cuda")
        
        # Manual copy info
        copy_info = CopyInfo(
            src_cpu_locs=torch.tensor([10, 20, 30], dtype=torch.int64, device="cuda"),
            dst_gpu_locs=torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda"),
            num_copies=3,
        )
        
        manager.execute_copies(cpu_cache, gpu_cache, copy_info, item_size * 4)  # 4 bytes per float
        
        torch.cuda.synchronize()
        
        # Verify
        torch.testing.assert_close(gpu_cache[0].cpu(), cpu_cache[10])
        torch.testing.assert_close(gpu_cache[1].cpu(), cpu_cache[20])
        torch.testing.assert_close(gpu_cache[2].cpu(), cpu_cache[30])
    
    def test_empty_copies(self):
        """Test with no copies needed."""
        from sgl_kernel.sparse_kv_cache_cuda import SparseCacheManagerCUDA, CopyInfo
        
        manager = SparseCacheManagerCUDA(max_tokens=256, hot_buffer_size=64)
        
        cpu_cache = torch.randn(256, 128).pin_memory()
        gpu_cache = torch.zeros(64, 128, device="cuda")
        
        copy_info = CopyInfo(
            src_cpu_locs=torch.tensor([], dtype=torch.int64, device="cuda"),
            dst_gpu_locs=torch.tensor([], dtype=torch.int64, device="cuda"),
            num_copies=0,
        )
        
        # Should not crash
        manager.execute_copies(cpu_cache, gpu_cache, copy_info, 128 * 4)


class TestStandaloneFunction:
    """Test the standalone process_sparse_cache function."""
    
    def test_standalone_function(self):
        """Test using the standalone function."""
        from sgl_kernel.sparse_kv_cache_cuda import process_sparse_cache
        
        max_tokens = 256
        hot_buffer = 32
        device = "cuda"
        
        # Create state tensors
        bitmap = torch.zeros(max_tokens, dtype=torch.int8, device=device)
        tok_to_loc = torch.full((max_tokens,), -1, dtype=torch.int32, device=device)
        loc_to_tok = torch.full((hot_buffer,), -1, dtype=torch.int32, device=device)
        
        top_k = torch.tensor([5, 10, 15, 20], dtype=torch.int32, device=device)
        cpu_locs = torch.arange(max_tokens, dtype=torch.int64, device=device)
        
        gpu_locs, copy_src, copy_dst, num_copies = process_sparse_cache(
            top_k, cpu_locs, bitmap, tok_to_loc, loc_to_tok
        )
        
        # All misses on first call
        assert num_copies == 4
        assert (gpu_locs >= 0).all()
        
        # State should be updated
        assert bitmap[5].item() == 1
        assert bitmap[10].item() == 1
        
        # Call again - should be all hits
        gpu_locs2, _, _, num_copies2 = process_sparse_cache(
            top_k, cpu_locs, bitmap, tok_to_loc, loc_to_tok
        )
        
        assert num_copies2 == 0
        assert (gpu_locs == gpu_locs2).all()


class TestLargeScale:
    """Test with larger sizes."""
    
    @pytest.mark.parametrize("top_k_size", [16, 64, 128, 256])
    @pytest.mark.parametrize("hot_buffer_size", [256, 512, 1024])
    def test_various_sizes(self, top_k_size, hot_buffer_size):
        """Test with various sizes."""
        from sgl_kernel.sparse_kv_cache_cuda import SparseCacheManagerCUDA
        
        if top_k_size > hot_buffer_size:
            pytest.skip("top_k cannot exceed hot_buffer")
        
        manager = SparseCacheManagerCUDA(
            max_tokens=4096,
            hot_buffer_size=hot_buffer_size,
        )
        
        cpu_locs = torch.arange(4096, dtype=torch.int64, device="cuda")
        
        # Random tokens
        top_k = torch.randperm(4096, device="cuda")[:top_k_size].to(torch.int32)
        
        gpu_locs, copy_info = manager.process_topk(top_k, cpu_locs)
        
        # All should get valid locations
        assert (gpu_locs >= 0).all()
        assert (gpu_locs < hot_buffer_size).all()
        
        # First call should be all misses
        assert copy_info.num_copies == top_k_size
    
    def test_sequential_requests(self):
        """Test many sequential requests."""
        from sgl_kernel.sparse_kv_cache_cuda import SparseCacheManagerCUDA
        
        manager = SparseCacheManagerCUDA(
            max_tokens=2048,
            hot_buffer_size=256,
        )
        
        cpu_locs = torch.arange(2048, dtype=torch.int64, device="cuda")
        
        for i in range(100):
            base = (i * 8) % 1800
            top_k = torch.arange(base, base + 8, dtype=torch.int32, device="cuda")
            
            gpu_locs, copy_info = manager.process_topk(top_k, cpu_locs)
            
            # All should get valid locations
            assert (gpu_locs >= 0).all()
            assert (gpu_locs < 256).all()
        
        # After many iterations, buffer should be full
        stats = manager.get_stats()
        assert stats["num_resident"] <= 256


class TestPrefill:
    """Test prefill functionality."""
    
    def test_prefill_tokens(self):
        """Test pre-filling hot buffer."""
        from sgl_kernel.sparse_kv_cache_cuda import SparseCacheManagerCUDA
        
        manager = SparseCacheManagerCUDA(
            max_tokens=256,
            hot_buffer_size=32,
        )
        
        # Prefill some tokens
        tokens = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device="cuda")
        locs = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
        manager.prefill_tokens(tokens, locs)
        
        # Check they're marked as resident
        assert manager.residence_bitmap[10].item() == 1
        assert manager.residence_bitmap[20].item() == 1
        assert manager.token_to_gpu_loc[10].item() == 0
        assert manager.token_to_gpu_loc[20].item() == 1
        
        # Request these tokens - should be hits
        cpu_locs = torch.arange(256, dtype=torch.int64, device="cuda")
        gpu_locs, copy_info = manager.process_topk(tokens, cpu_locs)
        
        assert copy_info.num_copies == 0
        assert gpu_locs[0].item() == 0
        assert gpu_locs[1].item() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
