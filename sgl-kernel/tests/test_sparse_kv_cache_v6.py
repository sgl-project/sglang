"""
Tests for Sparse KV Cache Manager v6 - 16-bit Token-to-Slot Index
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestSparseCacheManagerV6:
    """Tests for SparseCacheManagerV6."""
    
    @pytest.fixture
    def manager(self):
        """Create a test manager instance."""
        from sgl_kernel.sparse_kv_cache_v6 import SparseCacheManagerV6
        return SparseCacheManagerV6(
            max_tokens=1000,
            hot_buffer_size=100,
            item_size_bytes=64,
            device="cuda",
        )
    
    @pytest.fixture
    def host_cache(self):
        """Create a test host cache (pinned memory)."""
        return torch.arange(1000 * 64, dtype=torch.uint8, device="cpu").pin_memory()
    
    @pytest.fixture  
    def device_buffer(self):
        """Create a test device buffer."""
        return torch.zeros(100 * 64, dtype=torch.uint8, device="cuda")
    
    def test_all_misses(self, manager, host_cache, device_buffer):
        """Test with all cache misses."""
        # All tokens are new
        top_k_tokens = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32, device="cuda")
        
        top_k_device_locs = manager.process_topk(top_k_tokens, host_cache, device_buffer)
        
        # All should get unique device locations
        assert top_k_device_locs.shape == (5,)
        assert len(torch.unique(top_k_device_locs)) == 5
        
        # Verify token_to_slot is updated
        for i, token in enumerate(top_k_tokens.cpu().tolist()):
            slot = manager.get_slot(token)
            assert slot >= 0, f"Token {token} should be resident"
            assert manager.is_resident(token)
    
    def test_all_hits(self, manager, host_cache, device_buffer):
        """Test with all cache hits."""
        # First, populate the cache
        top_k_tokens = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32, device="cuda")
        locs1 = manager.process_topk(top_k_tokens, host_cache, device_buffer)
        
        # Same tokens again - all hits
        locs2 = manager.process_topk(top_k_tokens, host_cache, device_buffer)
        
        # Locations should be the same
        assert torch.equal(locs1, locs2)
    
    def test_partial_hits(self, manager, host_cache, device_buffer):
        """Test with some hits and some misses."""
        # First, populate with tokens 10, 20, 30
        top_k1 = torch.tensor([10, 20, 30], dtype=torch.int32, device="cuda")
        locs1 = manager.process_topk(top_k1, host_cache, device_buffer)
        
        # Get slots for tokens 10, 20, 30
        slots_before = {
            10: manager.get_slot(10),
            20: manager.get_slot(20),
            30: manager.get_slot(30),
        }
        
        # Request tokens 10, 40, 50 - token 10 is hit, 40 and 50 are misses
        top_k2 = torch.tensor([10, 40, 50], dtype=torch.int32, device="cuda")
        locs2 = manager.process_topk(top_k2, host_cache, device_buffer)
        
        # Token 10 should have the same slot
        assert manager.get_slot(10) == slots_before[10]
        
        # Tokens 40 and 50 should be resident now
        assert manager.is_resident(40)
        assert manager.is_resident(50)
    
    def test_eviction(self, manager, host_cache, device_buffer):
        """Test that eviction works correctly when buffer is full."""
        # Create a small manager for easier testing
        from sgl_kernel.sparse_kv_cache_v6 import SparseCacheManagerV6
        small_manager = SparseCacheManagerV6(
            max_tokens=100,
            hot_buffer_size=5,
            item_size_bytes=8,
            device="cuda",
        )
        small_host = torch.arange(100 * 8, dtype=torch.uint8, device="cpu").pin_memory()
        small_device = torch.zeros(5 * 8, dtype=torch.uint8, device="cuda")
        
        # Fill the buffer with tokens 0, 1, 2, 3, 4
        top_k1 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device="cuda")
        small_manager.process_topk(top_k1, small_host, small_device)
        
        # All should be resident
        for t in range(5):
            assert small_manager.is_resident(t)
        
        # Now request tokens 0, 1, 10, 11 - 0,1 are hits, 10,11 are misses
        # This should evict 2 tokens (e.g., 2, 3 or 3, 4)
        top_k2 = torch.tensor([0, 1, 10, 11], dtype=torch.int32, device="cuda")
        small_manager.process_topk(top_k2, small_host, small_device)
        
        # 0, 1, 10, 11 should be resident
        assert small_manager.is_resident(0)
        assert small_manager.is_resident(1)
        assert small_manager.is_resident(10)
        assert small_manager.is_resident(11)
        
        # At least 2 of {2, 3, 4} should be evicted
        evicted_count = sum(1 for t in [2, 3, 4] if not small_manager.is_resident(t))
        assert evicted_count >= 2
    
    def test_token_to_slot_consistency(self, manager, host_cache, device_buffer):
        """Test that token_to_slot is consistent with device_buffer_tokens."""
        # Process some tokens
        top_k_tokens = torch.tensor([5, 15, 25, 35, 45], dtype=torch.int32, device="cuda")
        manager.process_topk(top_k_tokens, host_cache, device_buffer)
        
        # For each resident token, verify consistency
        for token in top_k_tokens.cpu().tolist():
            slot = manager.get_slot(token)
            assert slot >= 0
            # device_buffer_tokens[slot] should equal token
            assert manager.device_buffer_tokens[slot].item() == token
    
    def test_repeated_iterations(self, manager, host_cache, device_buffer):
        """Test multiple iterations to verify state consistency."""
        import random
        random.seed(42)
        
        for iteration in range(10):
            # Generate random top_k tokens
            tokens = random.sample(range(500), k=20)
            top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
            
            locs = manager.process_topk(top_k, host_cache, device_buffer)
            
            # All requested tokens should be resident after processing
            for token in tokens:
                assert manager.is_resident(token), f"Iteration {iteration}: Token {token} should be resident"
            
            # All returned locations should be valid
            assert locs.min() >= 0
            assert locs.max() < manager.hot_buffer_size
    
    def test_stats(self, manager, host_cache, device_buffer):
        """Test get_stats method."""
        stats = manager.get_stats()
        assert stats["num_resident"] == 0
        assert stats["hot_buffer_size"] == 100
        assert stats["memory_bytes"] == 1000 * 2  # 16 bits per token
        
        # Process some tokens
        top_k = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device="cuda")
        manager.process_topk(top_k, host_cache, device_buffer)
        
        stats = manager.get_stats()
        assert stats["num_resident"] == 5
    
    def test_reset(self, manager, host_cache, device_buffer):
        """Test reset functionality."""
        # Process some tokens
        top_k = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
        manager.process_topk(top_k, host_cache, device_buffer)
        
        assert manager.is_resident(1)
        assert manager.is_resident(2)
        assert manager.is_resident(3)
        
        # Reset
        manager.reset()
        
        # Should not be resident anymore
        assert not manager.is_resident(1)
        assert not manager.is_resident(2)
        assert not manager.is_resident(3)
    
    def test_o1_lookup_complexity(self, manager, host_cache, device_buffer):
        """
        Verify O(1) lookup by checking execution time is constant
        regardless of buffer size.
        """
        from sgl_kernel.sparse_kv_cache_v6 import SparseCacheManagerV6
        import time
        
        # Create two managers with different buffer sizes
        small_manager = SparseCacheManagerV6(
            max_tokens=10000,
            hot_buffer_size=100,
            item_size_bytes=8,
            device="cuda",
        )
        large_manager = SparseCacheManagerV6(
            max_tokens=10000,
            hot_buffer_size=4000,  # 40x larger
            item_size_bytes=8,
            device="cuda",
        )
        
        small_host = torch.arange(10000 * 8, dtype=torch.uint8, device="cpu").pin_memory()
        small_device = torch.zeros(100 * 8, dtype=torch.uint8, device="cuda")
        large_device = torch.zeros(4000 * 8, dtype=torch.uint8, device="cuda")
        
        # Fill both buffers halfway
        small_fill = torch.arange(50, dtype=torch.int32, device="cuda")
        large_fill = torch.arange(2000, dtype=torch.int32, device="cuda")
        
        small_manager.process_topk(small_fill, small_host, small_device)
        large_manager.process_topk(large_fill, small_host, large_device)
        
        # Time lookups with all hits (same tokens)
        test_tokens = torch.arange(10, 30, dtype=torch.int32, device="cuda")
        
        # Warmup
        for _ in range(3):
            small_manager.process_topk(test_tokens, small_host, small_device)
            large_manager.process_topk(test_tokens, small_host, large_device)
        torch.cuda.synchronize()
        
        # Time small buffer
        start = time.perf_counter()
        for _ in range(100):
            small_manager.process_topk(test_tokens, small_host, small_device)
        torch.cuda.synchronize()
        small_time = time.perf_counter() - start
        
        # Time large buffer
        start = time.perf_counter()
        for _ in range(100):
            large_manager.process_topk(test_tokens, small_host, large_device)
        torch.cuda.synchronize()
        large_time = time.perf_counter() - start
        
        # v6 should have similar time for both (O(1) lookup)
        # Allow 3x tolerance for system variance
        # v5 would have 40x difference due to O(H) scan
        ratio = large_time / small_time
        print(f"\nSmall buffer time: {small_time:.4f}s, Large buffer time: {large_time:.4f}s, Ratio: {ratio:.2f}x")
        # Note: In practice, the ratio should be close to 1 for v6, but we use 5x tolerance
        # due to system variance and other factors (kernel launch overhead, etc.)
        assert ratio < 5, f"Large buffer took {ratio:.2f}x longer than small buffer, expected ~1x for O(1) lookup"


class TestV6VsReference:
    """Test v6 against a Python reference implementation."""
    
    def reference_process_topk(
        self,
        top_k_tokens: torch.Tensor,
        token_to_slot: torch.Tensor,
        device_buffer_tokens: torch.Tensor,
        host_cache_locs: torch.Tensor,
        device_buffer_locs: torch.Tensor,
        NOT_PRESENT: int = 0xFFFF,
    ):
        """
        Python reference implementation.
        
        Returns: top_k_device_locs, updated token_to_slot, updated device_buffer_tokens
        """
        top_k_tokens = top_k_tokens.cpu().tolist()
        token_to_slot = token_to_slot.cpu().numpy().astype('uint16').tolist()
        device_buffer_tokens = device_buffer_tokens.cpu().tolist()
        device_buffer_locs = device_buffer_locs.cpu().tolist()
        hot_buffer_size = len(device_buffer_tokens)
        
        top_k_device_locs = []
        hits = []
        misses = []
        
        # Phase 1: Identify hits and misses
        for idx, token in enumerate(top_k_tokens):
            slot = token_to_slot[token]
            if slot != NOT_PRESENT:
                # Hit
                hits.append((idx, token, slot))
                top_k_device_locs.append(device_buffer_locs[slot])
            else:
                # Miss
                misses.append((idx, token))
                top_k_device_locs.append(None)
        
        # Phase 2: Find evictable slots (not in top_k)
        protected_slots = set(slot for _, _, slot in hits)
        evictable_slots = [s for s in range(hot_buffer_size) if s not in protected_slots]
        
        # Phase 3: Assign evictable slots to misses
        for i, (idx, token) in enumerate(misses):
            if i >= len(evictable_slots):
                raise RuntimeError("Not enough evictable slots")
            
            evict_slot = evictable_slots[i]
            old_token = device_buffer_tokens[evict_slot]
            
            # Clear old token
            if old_token >= 0:
                token_to_slot[old_token] = NOT_PRESENT
            
            # Set new token
            token_to_slot[token] = evict_slot
            device_buffer_tokens[evict_slot] = token
            top_k_device_locs[idx] = device_buffer_locs[evict_slot]
        
        return top_k_device_locs, token_to_slot, device_buffer_tokens
    
    def test_vs_reference(self):
        """Test v6 kernel against Python reference."""
        from sgl_kernel.sparse_kv_cache_v6 import SparseCacheManagerV6
        import random
        random.seed(123)
        
        max_tokens = 200
        hot_buffer_size = 50
        item_size_bytes = 8
        
        manager = SparseCacheManagerV6(
            max_tokens=max_tokens,
            hot_buffer_size=hot_buffer_size,
            item_size_bytes=item_size_bytes,
            device="cuda",
        )
        
        host_cache = torch.arange(max_tokens * item_size_bytes, dtype=torch.uint8, device="cpu").pin_memory()
        device_buffer = torch.zeros(hot_buffer_size * item_size_bytes, dtype=torch.uint8, device="cuda")
        
        # Reference state
        ref_token_to_slot = [0xFFFF] * max_tokens
        ref_device_buffer_tokens = [-1] * hot_buffer_size
        device_buffer_locs = list(range(hot_buffer_size))
        
        for iteration in range(20):
            # Generate random tokens
            num_tokens = random.randint(5, 30)
            tokens = random.sample(range(max_tokens), k=num_tokens)
            top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
            
            # Run kernel
            kernel_locs = manager.process_topk(top_k, host_cache, device_buffer)
            
            # Run reference
            ref_locs, ref_token_to_slot, ref_device_buffer_tokens = self.reference_process_topk(
                top_k,
                torch.tensor(ref_token_to_slot, dtype=torch.int16),
                torch.tensor(ref_device_buffer_tokens, dtype=torch.int32),
                manager.host_cache_locs,
                manager.device_buffer_locs,
            )
            
            # Verify locations match
            kernel_locs_list = kernel_locs.cpu().tolist()
            for idx, (kl, rl) in enumerate(zip(kernel_locs_list, ref_locs)):
                assert kl == rl, f"Iteration {iteration}, token {tokens[idx]}: kernel={kl}, ref={rl}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
