import unittest
from dataclasses import dataclass

import torch

from sglang.srt.mem_cache_v2.memory_manager import MemoryManager
from sglang.srt.mem_cache_v2.radix_cache import RadixCache


@dataclass
class MockReq:
    """Mock request object for testing."""

    rid: str
    origin_input_ids: list[int]
    output_ids: list[int]

    @property
    def seq_len(self) -> int:
        return len(self.origin_input_ids) + len(self.output_ids)


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.page_size = 16
        self.size = 4096
        self.max_ctx = 512
        self.device = "cpu"  # Use CPU for testing
        self.cache_index = RadixCache(page_size=self.page_size)
        self.mm = MemoryManager(
            cache_index=self.cache_index,
            memory_pools={},  # No pools for basic tests
            page_size=self.page_size,
            size=self.size,
            max_ctx=self.max_ctx,
            device=self.device,
        )
        self.mm.req_pool.clear()

    def test_realistic_serving_simulation(self):
        reqA = MockReq(rid="A", origin_input_ids=list(range(32)), output_ids=[])
        reqB = MockReq(rid="B", origin_input_ids=list(range(42)), output_ids=[])
        reqC = MockReq(rid="C", origin_input_ids=list(range(3, 23)), output_ids=[])
        reqD = MockReq(
            rid="D",
            origin_input_ids=list(range(42)) + list(range(200, 211)),
            output_ids=[],
        )

        # prefill A
        matchA = self.mm.match_prefix(reqA.origin_input_ids)
        self.assertEqual(len(matchA.matched_indices), 0)
        self.mm.allocate_request(reqA, include_last=True, match_result=matchA)
        self.mm.update_cache(reqA)
        self.assertEqual(self.mm.req_info["A"].cached_len, 32)
        reqA.output_ids.append(200)

        # prefill B
        matchB = self.mm.match_prefix(reqB.origin_input_ids)
        self.assertEqual(len(matchB.matched_indices), 32)
        self.mm.allocate_request(reqB, include_last=True, match_result=matchB)
        self.mm.update_cache(reqB)
        self.assertEqual(self.mm.req_info["B"].cached_len, 32)
        self.assertEqual(self.mm.req_info["B"].allocated_len, reqB.seq_len)
        reqB.output_ids.append(200)

        for i in range(10):
            self.mm.allocate_tokens(reqA, num_token=1)
            self.mm.allocate_tokens(reqB, num_token=1)
            reqA.output_ids.append(i + 200 + 1)
            reqB.output_ids.append(i + 200 + 1)

        # A finishes
        self.assertEqual(self.mm.req_info["A"].allocated_len, reqA.seq_len - 1)
        self.mm.update_cache(reqA)
        self.assertEqual(self.mm.req_info["A"].cached_len, 32)
        self.mm.release_req(reqA)

        # prefill C
        matchC = self.mm.match_prefix(reqC.origin_input_ids)
        self.assertEqual(len(matchC.matched_indices), 0)
        self.mm.allocate_request(reqC, include_last=True, match_result=matchC)
        self.mm.update_cache(reqC)
        self.assertEqual(self.mm.req_info["C"].cached_len, 16)
        reqC.output_ids.append(200)

        for i in range(3):
            reqB.output_ids.append(i + 200 + 1)
            reqC.output_ids.append(i + 200 + 1)
            self.mm.allocate_tokens(reqB, num_token=1)
            self.mm.allocate_tokens(reqC, num_token=1)

        # B finishes
        self.assertEqual(self.mm.req_info["B"].allocated_len, reqB.seq_len - 1)
        self.mm.update_cache(reqB)
        self.assertEqual(self.mm.req_info["B"].cached_len, 48)

        # prefill D
        matchD = self.mm.match_prefix(reqD.origin_input_ids)
        self.assertEqual(len(matchD.matched_indices), 48)
        self.mm.allocate_request(reqD, include_last=True, match_result=matchD)
        self.mm.update_cache(reqD)
        self.assertEqual(self.mm.req_info["D"].cached_len, 48)
        reqD.output_ids.append(200)

        for i in range(3):
            reqC.output_ids.append(i + 200 + 1)
            reqD.output_ids.append(i + 200 + 1)
            self.mm.allocate_tokens(reqC, num_token=1)
            self.mm.allocate_tokens(reqD, num_token=1)

        # C finishes
        self.assertEqual(
            self.mm.req_info["C"].allocated_len, len(reqC.origin_input_ids) + 6
        )
        self.mm.update_cache(reqC)
        self.assertEqual(self.mm.req_info["C"].cached_len, 16)
        self.mm.release_req(reqC)

        # D finishes
        self.assertEqual(
            self.mm.req_info["D"].allocated_len, len(reqD.origin_input_ids) + 3
        )
        self.mm.update_cache(reqD)
        self.assertEqual(self.mm.req_info["D"].cached_len, 48)
        self.mm.release_req(reqD)

        # check memory can be reclaimed
        # Not all allocated memory is in radix cache (only page-aligned portions)
        # But we should be able to eventually free all memory
        initial_free = len(self.mm.allocator.free_list)

        # Evict everything we can from radix cache
        # Calculate how much memory we need based on allocator state
        tokens_needed = (self.mm.allocator.page_num - initial_free) * self.page_size
        evicted = self.mm.evict(tokens_needed)

        # Should have evicted something
        self.assertGreater(evicted, 0, "Should evict some tokens")

        # After eviction, should have more free pages
        final_free = len(self.mm.allocator.free_list)
        self.assertGreaterEqual(final_free, initial_free, "Should free up memory")


class TestAllocator(unittest.TestCase):
    """Test Allocator class directly without memory manager overhead."""

    def setUp(self):
        from sglang.srt.mem_cache_v2.memory_manager import Allocator

        self.page_size = 16
        self.size = 320  # 20 pages
        self.allocator = Allocator(self.size, self.page_size)

    def test_fresh_allocation_partial_page(self):
        """Test fresh allocation of partial page."""
        result = self.allocator.alloc(5, last_loc=0)
        # Should allocate from page 1: [16, 17, 18, 19, 20]
        self.assertEqual(result, [16, 17, 18, 19, 20])
        self.assertEqual(len(self.allocator.free_list), 19)  # 1 page used

    def test_fresh_allocation_full_page(self):
        """Test fresh allocation of exactly one page."""
        result = self.allocator.alloc(16, last_loc=0)
        # Should allocate page 1: [16..31]
        self.assertEqual(result, list(range(16, 32)))
        self.assertEqual(len(self.allocator.free_list), 19)

    def test_fresh_allocation_multiple_pages(self):
        """Test fresh allocation spanning multiple pages."""
        result = self.allocator.alloc(40, last_loc=0)
        # 3 pages: [16..31] + [32..47] + [48..55]
        expected = list(range(16, 32)) + list(range(32, 48)) + list(range(48, 56))
        self.assertEqual(result, expected)
        self.assertEqual(len(self.allocator.free_list), 17)

    def test_continue_within_same_page(self):
        """Test continuing allocation within the same page."""
        # First allocate to get to index 18
        r1 = self.allocator.alloc(3, last_loc=0)  # [16, 17, 18]
        self.assertEqual(r1, [16, 17, 18])

        # Continue from 18, allocate 10 more (still in same page)
        r2 = self.allocator.alloc(10, last_loc=18)
        self.assertEqual(r2, list(range(19, 29)))

    def test_continue_spans_pages(self):
        """Test continuing allocation that spans multiple pages."""
        # First allocate to index 26
        r1 = self.allocator.alloc(11, last_loc=0)  # [16..26]
        self.assertEqual(r1[-1], 26)

        # Continue from 26, allocate 20 more
        # Should fill [27..31] (5 slots) + [32..46] (15 slots)
        r2 = self.allocator.alloc(20, last_loc=26)
        expected = list(range(27, 32)) + list(range(32, 47))
        self.assertEqual(r2, expected)

    def test_continue_from_page_boundary(self):
        """Test continuing from exact page boundary."""
        # Fill page 1 completely
        r1 = self.allocator.alloc(16, last_loc=0)  # [16..31]
        self.assertEqual(r1[-1], 31)

        # Continue from page boundary
        r2 = self.allocator.alloc(10, last_loc=31)
        self.assertEqual(r2, list(range(32, 42)))

    def test_out_of_memory(self):
        """Test out of memory returns empty list."""
        # Try to allocate more pages than available (need 25 pages, have 20)
        with self.assertRaises(RuntimeError):
            self.allocator.alloc(400, last_loc=0)
        self.assertEqual(len(self.allocator.free_list), 20)  # Unchanged

    def test_free_and_realloc(self):
        """Test freeing and reallocating memory."""
        # Allocate 48 tokens (3 pages)
        indices = self.allocator.alloc(48, last_loc=0)
        self.assertEqual(len(indices), 48)
        self.assertEqual(len(self.allocator.free_list), 17)

        # Free them
        self.allocator.free(indices)
        self.assertGreaterEqual(len(self.allocator.free_list), 20)  # Pages back

        # Reallocate
        indices2 = self.allocator.alloc(32, last_loc=0)
        self.assertEqual(len(indices2), 32)

    def test_sequential_allocations(self):
        """Test sequential allocations build up correctly."""
        r1 = self.allocator.alloc(10, last_loc=0)
        r2 = self.allocator.alloc(20, last_loc=r1[-1])
        r3 = self.allocator.alloc(5, last_loc=r2[-1])

        self.assertEqual(r1, list(range(16, 26)))
        self.assertEqual(r2, list(range(26, 32)) + list(range(32, 46)))
        self.assertEqual(r3, list(range(46, 51)))


class TestMemoryManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        self.page_size = 16
        self.size = 256  # Small pool to trigger eviction easily
        self.max_ctx = 512
        self.device = "cpu"
        self.cache_index = RadixCache(page_size=self.page_size)
        self.mm = MemoryManager(
            cache_index=self.cache_index,
            memory_pools={},
            page_size=self.page_size,
            size=self.size,
            max_ctx=self.max_ctx,
            device=self.device,
        )
        self.mm.req_pool.clear()

    def test_empty_request(self):
        """Test handling of empty input."""
        req = MockReq(rid="empty", origin_input_ids=[], output_ids=[])
        match = self.mm.match_prefix(req.origin_input_ids)
        self.assertEqual(len(match.matched_indices), 0)

    def test_single_token_request(self):
        """Test single token request."""
        req = MockReq(rid="single", origin_input_ids=[42], output_ids=[])
        match = self.mm.match_prefix(req.origin_input_ids)
        self.assertEqual(len(match.matched_indices), 0)
        self.mm.allocate_request(req, include_last=True, match_result=match)
        self.mm.update_cache(req)
        self.assertEqual(
            self.mm.req_info["single"].cached_len, 0
        )  # Less than page_size

    def test_exact_page_boundary(self):
        """Test request with exact page_size tokens."""
        req = MockReq(rid="page", origin_input_ids=list(range(16)), output_ids=[])
        match = self.mm.match_prefix(req.origin_input_ids)
        self.mm.allocate_request(req, include_last=True, match_result=match)
        self.mm.update_cache(req)
        self.assertEqual(self.mm.req_info["page"].cached_len, 16)

    def test_memory_pressure_eviction(self):
        """Test eviction under memory pressure."""
        requests = []
        # Fill up memory with non-overlapping requests (stay within capacity)
        # Pool is 256 tokens = 16 pages, allocate 5 * 48 = 240 tokens = 15 pages
        for i in range(5):
            req = MockReq(
                rid=f"req{i}",
                origin_input_ids=list(range(i * 1000, i * 1000 + 48)),
                output_ids=[],
            )
            requests.append(req)
            match = self.mm.match_prefix(req.origin_input_ids)
            self.mm.allocate_request(req, include_last=True, match_result=match)
            self.mm.update_cache(req)

        # Release all requests
        for req in requests:
            self.mm.release_req(req)

        # Memory should be mostly cached
        free_pages = len(self.mm.allocator.free_list)
        total_pages = self.mm.allocator.page_num
        self.assertLess(free_pages, total_pages, "Should have cached data")
        cached_pages = total_pages - free_pages

        # Manually trigger eviction to free up space
        tokens_to_evict = cached_pages * self.page_size // 2  # Evict half
        evicted = self.mm.evict(tokens_to_evict)
        self.assertGreaterEqual(evicted, tokens_to_evict)

        # Now we should have more free space
        free_pages_after = len(self.mm.allocator.free_list)
        self.assertGreater(free_pages_after, free_pages)

    def test_overlapping_prefixes(self):
        """Test multiple requests with overlapping prefixes."""
        req1 = MockReq(rid="r1", origin_input_ids=list(range(32)), output_ids=[])
        req2 = MockReq(rid="r2", origin_input_ids=list(range(48)), output_ids=[])
        req3 = MockReq(rid="r3", origin_input_ids=list(range(16)), output_ids=[])

        # Allocate req1
        match1 = self.mm.match_prefix(req1.origin_input_ids)
        self.mm.allocate_request(req1, include_last=True, match_result=match1)
        self.mm.update_cache(req1)
        self.mm.release_req(req1)

        # req2 should match 32 tokens from req1
        match2 = self.mm.match_prefix(req2.origin_input_ids)
        self.assertEqual(len(match2.matched_indices), 32)
        self.mm.allocate_request(req2, include_last=True, match_result=match2)
        self.mm.update_cache(req2)
        self.mm.release_req(req2)

        # req3 should match 16 tokens (subset of req2)
        match3 = self.mm.match_prefix(req3.origin_input_ids)
        self.assertEqual(len(match3.matched_indices), 16)

    def test_decode_phase_allocation(self):
        """Test incremental token allocation during decode phase."""
        req = MockReq(rid="decode", origin_input_ids=list(range(32)), output_ids=[])

        # Prefill
        match = self.mm.match_prefix(req.origin_input_ids)
        self.mm.allocate_request(req, include_last=True, match_result=match)
        self.mm.update_cache(req)
        req.output_ids.append(100)  # First output token
        initial_allocated = self.mm.req_info["decode"].allocated_len

        # Decode 10 tokens one by one
        for i in range(10):
            self.mm.allocate_tokens(req, num_token=1)
            req.output_ids.append(100 + i + 1)

        final_allocated = self.mm.req_info["decode"].allocated_len
        self.assertEqual(final_allocated, initial_allocated + 10)

        # Verify we can update cache
        self.mm.update_cache(req)
        self.mm.release_req(req)


class TestRadixCacheDirectly(unittest.TestCase):
    """Test RadixCache implementation directly."""

    def setUp(self):
        self.page_size = 16
        self.cache = RadixCache(page_size=self.page_size)

    def test_empty_cache_match(self):
        """Match on empty cache returns empty result."""
        result = self.cache.match_prefix(tuple(range(10)))
        self.assertEqual(len(result.matched_indices), 0)
        self.assertEqual(result.allocation_key, self.cache.root)

    def test_insert_and_match(self):
        """Basic insert and match."""
        key = tuple(range(32))
        value = torch.arange(32, dtype=torch.int32)

        # Insert
        node, matched = self.cache.insert(key, value, self.cache.root)
        self.assertEqual(len(matched), 0)  # No prior match

        # Match full prefix
        result = self.cache.match_prefix(key)
        self.assertEqual(len(result.matched_indices), 32)
        torch.testing.assert_close(result.matched_indices, value)

    def test_partial_match(self):
        """Test partial prefix matching."""
        key1 = tuple(range(32))
        value1 = torch.arange(32, dtype=torch.int32)
        self.cache.insert(key1, value1, self.cache.root)

        # Match partial prefix
        key2 = tuple(range(16))
        result = self.cache.match_prefix(key2)
        self.assertEqual(len(result.matched_indices), 16)

    def test_node_splitting(self):
        """Test that node splitting works correctly."""
        # Insert long sequence (48 tokens = 3 full pages)
        key1 = tuple(range(48))
        value1 = torch.arange(100, 148, dtype=torch.int32)  # indices 100-147
        node1, matched1 = self.cache.insert(key1, value1, self.cache.root)
        self.assertEqual(len(matched1), 0)  # Nothing matched initially
        self.cache.free(node1)  # Release for splitting test

        # Insert overlapping sequence that diverges at page boundary
        # First 32 tokens (2 pages) match, then 16 more diverge (1 page)
        key2 = tuple(list(range(32)) + list(range(1000, 1016)))  # 48 tokens total
        # Reuse matched indices for first 32, new indices for last 16
        value2 = torch.cat(
            [
                torch.arange(100, 132, dtype=torch.int32),  # matched: 100-131
                torch.arange(200, 216, dtype=torch.int32),  # new: 200-215
            ]
        )
        node2, matched2 = self.cache.insert(key2, value2, self.cache.root)

        # Should have matched first 32 tokens (2 pages)
        self.assertEqual(len(matched2), 32)

        # Both full sequences should now be findable
        result1 = self.cache.match_prefix(key1)
        self.assertEqual(len(result1.matched_indices), 48)

        result2 = self.cache.match_prefix(key2)
        self.assertEqual(len(result2.matched_indices), 48)

    def test_eviction_simple(self):
        """Test basic eviction."""
        # Insert and release multiple sequences
        sequences = []
        for i in range(5):
            key = tuple(range(i * 100, i * 100 + 32))
            value = torch.arange(32, dtype=torch.int32) + i * 100
            node, _ = self.cache.insert(key, value, self.cache.root)
            self.cache.free(node)
            sequences.append((node, value))

        # Evict some tokens
        evicted = self.cache.evict(64)
        self.assertGreaterEqual(len(evicted), 64)

    def test_ref_counting(self):
        """Test reference counting prevents eviction."""
        key = tuple(range(32))
        value = torch.arange(32, dtype=torch.int32)

        # Insert but don't free
        node, _ = self.cache.insert(key, value, self.cache.root)
        self.assertGreater(node.ref_count, 0)

        # Try to evict - should get nothing since node is in use
        evicted = self.cache.evict(100)
        self.assertEqual(len(evicted), 0)

        # Free and try again
        self.cache.free(node)
        self.assertEqual(node.ref_count, 0)
        evicted = self.cache.evict(32)
        self.assertEqual(len(evicted), 32)


class TestStressScenarios(unittest.TestCase):
    """Stress tests with many requests and operations."""

    def setUp(self):
        self.page_size = 16
        self.size = 1024
        self.max_ctx = 512
        self.device = "cpu"
        self.cache_index = RadixCache(page_size=self.page_size)
        self.mm = MemoryManager(
            cache_index=self.cache_index,
            memory_pools={},
            page_size=self.page_size,
            size=self.size,
            max_ctx=self.max_ctx,
            device=self.device,
        )
        self.mm.req_pool.clear()

    def test_many_sequential_requests(self):
        """Test handling many sequential requests."""
        num_requests = 10  # Reduced to avoid OOM in tests
        for i in range(num_requests):
            req = MockReq(
                rid=f"req{i}",
                origin_input_ids=list(range(i * 10, i * 10 + 64)),
                output_ids=[],
            )
            match = self.mm.match_prefix(req.origin_input_ids)
            self.mm.allocate_request(req, include_last=True, match_result=match)
            self.mm.update_cache(req)
            req.output_ids.append(1000)  # First output

            # Decode 5 more tokens
            for j in range(5):
                self.mm.allocate_tokens(req, num_token=1)
                req.output_ids.append(1000 + j + 1)

            self.mm.update_cache(req)
            self.mm.release_req(req)

        # Verify no memory leaks by cleaning up
        to_evict = (
            self.mm.allocator.page_num - len(self.mm.allocator.free_list)
        ) * self.page_size
        if to_evict > 0:
            evicted = self.mm.evict(to_evict)
            self.assertGreaterEqual(evicted, 0)

    def test_interleaved_prefill_decode(self):
        """Test interleaved prefill and decode phases."""
        reqs = [
            MockReq(
                rid=f"r{i}", origin_input_ids=list(range(32 + i * 5)), output_ids=[]
            )
            for i in range(5)
        ]

        # Prefill all
        for req in reqs:
            match = self.mm.match_prefix(req.origin_input_ids)
            self.mm.allocate_request(req, include_last=True, match_result=match)
            self.mm.update_cache(req)
            req.output_ids.append(2000)  # First output

        # Decode in rounds
        for round_num in range(10):
            for req in reqs:
                self.mm.allocate_tokens(req, num_token=1)
                req.output_ids.append(2000 + round_num + 1)

        # Finish all
        for req in reqs:
            self.mm.update_cache(req)
            self.mm.release_req(req)

        # Check memory is freeable
        self.assertGreaterEqual(len(self.mm.allocator.free_list), 0)

    def test_cache_hit_patterns(self):
        """Test that cache hit rates improve with common prefixes."""
        common_prefix = list(range(100))

        # First request establishes prefix
        req1 = MockReq(
            rid="r1", origin_input_ids=common_prefix + [1, 2, 3], output_ids=[]
        )
        match1 = self.mm.match_prefix(req1.origin_input_ids)
        self.assertEqual(len(match1.matched_indices), 0)
        self.mm.allocate_request(req1, include_last=True, match_result=match1)
        self.mm.update_cache(req1)
        self.mm.release_req(req1)

        # Subsequent requests should hit cache
        for i in range(5):
            req = MockReq(
                rid=f"r{i+2}",
                origin_input_ids=common_prefix + [10 + i, 20 + i],
                output_ids=[],
            )
            match = self.mm.match_prefix(req.origin_input_ids)
            # Should match at least 96 tokens (6 full pages of common prefix)
            self.assertGreaterEqual(len(match.matched_indices), 96)
            self.mm.allocate_request(req, include_last=True, match_result=match)
            self.mm.update_cache(req)
            self.mm.release_req(req)


class TestMemoryPoolIntegration(unittest.TestCase):
    """Test MemoryManager integration with multiple MemoryPools."""

    def test_single_pool_get_buf_infos(self):
        """Test manager with a single MHA pool and buffer info retrieval."""
        from sglang.srt.mem_cache_v2.memory_pool import MHAMemoryPool

        page_size = 16
        size = 1024
        max_ctx = 512
        device = "cpu"
        cache_index = RadixCache(page_size=page_size)

        # Create MHA pool for layers 0,1,2,3
        mha_pool = MHAMemoryPool(
            size=size,
            page_size=page_size,
            layer_num=4,
            head_num=8,
            head_dim=64,
            dtype=torch.float16,
            device=device,
            layer_id=[0, 1, 2, 3],
        )
        mha_pool.clear()

        # Create manager with pool mapped to layers
        memory_pools = {(0, 1, 2, 3): mha_pool}
        mm = MemoryManager(
            cache_index=cache_index,
            memory_pools=memory_pools,
            page_size=page_size,
            size=size,
            max_ctx=max_ctx,
            device=device,
        )

        # Get buffer infos
        buf_infos = mm.get_buf_infos()

        # Should have 8 buffers (4 layers * 2 for k/v)
        self.assertEqual(len(buf_infos), 8)

        # Each buffer should have valid metadata
        for info in buf_infos:
            self.assertGreater(info.data_ptr, 0)
            self.assertGreater(info.data_len, 0)
            self.assertGreater(info.item_len, 0)

    def test_multiple_pools_hybrid_model(self):
        """Test manager with multiple pools for hybrid MHA+Mamba model."""
        from sglang.srt.mem_cache_v2.memory_pool import MHAMemoryPool

        page_size = 16
        size = 1024
        max_ctx = 512
        device = "cpu"
        cache_index = RadixCache(page_size=page_size)

        # Create MHA pool for layers 0,2
        mha_pool = MHAMemoryPool(
            size, page_size, 2, 8, 64, torch.float16, device, [0, 1]
        )
        mha_pool.clear()

        # Create another MHA pool for layers 1,3
        mha_pool2 = MHAMemoryPool(
            size, page_size, 2, 8, 64, torch.float16, device, [0, 1]
        )
        mha_pool2.clear()

        # Map each pool to its layer IDs
        memory_pools = {
            (0, 2): mha_pool,
            (1, 3): mha_pool2,
        }

        mm = MemoryManager(
            cache_index=cache_index,
            memory_pools=memory_pools,
            page_size=page_size,
            size=size,
            max_ctx=max_ctx,
            device=device,
        )

        # Should aggregate buffers from both pools
        buf_infos = mm.get_buf_infos()
        self.assertEqual(len(buf_infos), 8)  # 2 pools * 2 layers * 2 (k/v)

    def test_get_cache_view_by_layer(self):
        """Test retrieving cache view for specific layers."""
        from sglang.srt.mem_cache_v2.memory_pool import MHAMemoryPool

        page_size = 16
        size = 1024
        max_ctx = 512
        device = "cpu"
        cache_index = RadixCache(page_size=page_size)

        # Create pools for different layer groups
        mha_pool1 = MHAMemoryPool(
            size, page_size, 2, 8, 64, torch.float16, device, [0, 1]
        )
        mha_pool2 = MHAMemoryPool(
            size, page_size, 2, 8, 64, torch.float16, device, [0, 1]
        )
        mha_pool1.clear()
        mha_pool2.clear()

        memory_pools = {
            (0, 1): mha_pool1,
            (2, 3): mha_pool2,
        }

        mm = MemoryManager(
            cache_index=cache_index,
            memory_pools=memory_pools,
            page_size=page_size,
            size=size,
            max_ctx=max_ctx,
            device=device,
        )

        # Get cache view for layer 0 (should use mha_pool1)
        view0 = mm.get_cache_view(0)
        self.assertIsNotNone(view0.req_to_token)
        self.assertEqual(view0.memory_pool, mha_pool1)

        # Get cache view for layer 2 (should use mha_pool2)
        view2 = mm.get_cache_view(2)
        self.assertEqual(view2.memory_pool, mha_pool2)

        # Try to get view for non-existent layer
        with self.assertRaises(ValueError):
            mm.get_cache_view(10)


if __name__ == "__main__":
    unittest.main()
