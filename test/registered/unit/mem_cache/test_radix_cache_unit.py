"""
Unit tests for the RadixCache implementation.

This module tests the core functionality of RadixCache, RadixKey, and TreeNode
following SGLang testing patterns.

Test Coverage:
- RadixKey: token ID management, slicing, iteration, representation
- TreeNode: node properties, reference counting, hash values
- RadixCache: insert/match operations, eviction, page alignment, error handling
- Cache events and request handling
- Boundary conditions with parameterized testing

Usage:
    python test_radix_cache_unit.py
    python -m pytest test_radix_cache_unit.py -v
    python -m pytest test_radix_cache_unit.py::TestRadixCache::test_insert_basic
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import random
import unittest
import unittest.mock
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    BlockStoredMetadata,
    BlockStoredWithMetadata,
    StorageMedium,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache import kv_cache_configurator
from sglang.srt.mem_cache.allocator.paged import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.mamba_radix_cache import TreeNode as MambaTreeNode
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.utils import get_device

# Test constants
DEFAULT_PAGE_SIZE = 4


class _FakeKVCache:
    def __init__(self, size: int, page_size: int):
        self.k = torch.full((size + page_size, 2), -1, dtype=torch.int64)
        self.v = torch.full((size + page_size, 2), -1, dtype=torch.int64)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.k[tgt_loc] = self.k[src_loc].clone()
        self.v[tgt_loc] = self.v[src_loc].clone()


class _ReqToTokenPool:
    def __init__(self, width: int):
        self.req_to_token = torch.full((1, width), -1, dtype=torch.int64)

    def write(self, indices, values):
        self.req_to_token[indices] = values


def _make_partial_prefix_req(prefix_indices: torch.Tensor):
    return SimpleNamespace(
        prefix_indices=prefix_indices,
        cache_protected_len=len(prefix_indices),
    )


class TestKVCacheEventQueue(unittest.TestCase):
    @staticmethod
    def _store(
        block_hash: int,
        parent_block_hash: int | None,
        *,
        block_size: int = 2,
        medium: StorageMedium = StorageMedium.GPU,
        lora_id: int | None = None,
        cache_salt: str | None = None,
    ) -> BlockStored:
        event_args = dict(
            block_hashes=[block_hash],
            parent_block_hash=parent_block_hash,
            token_ids=[block_hash, block_hash + 1][:block_size],
            block_size=block_size,
            lora_id=lora_id,
            medium=medium,
        )
        if cache_salt is None:
            return BlockStored(**event_args)
        return BlockStoredWithMetadata(
            **event_args,
            metadata=BlockStoredMetadata(cache_salt=cache_salt),
        )

    def test_enqueue_coalesces_compatible_stores(self):
        queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
        queue.enqueue(self._store(1, None))
        queue.enqueue(self._store(2, 1))

        events = queue.take()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].block_hashes, [1, 2])
        self.assertEqual(events[0].parent_block_hash, None)
        self.assertEqual(events[0].token_ids, [1, 2, 2, 3])

    def test_enqueue_coalesces_compatible_removes(self):
        queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
        queue.enqueue(BlockRemoved(block_hashes=[1], medium=StorageMedium.GPU))
        queue.enqueue(BlockRemoved(block_hashes=[2, 3], medium=StorageMedium.GPU))

        events = queue.take()
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], BlockRemoved)
        self.assertEqual(events[0].block_hashes, [1, 2, 3])

    def test_enqueue_preserves_fusion_boundaries(self):
        incompatible_stores = [
            self._store(2, 1, medium=StorageMedium.CPU),
            self._store(3, 1, lora_id=1),
            self._store(4, 1, block_size=1),
            self._store(5, None),
        ]
        for incoming in incompatible_stores:
            queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
            queue.enqueue(self._store(1, None))
            queue.enqueue(incoming)
            self.assertEqual(len(queue.take()), 2)

        queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
        queue.enqueue(self._store(1, None))
        queue.enqueue(BlockRemoved(block_hashes=[1], medium=StorageMedium.GPU))
        queue.enqueue(AllBlocksCleared())
        queue.enqueue(self._store(2, None))
        self.assertEqual(len(queue.take()), 4)

        queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
        queue.enqueue(BlockRemoved(block_hashes=[1], medium=StorageMedium.GPU))
        queue.enqueue(BlockRemoved(block_hashes=[2], medium=StorageMedium.CPU))
        self.assertEqual(len(queue.take()), 2)

        queue = KVCacheEventRecorder(enabled=True, page_size=DEFAULT_PAGE_SIZE)
        queue.enqueue(self._store(1, None, cache_salt="tenant-a"))
        queue.enqueue(self._store(2, 1, cache_salt="tenant-b"))
        self.assertEqual(len(queue.take()), 2)


class TestRadixKey(unittest.TestCase):
    """Test cases for RadixKey class."""

    def test_init_with_extra_key(self):
        """Test initialization with extra_key."""
        token_ids = [1, 2, 3]
        extra_key = "test_key"
        key = RadixKey(array("q", token_ids), extra_key)
        self.assertEqual(list(key.token_ids), token_ids)
        self.assertEqual(key.extra_key, extra_key)

    def test_len_and_iter(self):
        """Test __len__ and __iter__ methods."""
        test_cases = [
            ([1, 2, 3], 3),
            ([], 0),
            ([42], 1),
        ]

        for tokens, expected in test_cases:
            with self.subTest(tokens=tokens):
                key = RadixKey(array("q", tokens))
                self.assertEqual(len(key), expected)
                self.assertEqual(list(key), tokens)

    def test_getitem_int(self):
        """Test __getitem__ with int index."""
        test_cases = [
            ([10, 20, 30], 0, [10]),
            ([10, 20, 30], -1, [30]),
            ([10, 20, 30], 2, [30]),
        ]

        for tokens, index, expected in test_cases:
            with self.subTest(tokens=tokens, index=index):
                key = RadixKey(array("q", tokens))
                result = key[index]
                self.assertIsInstance(result, RadixKey)
                self.assertEqual(list(result.token_ids), expected)

    def test_getitem_slice(self):
        """Test __getitem__ with slice and edge cases."""
        key = RadixKey(array("q", [1, 2, 3, 4, 5]), "extra")

        # Basic slice
        sliced = key[1:4]
        self.assertIsInstance(sliced, RadixKey)
        self.assertEqual(list(sliced.token_ids), [2, 3, 4])
        self.assertEqual(sliced.extra_key, "extra")

        # Edge cases
        self.assertEqual(list(key[2:2].token_ids), [])  # Empty slice
        self.assertEqual(list(key[:].token_ids), [1, 2, 3, 4, 5])  # Full slice

    def test_cache_salt_is_preserved_by_slicing(self):
        key = RadixKey(
            array("q", [1, 2, 3, 4]),
            extra_key="classification",
            cache_salt="tenant-a",
        )
        sliced = key[1:3]
        self.assertEqual(sliced.extra_key, "classification")
        self.assertEqual(sliced.cache_salt, "tenant-a")

    def test_getitem_invalid_index(self):
        """Test __getitem__ with invalid indices."""
        key = RadixKey(array("q", [1, 2, 3]))
        with self.assertRaises(IndexError):
            _ = key[10]  # Out of bounds

    def _assert_match(
        self, a, b, page_size, expected, is_bigram=False, return_exact=False
    ):
        key_a = RadixKey(array("q", a), is_bigram=is_bigram)
        key_b = RadixKey(array("q", b), is_bigram=is_bigram)
        self.assertEqual(
            key_a.match(
                key_b,
                page_size=page_size,
                return_exact=return_exact,
            ),
            expected,
        )

    def test_match_page_size_1(self):
        """match() with page_size=1: full, partial, none, prefix, and empty keys."""
        self._assert_match([1, 2, 3, 4], [1, 2, 3, 4], 1, 4)  # identical
        self._assert_match([1, 2, 3, 4], [1, 2, 9, 9], 1, 2)  # diverge at index 2
        self._assert_match([9, 2, 3], [1, 2, 3], 1, 0)  # diverge at index 0
        self._assert_match([1, 2, 3, 4], [1, 2, 3], 1, 3)  # other is a prefix
        self._assert_match([], [1, 2], 1, 0)  # empty self
        self._assert_match([1, 2], [], 1, 0)  # empty other
        self._assert_match([], [], 1, 0)  # both empty

    def test_match_page_size_gt_1_is_aligned_by_default(self):
        self._assert_match([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 9, 8], 4, 4)
        self._assert_match([1, 2, 3, 4], [1, 9, 3, 4], 4, 0)
        self._assert_match([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 9, 6, 7, 8], 4, 4)
        self._assert_match([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], 4, 8)
        self._assert_match([1, 2, 3], [1, 2, 3], 4, 0)

    def test_match_can_return_exact_lcp(self):
        self._assert_match(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 9, 8],
            4,
            6,
            return_exact=True,
        )
        self._assert_match(
            [1, 2, 3, 4],
            [1, 9, 3, 4],
            4,
            1,
            return_exact=True,
        )
        self._assert_match([1, 2, 3], [1, 2, 3], 4, 3, return_exact=True)

    def test_match_long_keys_exponential_search(self):
        """Deep divergences exercise the doubling gallop windows + binary search.

        ``base`` has distinct values, so flipping one position diverges the prefix
        exactly there; the shared length is that exact index.
        """
        base = list(range(2000))
        for div in (1, 2, 63, 64, 65, 127, 128, 511, 512, 513, 1234, 1999):
            b = base[:]
            b[div] = -1
            for page_size in (1, 4, 64):
                with self.subTest(div=div, page_size=page_size):
                    self._assert_match(base, b, page_size, div, return_exact=True)
        # Full match of a long key: the gallop must reach the end.
        self._assert_match(base, base[:], 64, 2000, return_exact=True)

    def test_match_bigram(self):
        """is_bigram: L matching raw tokens imply L-1 matching bigrams."""
        self._assert_match([1, 2, 3, 4, 5], [1, 2, 3, 9, 5], 1, 2, is_bigram=True)
        self._assert_match([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1, 4, is_bigram=True)
        self._assert_match([1, 2], [1, 2], 1, 1, is_bigram=True)
        # Raw diverge at token 70 -> 69 matching bigrams.
        long_a = list(range(130))
        long_b = list(range(130))
        long_b[70] = -1
        self._assert_match(
            long_a,
            long_b,
            64,
            69,
            is_bigram=True,
            return_exact=True,
        )

    def test_first_page_matched_hint_preserves_lcp(self):
        base = list(range(257))
        for div in (4, 5, 7, 8, 31, 32, 127, 256, len(base)):
            other = base[:]
            if div < len(other):
                other[div] = -1
            expected = div
            generic = RadixKey(array("q", base)).match(
                RadixKey(array("q", other)), page_size=4, return_exact=True
            )
            hinted = RadixKey(array("q", base)).match(
                RadixKey(array("q", other)),
                page_size=4,
                first_page_matched=True,
                return_exact=True,
            )
            self.assertEqual(generic, expected)
            self.assertEqual(hinted, expected)


class TestTreeNode(unittest.TestCase):
    """Test cases for TreeNode class."""

    def setUp(self):
        """Reset the counter before each test."""
        TreeNode.counter = 0

    def test_init_basic(self):
        """Test basic initialization of TreeNode."""
        node = TreeNode()
        self.assertEqual(node.id, 0)
        self.assertEqual(len(node.children), 0)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.key)
        self.assertIsNone(node.value)
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(node.hit_count, 0)
        self.assertEqual(node.host_ref_counter, 0)
        self.assertIsNone(node.host_value)
        self.assertIsNone(node.hash_value)

    def test_init_with_id(self):
        """Test initialization with custom ID."""
        node = TreeNode(id=42)
        self.assertEqual(node.id, 42)
        node2 = TreeNode()
        self.assertEqual(node2.id, 1)  # Counter was incremented

    def test_evicted_backuped_properties(self):
        """Test evicted and backuped properties."""
        test_cases = [
            (False, False, True, False),
            (True, False, False, False),
            (True, True, False, True),
            (False, True, True, True),
        ]

        for (
            has_value,
            has_host_value,
            expected_evicted,
            expected_backuped,
        ) in test_cases:
            with self.subTest(has_value=has_value, has_host_value=has_host_value):
                node = TreeNode()

                if has_value:
                    node.value = torch.tensor([1, 2, 3])
                if has_host_value:
                    node.host_value = torch.tensor([4, 5, 6])

                self.assertEqual(node.evicted, expected_evicted)
                self.assertEqual(node.backuped, expected_backuped)

    def test_protect_release_host(self):
        """Test protect_host and release_host methods."""
        node = TreeNode()
        self.assertEqual(node.host_ref_counter, 0)

        node.protect_host()
        self.assertEqual(node.host_ref_counter, 1)

        node.release_host()
        self.assertEqual(node.host_ref_counter, 0)

        # Test error case
        with self.assertRaises(RuntimeError):
            node.release_host()

    def test_get_last_hash_value(self):
        """Test get_last_hash_value method."""
        node = TreeNode()
        self.assertIsNone(node.get_last_hash_value())

        node.hash_value = ["hash1", "hash2", "hash3"]
        self.assertEqual(node.get_last_hash_value(), "hash3")

    def test_get_prefix_hash_values_not_shared_across_calls(self):
        """Regression guard for cached mutable prefix hash lists."""
        for node_cls in (TreeNode, MambaTreeNode):
            with self.subTest(node_cls=node_cls.__module__):
                root = node_cls()
                n1 = node_cls()
                n1.parent = root
                n1.hash_value = ["h1"]
                n2 = node_cls()
                n2.parent = n1
                n2.hash_value = ["h2"]
                n3 = node_cls()
                n3.parent = n2
                n3.hash_value = ["h3"]

                first = n3.get_prefix_hash_values(n2)
                self.assertEqual(first, ["h1", "h2"])

                # Downstream storage code extends prefix_keys in place while
                # processing pages. A cached list must not be observable by a
                # later call.
                first += ["h3"]

                second = n3.get_prefix_hash_values(n2)
                self.assertEqual(second, ["h1", "h2"])
                self.assertIsNot(second, first)

                n4 = node_cls()
                n4.parent = n3
                n4.hash_value = ["h4"]
                self.assertEqual(n4.get_prefix_hash_values(n3), ["h1", "h2", "h3"])


class TestRadixCache(unittest.TestCase):
    """Test cases for RadixCache class."""

    def setUp(self):
        """Set up test fixtures."""
        TreeNode.counter = 0

    def _build_partial_prefix_cache(self, *, enabled: bool = True):
        page_size = 4
        pool_size = 64
        kv_cache = _FakeKVCache(pool_size, page_size)
        allocator = PagedTokenToKVPoolAllocator(
            size=pool_size,
            page_size=page_size,
            dtype=torch.float16,
            device="cpu",
            kvcache=kv_cache,
            need_sort=False,
        )
        cache = RadixCache.create_simulated(
            mock_allocator=allocator,
            page_size=page_size,
            enable_partial_prefix_reuse=enabled,
        )
        cached_tokens = array("q", range(1, 13))
        cached_indices = allocator.alloc(len(cached_tokens))
        assert cached_indices is not None
        kv_cache.k[cached_indices] = torch.arange(24, dtype=torch.int64).view(12, 2)
        kv_cache.v[cached_indices] = torch.arange(100, 124, dtype=torch.int64).view(
            12, 2
        )
        cache.insert(
            InsertParams(
                key=RadixKey(cached_tokens),
                value=cached_indices,
            )
        )
        query_tokens = array("q", [*range(1, 11), 101, 102, 103])
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(query_tokens)))
        return cache, allocator, kv_cache, cached_indices, query_tokens, result

    def _build_split_child_partial_cache(
        self,
        *,
        workload: str,
        partial_reuse: bool,
    ):
        page_size = 4
        pool_size = 128
        kv_cache = _FakeKVCache(pool_size, page_size)
        allocator = PagedTokenToKVPoolAllocator(
            size=pool_size,
            page_size=page_size,
            dtype=torch.float16,
            device="cpu",
            kvcache=kv_cache,
            need_sort=False,
        )
        cache = RadixCache.create_simulated(
            mock_allocator=allocator,
            page_size=page_size,
            enable_partial_prefix_reuse=partial_reuse,
        )
        common = [1, 2, 3, 4]
        if workload == "lookup_limited":
            target_tail = [5, 6, 7, 8]
            sibling_tail = [105, 106, 107, 108]
            query_tokens = array("q", common + target_tail[:2] + [999, 1000, 1001])
        elif workload == "legacy_reachable":
            target_tail = [5, 6, 7, 8, 9, 10, 11, 12]
            sibling_tail = [105, 106, 107, 108, 109, 110, 111, 112]
            query_tokens = array("q", common + target_tail[:6] + [999, 1000, 1001])
        else:
            raise AssertionError(f"unknown workload {workload!r}")

        target_tokens = array("q", common + target_tail)
        sibling_tokens = array("q", common + sibling_tail)
        target_indices = allocator.alloc(len(target_tokens))
        sibling_indices = allocator.alloc(len(sibling_tokens))
        self.assertIsNotNone(target_indices)
        self.assertIsNotNone(sibling_indices)
        cache.insert(InsertParams(key=RadixKey(target_tokens), value=target_indices))
        cache.insert(InsertParams(key=RadixKey(sibling_tokens), value=sibling_indices))
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(query_tokens)))
        return cache, target_indices, query_tokens, result

    def test_init_variations(self):
        """Test cache initialization with different parameters."""
        test_cases = [
            (1, False, False),
            (4, False, True),
            (1, True, False),
        ]

        for page_size, disable, enable_events in test_cases:
            with self.subTest(
                page_size=page_size, disable=disable, enable_events=enable_events
            ):
                cache = RadixCache.create_simulated(
                    disable=disable,
                    page_size=page_size,
                    enable_kv_cache_events=enable_events,
                )

                self.assertEqual(cache.page_size, page_size)
                self.assertEqual(cache.disable, disable)
                self.assertEqual(cache.kv_events.enabled, enable_events)
                self.assertEqual(cache.device, torch.device("cpu"))
                self.assertIsNotNone(cache.root_node)
                self.assertEqual(len(cache.root_node.key), 0)

    def test_partial_prefix_rejects_unsupported_production_pool(self):
        allocator = PagedTokenToKVPoolAllocator(
            size=16,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=_FakeKVCache(size=16, page_size=4),
            need_sort=False,
        )
        with self.assertRaisesRegex(
            ValueError,
            "enable_partial_prefix_reuse is unsupported.*device cpu.*_FakeKVCache",
        ):
            RadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=object(),
                    token_to_kv_pool_allocator=allocator,
                    page_size=4,
                    enable_partial_prefix_reuse=True,
                )
            )

    def test_partial_prefix_enables_copy_primitive_for_ordinary_mha_pool(self):
        for partial_reuse, speculative, expected in (
            (False, None, False),
            (True, None, True),
            (False, object(), True),
        ):
            with self.subTest(
                partial_reuse=partial_reuse,
                speculative=speculative is not None,
            ):
                captured = {}

                def pool_cls(*args, **kwargs):
                    captured.update(kwargs)
                    return object()

                configurator = SimpleNamespace(
                    kv_cache_dtype_str="bfloat16",
                    pool_page_size=4,
                    kv_cache_dtype=torch.bfloat16,
                    model_config=SimpleNamespace(
                        get_num_kv_heads=lambda *_: 8,
                        head_dim=128,
                        v_head_dim=128,
                    ),
                    layer_info=SimpleNamespace(
                        num_effective_layers=32,
                        start_layer=0,
                        end_layer=32,
                    ),
                    device="cuda",
                    post_capture_kv_active=False,
                    server_args=SimpleNamespace(
                        enable_partial_prefix_reuse=partial_reuse
                    ),
                )
                with (
                    unittest.mock.patch.object(
                        kv_cache_configurator,
                        "get_schedule",
                        return_value=SimpleNamespace(
                            prefill_only_disable_kv_cache=False
                        ),
                    ),
                    unittest.mock.patch.object(
                        kv_cache_configurator,
                        "get_parallel",
                        return_value=SimpleNamespace(attn_tp_size=1, attn_dcp_size=1),
                    ),
                    unittest.mock.patch.object(
                        kv_cache_configurator,
                        "get_exec",
                        return_value=SimpleNamespace(
                            features=SimpleNamespace(enable_memory_saver=False)
                        ),
                    ),
                    unittest.mock.patch.object(
                        kv_cache_configurator,
                        "get_disagg",
                        return_value=SimpleNamespace(enable_pdmux=False),
                    ),
                    unittest.mock.patch.object(
                        kv_cache_configurator,
                        "get_spec",
                        return_value=SimpleNamespace(speculative_algorithm=speculative),
                    ),
                ):
                    KVCacheConfigurator._build_mha_kv_pool(
                        configurator,
                        max_total_num_tokens=64,
                        mha_pool_class=pool_cls,
                    )

                self.assertEqual(captured["enable_kv_cache_copy"], expected)

    def test_reset(self):
        """Test reset method."""
        cache = RadixCache.create_simulated()

        # Insert some data
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3])),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        self.assertGreater(cache.total_size(), 0)

        # Reset
        cache.reset()
        self.assertEqual(cache.total_size(), 0)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 0)

    def test_insert_and_match_basic(self):
        """Test basic insert and match operations."""
        for disable_cache in [False, True]:
            with self.subTest(disable_cache=disable_cache):
                cache = RadixCache.create_simulated(disable=disable_cache)

                key = RadixKey(array("q", [1, 2, 3]))
                value = torch.tensor([10, 20, 30], dtype=torch.int64)
                result = cache.insert(InsertParams(key=key, value=value))
                prefix_len = result.prefix_len

                if disable_cache:
                    self.assertEqual(prefix_len, 0)
                    self.assertEqual(cache.total_size(), 0)
                    continue

                self.assertEqual(prefix_len, 0)  # No existing prefix
                self.assertEqual(cache.total_size(), 3)
                self.assertEqual(cache.evictable_size(), 3)

                # Test match_prefix
                result = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3])))
                )
                self.assertEqual(len(result.device_indices), 3)
                torch.testing.assert_close(result.device_indices, value)

                # Test partial match
                result = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", [1, 2])))
                )
                self.assertEqual(len(result.device_indices), 2)
                torch.testing.assert_close(
                    result.device_indices, torch.tensor([10, 20], dtype=torch.int64)
                )

    def test_insert_with_none_value(self):
        """Test insert with None value (should use token_ids as list)."""
        cache = RadixCache.create_simulated()

        key = RadixKey(array("q", [1, 2, 3]))
        result = cache.insert(InsertParams(key=key, value=None))
        prefix_len = result.prefix_len

        # When None is passed, it should create value from token_ids
        self.assertEqual(prefix_len, 0)
        self.assertEqual(cache.total_size(), 3)

    def test_total_size(self):
        """Test total_size calculation."""
        cache = RadixCache.create_simulated()

        self.assertEqual(cache.total_size(), 0)

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3])),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        self.assertEqual(cache.total_size(), 3)

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [4, 5])),
                value=torch.tensor([40, 50], dtype=torch.int64),
            )
        )
        self.assertEqual(cache.total_size(), 5)

    def test_cache_unfinished_req_deferred_free_owns_original_indices(self):
        class ReqToTokenPool:
            def __init__(self, row):
                self.req_to_token = row.unsqueeze(0)

            def write(self, indices, values):
                self.req_to_token[indices] = values

        allocator = TokenToKVPoolAllocator(
            size=16,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        token_ids = array("q", [1, 2, 3])
        tree_indices = allocator.alloc(3)
        request_indices = allocator.alloc(3)
        assert tree_indices is not None
        assert request_indices is not None
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", token_ids)),
                value=tree_indices,
            )
        )
        cache.req_to_token_pool = ReqToTokenPool(request_indices.clone())
        req = unittest.mock.Mock(
            req_pool_idx=0,
            cache_protected_len=0,
            extra_key=None,
            cache_salt=None,
            priority=0,
            last_node=cache.root_node,
        )
        req.get_fill_ids.return_value = token_ids

        available_before_free = allocator.available_size()
        allocator.free_group_begin()
        cache.cache_unfinished_req(req)
        allocator.free_group_end()

        self.assertEqual(
            allocator.available_size(),
            available_before_free + request_indices.numel(),
        )
        torch.testing.assert_close(allocator.free_pages[-3:], request_indices)
        torch.testing.assert_close(
            cache.req_to_token_pool.req_to_token[0], tree_indices
        )

    def test_kv_cache_events(self):
        """Test KV cache events functionality."""
        test_cases = [
            (1, True),
            (2, True),
            (1, False),
        ]

        for page_size, enable_events in test_cases:
            with self.subTest(page_size=page_size, enable_events=enable_events):
                cache = RadixCache.create_simulated(
                    page_size=page_size, enable_kv_cache_events=enable_events
                )

                # Insert data
                cache.insert(
                    InsertParams(key=RadixKey(array("q", [1, 2, 3, 4, 5])), value=None)
                )

                # Take events
                events = cache.take_events()

                if enable_events:
                    self.assertGreater(len(events), 0)
                    # Verify events include BlockStored events (there might be other event types)
                    block_stored_events = [
                        e for e in events if isinstance(e, BlockStored)
                    ]
                    self.assertGreater(len(block_stored_events), 0)
                    for event in block_stored_events:
                        self.assertLessEqual(event.block_size, page_size)
                        self.assertEqual(
                            len(event.token_ids),
                            event.block_size * len(event.block_hashes),
                        )
                else:
                    self.assertEqual(len(events), 0)

    def test_kv_cache_events_with_eviction(self):
        """Test KV cache events include removal events."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache.create_simulated(
            mock_allocator=mock_allocator,
            page_size=2,
            enable_kv_cache_events=True,
        )

        # Insert and then evict data
        seq = [1, 2, 3, 4]
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", seq)),
                value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
            )
        )
        result = cache.evict(EvictParams(num_tokens=len(seq)))
        self.assertIsInstance(result, EvictResult)
        self.assertGreaterEqual(
            result.num_tokens_evicted,
            len(seq),
            f"evicted {result.num_tokens_evicted} tokens, expected at least {len(seq)}",
        )

        # Take events - should include both store and remove events
        events = cache.take_events()
        self.assertGreater(len(events), 0)

        # Check event types
        event_types = [type(event).__name__ for event in events]
        self.assertIn("BlockStored", event_types)

        stored_hashes = [
            block_hash
            for event in events
            if isinstance(event, BlockStored)
            for block_hash in event.block_hashes
        ]
        self.assertEqual(len(stored_hashes), 2)

        # Verify BlockRemoved event content
        remove_events = [e for e in events if isinstance(e, BlockRemoved)]
        self.assertEqual(len(remove_events), 1)
        self.assertEqual(remove_events[0].block_hashes, stored_hashes)

    def test_extra_key_isolation(self):
        """Test that keys with different extra_key values are isolated."""
        cache = RadixCache.create_simulated()

        # Insert same token sequence with different extra keys
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3]), "key1"),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3]), "key2"),
                value=torch.tensor([40, 50, 60], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3]), None),
                value=torch.tensor([70, 80, 90], dtype=torch.int64),
            )
        )

        # Keys with different extra_key should not match each other
        result1 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), "key1"))
        )
        result2 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), "key2"))
        )
        result3 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), None))
        )
        result4 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), "nonexistent"))
        )

        # Each should match only its own data
        self.assertEqual(len(result1.device_indices), 3)
        torch.testing.assert_close(
            result1.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64)
        )

        self.assertEqual(len(result2.device_indices), 3)
        torch.testing.assert_close(
            result2.device_indices, torch.tensor([40, 50, 60], dtype=torch.int64)
        )

        self.assertEqual(len(result3.device_indices), 3)
        torch.testing.assert_close(
            result3.device_indices, torch.tensor([70, 80, 90], dtype=torch.int64)
        )

        # Non-existent extra_key should not match
        self.assertEqual(len(result4.device_indices), 0)

    def test_cache_salt_isolation_is_independent_of_extra_key(self):
        cache = RadixCache.create_simulated()
        tokens = array("q", [1, 2, 3])

        cache.insert(
            InsertParams(
                key=RadixKey(tokens, extra_key="bc", cache_salt="a"),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(tokens, extra_key="c", cache_salt="ab"),
                value=torch.tensor([40, 50, 60], dtype=torch.int64),
            )
        )

        first = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens, extra_key="bc", cache_salt="a"))
        )
        second = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens, extra_key="c", cache_salt="ab"))
        )
        torch.testing.assert_close(
            first.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64)
        )
        torch.testing.assert_close(
            second.device_indices, torch.tensor([40, 50, 60], dtype=torch.int64)
        )

    def test_cache_salt_is_included_in_store_and_remove_events(self):
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")
        cache = RadixCache.create_simulated(
            mock_allocator=mock_allocator,
            page_size=2,
            enable_kv_cache_events=True,
        )
        tokens = array("q", [1, 2, 3, 4])
        cache.insert(
            InsertParams(
                key=RadixKey(tokens, cache_salt="tenant-a"),
                value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
            )
        )
        cache.evict(EvictParams(num_tokens=len(tokens)))
        events = cache.take_events()
        stored = [event for event in events if isinstance(event, BlockStored)]
        removed = [event for event in events if isinstance(event, BlockRemoved)]

        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].metadata.cache_salt, "tenant-a")
        self.assertEqual(stored[0].parent_block_hash, None)
        self.assertEqual(len(stored[0].block_hashes), 2)
        self.assertEqual(removed[0].block_hashes, stored[0].block_hashes)

        unsalted = RadixCache.create_simulated(page_size=2, enable_kv_cache_events=True)
        unsalted.insert(InsertParams(key=RadixKey(tokens), value=None))
        unsalted_hashes = [
            block_hash
            for event in unsalted.take_events()
            if isinstance(event, BlockStored)
            for block_hash in event.block_hashes
        ]
        self.assertNotEqual(unsalted_hashes, stored[0].block_hashes)

    def test_cache_salt_event_hashes_are_preserved_across_node_split(self):
        cache = RadixCache.create_simulated(page_size=2, enable_kv_cache_events=True)
        original = RadixKey(array("q", [1, 2, 3, 4]), cache_salt="tenant-a")
        cache.insert(
            InsertParams(
                key=original,
                value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
            )
        )
        original_node = cache.match_prefix(
            MatchPrefixParams(key=original)
        ).last_device_node
        original_hashes = list(original_node.event_hash_value)

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 9, 10]), cache_salt="tenant-a"),
                value=torch.tensor([10, 20, 90, 100], dtype=torch.int64),
            )
        )
        split_child = cache.match_prefix(
            MatchPrefixParams(key=original)
        ).last_device_node
        split_parent = split_child.parent

        self.assertEqual(
            split_parent.event_hash_value + split_child.event_hash_value,
            original_hashes,
        )

    def test_lock_ref_operations(self):
        """Test lock reference counting operations."""
        cache = RadixCache.create_simulated()

        # Insert sequence
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3])),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )

        # Get node
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3])))
        )
        node = result.last_device_node

        initial_evictable = cache.evictable_size()
        initial_protected = cache.protected_size()

        # Lock the node
        cache.inc_lock_ref(node)
        self.assertEqual(cache.protected_size(), initial_protected + 3)
        self.assertEqual(cache.evictable_size(), initial_evictable - 3)

        # Unlock the node
        cache.dec_lock_ref(node)
        self.assertEqual(cache.protected_size(), initial_protected)
        self.assertEqual(cache.evictable_size(), initial_evictable)

    def test_evict_functionality(self):
        """Test eviction functionality."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache.create_simulated(mock_allocator=mock_allocator)

        # Insert sequences
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2])),
                value=torch.tensor([10, 20], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [3, 4])),
                value=torch.tensor([30, 40], dtype=torch.int64),
            )
        )

        initial_size = cache.total_size()

        # Evict some tokens
        result = cache.evict(EvictParams(num_tokens=2))
        self.assertIsInstance(result, EvictResult)
        self.assertGreaterEqual(
            result.num_tokens_evicted,
            2,
            f"evicted {result.num_tokens_evicted} tokens, expected at least 2",
        )

        # Should have called free_segment and reduced size
        mock_allocator.free_segment.assert_called()
        self.assertLess(cache.total_size(), initial_size)

    def test_page_alignment_boundary(self):
        """Test page alignment with different sizes."""
        test_cases = [
            (1, 5),
            (2, 5),
            (4, 6),
        ]

        for page_size, sequence_length in test_cases:
            with self.subTest(page_size=page_size, sequence_length=sequence_length):
                cache = RadixCache.create_simulated(page_size=page_size)

                tokens = list(range(sequence_length))
                key = RadixKey(array("q", tokens))
                cache.insert(
                    InsertParams(
                        key=key,
                        value=torch.tensor(tokens, dtype=torch.int64)[: len(key)],
                    )
                )

                result = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", tokens)))
                )
                self.assertGreater(len(result.device_indices), 0)

                # Match length should be page-aligned
                match_len = len(result.device_indices)
                self.assertEqual(match_len % page_size, 0)

    def test_partial_prefix_match_reports_exact_lcp_without_unaligned_tree_split(self):
        cache, _, _, cached_indices, _, result = self._build_partial_prefix_cache()

        self.assertEqual(len(result.device_indices), 8)
        partial_match = result.partial_prefix_match
        self.assertIsNotNone(partial_match)
        self.assertEqual(partial_match.exact_match_len, 10)
        torch.testing.assert_close(partial_match.source_indices, cached_indices[8:10])

        aligned_node = result.last_device_node
        self.assertEqual(len(aligned_node.key), 8)
        self.assertEqual(len(aligned_node.children), 1)
        source_node = next(iter(aligned_node.children.values()))
        self.assertIs(partial_match.source_node, source_node)
        self.assertEqual(len(source_node.key), 4)
        self.assertTrue(
            all(len(node.key) % 4 == 0 for node in (aligned_node, source_node))
        )
        self.assertEqual(cache.total_size(), 12)

    def test_partial_reuse_bundles_discovery_and_private_page_copy(self):
        for workload in ("lookup_limited", "legacy_reachable"):
            for partial_reuse in (False, True):
                with self.subTest(workload=workload, partial_reuse=partial_reuse):
                    cache, target_indices, _, result = (
                        self._build_split_child_partial_cache(
                            workload=workload,
                            partial_reuse=partial_reuse,
                        )
                    )
                    req = _make_partial_prefix_req(result.device_indices)
                    copied = cache.materialize_partial_prefix(req, result)

                    if workload == "lookup_limited":
                        expected_exact = 6 if partial_reuse else 4
                        expected_kind = "fine_lookup" if partial_reuse else None
                        expected_source = target_indices[4:6] if partial_reuse else None
                        expected_copy = partial_reuse
                        expected_prefix_len = 6 if expected_copy else 4
                    else:
                        expected_exact = 10 if partial_reuse else 8
                        expected_kind = "legacy_reachable" if partial_reuse else None
                        expected_source = (
                            target_indices[8:10] if partial_reuse else None
                        )
                        expected_copy = partial_reuse
                        expected_prefix_len = 10 if expected_copy else 8

                    partial_match = result.partial_prefix_match
                    actual_exact = (
                        partial_match.exact_match_len
                        if partial_match is not None
                        else len(result.device_indices)
                    )
                    actual_kind = (
                        partial_match.match_kind if partial_match is not None else None
                    )
                    self.assertEqual(actual_exact, expected_exact)
                    self.assertEqual(actual_kind, expected_kind)
                    self.assertEqual(copied, expected_copy)
                    self.assertEqual(len(req.prefix_indices), expected_prefix_len)
                    if expected_source is None:
                        self.assertIsNone(partial_match)
                    else:
                        torch.testing.assert_close(
                            partial_match.source_indices, expected_source
                        )

                    # Tree-owned match output remains page aligned in all modes.
                    self.assertEqual(len(result.device_indices) % 4, 0)
                    cache.abort_partial_prefix(req)

    def test_radix_cache_child_lookup_uses_first_page_hint(self):
        seen_hints = []
        original_match = RadixKey.match

        def recording_match(
            key,
            other,
            page_size=1,
            first_page_matched=False,
            return_exact=False,
        ):
            seen_hints.append(first_page_matched)
            return original_match(
                key,
                other,
                page_size=page_size,
                first_page_matched=first_page_matched,
                return_exact=return_exact,
            )

        with unittest.mock.patch.object(RadixKey, "match", new=recording_match):
            self._build_split_child_partial_cache(
                workload="legacy_reachable",
                partial_reuse=False,
            )

        self.assertTrue(seen_hints)
        self.assertTrue(all(seen_hints))

    def test_exact_matching_and_child_scan_are_feature_gated(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                exact_requests = []
                original_match = RadixKey.match

                def recording_match(
                    key,
                    other,
                    page_size=1,
                    first_page_matched=False,
                    return_exact=False,
                    _exact_requests=exact_requests,
                    _original_match=original_match,
                ):
                    _exact_requests.append(return_exact)
                    return _original_match(
                        key,
                        other,
                        page_size=page_size,
                        first_page_matched=first_page_matched,
                        return_exact=return_exact,
                    )

                with unittest.mock.patch.object(RadixKey, "match", new=recording_match):
                    _, _, _, result = self._build_split_child_partial_cache(
                        workload="lookup_limited",
                        partial_reuse=enabled,
                    )

                self.assertEqual(any(exact_requests), enabled)
                self.assertEqual(result.partial_prefix_match is not None, enabled)

    def test_partial_prefix_copy_continuation_and_req_mapping(self):
        cache, allocator, kv_cache, cached_indices, _, result = (
            self._build_partial_prefix_cache()
        )
        req = _make_partial_prefix_req(result.device_indices)
        source_k_before = kv_cache.k[cached_indices[8:12]].clone()
        source_v_before = kv_cache.v[cached_indices[8:12]].clone()
        available_before = allocator.available_size()

        self.assertTrue(cache.materialize_partial_prefix(req, result))
        self.assertEqual(len(req.prefix_indices), 10)
        self.assertEqual(req.cache_protected_len, 8)
        self.assertEqual(allocator.available_size(), available_before - 4)
        self.assertEqual(cache.total_size(), 12)

        state = req._partial_prefix_copy_state
        dst_page = state.page_indices
        src = state.source_indices
        dst = state.destination_indices
        self.assertFalse(torch.equal(dst_page, cached_indices[8:12]))
        self.assertEqual(int(dst_page[0]) // 4, int(dst_page[-1]) // 4)

        kv_cache.move_kv_cache(dst, src)
        torch.testing.assert_close(kv_cache.k[dst], kv_cache.k[src])
        torch.testing.assert_close(kv_cache.v[dst], kv_cache.v[src])
        torch.testing.assert_close(kv_cache.k[cached_indices[8:12]], source_k_before)
        torch.testing.assert_close(kv_cache.v[cached_indices[8:12]], source_v_before)

        continuation = torch.empty(2, dtype=torch.int64)
        alloc_extend_naive(
            prefix_lens=torch.tensor([10]),
            seq_lens=torch.tensor([12]),
            last_loc=dst[-1:].clone(),
            free_pages=allocator.free_pages,
            out_indices=continuation,
            page_size=4,
            device="cpu",
        )
        torch.testing.assert_close(continuation, dst_page[2:4])
        kv_cache.k[continuation] = torch.tensor([[1001, 1002], [1003, 1004]])
        kv_cache.v[continuation] = torch.tensor([[2001, 2002], [2003, 2004]])
        torch.testing.assert_close(kv_cache.k[dst_page[2:4]], kv_cache.k[continuation])
        torch.testing.assert_close(kv_cache.k[cached_indices[8:12]], source_k_before)

        req_to_token = _ReqToTokenPool(width=16)
        req_to_token.write((0, slice(0, 10)), req.prefix_indices)
        req_to_token.write((0, slice(10, 12)), continuation)
        torch.testing.assert_close(
            req_to_token.req_to_token[0, :10], req.prefix_indices
        )
        torch.testing.assert_close(req_to_token.req_to_token[0, 10:12], dst_page[2:4])

    def test_deferred_partial_prefix_copy_collection_is_overlap_safe(self):
        source_node = object()
        state = SimpleNamespace(
            source_indices=torch.tensor([10, 11], dtype=torch.int64),
            destination_indices=torch.tensor([20, 21], dtype=torch.int64),
            source_node=source_node,
            page_indices=torch.tensor([20, 21, 22, 23], dtype=torch.int64),
        )
        req = SimpleNamespace(_partial_prefix_copy_state=state)
        batch = SimpleNamespace()

        ScheduleBatch._collect_deferred_partial_prefix_copy(batch, [req])
        torch.testing.assert_close(
            batch.partial_prefix_copy_src_indices,
            torch.tensor([10, 11], dtype=torch.int64),
        )
        torch.testing.assert_close(
            batch.partial_prefix_copy_dst_indices,
            torch.tensor([20, 21], dtype=torch.int64),
        )
        self.assertIsNone(state.source_indices)
        self.assertIsNone(state.destination_indices)
        self.assertIsNone(state.page_indices)
        self.assertIs(state.source_node, source_node)

        # A later chunk may be prepared before the earlier forward result
        # releases source_node. Collection must be idempotent in that window.
        ScheduleBatch._collect_deferred_partial_prefix_copy(batch, [req])
        self.assertIsNone(batch.partial_prefix_copy_src_indices)
        self.assertIsNone(batch.partial_prefix_copy_dst_indices)
        self.assertIs(state.source_node, source_node)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None,
        "requires standard CUDA",
    )
    def test_partial_prefix_uses_real_cuda_all_layer_copy(self):
        page_size = 4
        pool_size = 64
        kv_cache = MHATokenToKVPool(
            size=pool_size,
            page_size=page_size,
            dtype=torch.float16,
            head_num=2,
            head_dim=16,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
            enable_kv_cache_copy=True,
            kv_cache_layout="nhd",
        )
        allocator = PagedTokenToKVPoolAllocator(
            size=pool_size,
            page_size=page_size,
            dtype=torch.float16,
            device="cuda",
            kvcache=kv_cache,
            need_sort=False,
        )
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=object(),
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                enable_partial_prefix_reuse=True,
            )
        )

        cached_indices = allocator.alloc(12)
        self.assertIsNotNone(cached_indices)
        for layer_id, (k_buffer, v_buffer) in enumerate(
            zip(kv_cache.k_buffer, kv_cache.v_buffer, strict=True)
        ):
            values = torch.arange(12 * 2 * 16, dtype=torch.float16, device="cuda").view(
                12, 2, 16
            )
            k_buffer[cached_indices] = values + layer_id * 1000
            v_buffer[cached_indices] = values + layer_id * 2000 + 500

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(1, 13))),
                value=cached_indices,
            )
        )
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [*range(1, 11), 101, 102])))
        )
        req = _make_partial_prefix_req(result.device_indices)
        self.assertTrue(cache.materialize_partial_prefix(req, result))
        source_before = [
            (k[cached_indices[8:12]].clone(), v[cached_indices[8:12]].clone())
            for k, v in zip(kv_cache.k_buffer, kv_cache.v_buffer, strict=True)
        ]

        kv_cache.move_kv_cache(
            req._partial_prefix_copy_state.destination_indices,
            req._partial_prefix_copy_state.source_indices,
        )
        torch.cuda.synchronize()

        for (k_buffer, v_buffer), (source_k, source_v) in zip(
            zip(kv_cache.k_buffer, kv_cache.v_buffer, strict=True),
            source_before,
            strict=True,
        ):
            torch.testing.assert_close(
                k_buffer[req._partial_prefix_copy_state.destination_indices],
                source_k[:2],
            )
            torch.testing.assert_close(
                v_buffer[req._partial_prefix_copy_state.destination_indices],
                source_v[:2],
            )
            torch.testing.assert_close(k_buffer[cached_indices[8:12]], source_k)
            torch.testing.assert_close(v_buffer[cached_indices[8:12]], source_v)

        cache.release_partial_prefix_source(req)

    def test_partial_prefix_source_lock_blocks_eviction_until_copy_release(self):
        cache, _, kv_cache, _, _, result = self._build_partial_prefix_cache()
        req = _make_partial_prefix_req(result.device_indices)
        self.assertEqual(cache.evictable_size(), 12)
        self.assertEqual(cache.protected_size(), 0)
        self.assertTrue(cache.materialize_partial_prefix(req, result))
        state = req._partial_prefix_copy_state
        source_node = state.source_node
        self.assertGreater(source_node.lock_ref, 0)
        self.assertEqual(cache.total_size(), 12)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 12)

        evicted = cache.evict(EvictParams(num_tokens=cache.total_size()))
        self.assertEqual(evicted.num_tokens_evicted, 0)
        self.assertEqual(cache.total_size(), 12)

        kv_cache.move_kv_cache(
            state.destination_indices,
            state.source_indices,
        )
        cache.release_partial_prefix_source(req)
        self.assertEqual(source_node.lock_ref, 0)
        self.assertEqual(cache.evictable_size(), 12)
        self.assertEqual(cache.protected_size(), 0)
        evicted = cache.evict(EvictParams(num_tokens=cache.total_size()))
        self.assertEqual(evicted.num_tokens_evicted, 12)
        self.assertEqual(cache.total_size(), 0)

    def test_partial_prefix_allocation_failure_falls_back_without_lock_leak(self):
        page_size = 4
        kv_cache = _FakeKVCache(size=12, page_size=page_size)
        allocator = PagedTokenToKVPoolAllocator(
            size=12,
            page_size=page_size,
            dtype=torch.float16,
            device="cpu",
            kvcache=kv_cache,
            need_sort=False,
        )
        cache = RadixCache.create_simulated(
            mock_allocator=allocator,
            page_size=page_size,
            enable_partial_prefix_reuse=True,
        )
        cached_indices = allocator.alloc(12)
        self.assertIsNotNone(cached_indices)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(1, 13))),
                value=cached_indices,
            )
        )
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [*range(1, 11), 101, 102])))
        )
        req = _make_partial_prefix_req(result.device_indices)

        self.assertEqual(allocator.available_size(), 0)
        self.assertFalse(cache.materialize_partial_prefix(req, result))
        self.assertEqual(len(req.prefix_indices), 8)
        self.assertFalse(hasattr(req, "_partial_prefix_copy_state"))
        self.assertEqual(cache.evictable_size(), 12)
        self.assertEqual(cache.protected_size(), 0)

    def test_partial_prefix_finish_inserts_private_page_once(self):
        cache, allocator, kv_cache, _, query_tokens, result = (
            self._build_partial_prefix_cache()
        )
        req = _make_partial_prefix_req(result.device_indices)
        self.assertTrue(cache.materialize_partial_prefix(req, result))
        state = req._partial_prefix_copy_state
        dst_page = state.page_indices
        kv_cache.move_kv_cache(
            state.destination_indices,
            state.source_indices,
        )
        continuation = dst_page[2:4]
        kv_cache.k[continuation] = torch.tensor([[3001, 3002], [3003, 3004]])
        kv_cache.v[continuation] = torch.tensor([[4001, 4002], [4003, 4004]])
        cache.release_partial_prefix_source(req)

        req_to_token = _ReqToTokenPool(width=16)
        req_to_token.write((0, slice(0, 10)), req.prefix_indices)
        req_to_token.write((0, slice(10, 12)), continuation)
        cache.req_to_token_pool = req_to_token
        cache.inc_lock_ref(result.last_device_node)
        req.req_pool_idx = 0
        req.last_node = result.last_device_node
        req.origin_input_ids = query_tokens[:12]
        req.output_ids = array("q")
        req.extra_key = None
        req.cache_salt = None
        req.priority = 0

        cache.cache_finished_req(req, kv_len_to_handle=12)
        self.assertEqual(cache.total_size(), 16)
        self.assertEqual(allocator.available_size() + cache.total_size(), 64)
        new_match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(query_tokens[:12]))
        )
        self.assertEqual(len(new_match.device_indices), 12)
        torch.testing.assert_close(new_match.device_indices[8:12], dst_page)

    def test_partial_prefix_flag_changes_extend_start_only_after_copy(self):
        off_cache, _, _, _, query_tokens, off_result = self._build_partial_prefix_cache(
            enabled=False
        )
        self.assertEqual(len(off_result.device_indices), 8)
        self.assertIsNone(off_result.partial_prefix_match)
        self.assertEqual(
            list(query_tokens[len(off_result.device_indices) :])[:2], [9, 10]
        )
        self.assertEqual(off_cache.total_size(), 12)

        on_cache, _, _, _, _, on_result = self._build_partial_prefix_cache(enabled=True)
        req = _make_partial_prefix_req(on_result.device_indices)
        on_cache.materialize_partial_prefix(req, on_result)
        self.assertEqual(len(req.prefix_indices), 10)
        self.assertEqual(list(query_tokens[len(req.prefix_indices) :])[:2], [101, 102])
        on_cache.abort_partial_prefix(req)

    def test_schedule_batch_flag_controls_exact_reuse_extend_start(self):
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.utils.common import Range

        class _StopBeforeAllocation(Exception):
            pass

        for enabled, expected_prefix_len, expected_input_ids in (
            (False, 8, [9, 10, 101, 102, 103]),
            (True, 10, [101, 102, 103]),
        ):
            with self.subTest(enabled=enabled):
                cache, _, _, _, query_tokens, result = self._build_partial_prefix_cache(
                    enabled=enabled
                )
                req = _make_partial_prefix_req(result.device_indices)
                cache.materialize_partial_prefix(req, result)
                req.origin_input_ids = query_tokens
                req.full_untruncated_fill_ids = query_tokens
                req.extend_range = Range(len(req.prefix_indices), len(query_tokens))
                req.logprob_start_len = -1
                req.get_fill_ids = lambda query_tokens=query_tokens: query_tokens
                batch = ScheduleBatch(reqs=[req], tree_cache=cache, device="cpu")
                captured = {}
                query_len = len(query_tokens)

                def capture_input_ids(input_ids, _pin, captured=captured):
                    captured["input_ids"] = [list(ids) for ids in input_ids]
                    return torch.tensor(
                        [token for ids in input_ids for token in ids],
                        dtype=torch.int64,
                    )

                def stop_at_allocation(
                    batch_to_allocate,
                    expected_prefix_len=expected_prefix_len,
                    query_len=query_len,
                ):
                    self.assertEqual(
                        batch_to_allocate.prefix_lens, [expected_prefix_len]
                    )
                    self.assertEqual(
                        batch_to_allocate.extend_lens,
                        [query_len - expected_prefix_len],
                    )
                    self.assertEqual(
                        batch_to_allocate.extend_num_tokens,
                        query_len - expected_prefix_len,
                    )
                    raise _StopBeforeAllocation

                with (
                    unittest.mock.patch(
                        "sglang.srt.managers.schedule_batch.flatten_arrays_to_pinned_cpu",
                        side_effect=capture_input_ids,
                    ),
                    unittest.mock.patch(
                        "sglang.srt.managers.schedule_batch.alloc_for_extend",
                        side_effect=stop_at_allocation,
                    ),
                    self.assertRaises(_StopBeforeAllocation),
                ):
                    batch.prepare_for_extend()

                self.assertEqual(captured["input_ids"], [expected_input_ids])
                if enabled:
                    cache.release_partial_prefix_source(req)

    def test_page_size_one_needs_no_partial_copy(self):
        cache = RadixCache.create_simulated(
            page_size=1, enable_partial_prefix_reuse=True
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4])),
                value=torch.tensor([11, 12, 13, 14]),
            )
        )
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 9])))
        )
        self.assertEqual(len(result.device_indices), 3)
        self.assertIsNone(result.partial_prefix_match)
        req = _make_partial_prefix_req(result.device_indices)
        self.assertFalse(cache.materialize_partial_prefix(req, result))

    def test_advanced_prefix_match_with_node_splits(self):
        """Advanced prefix matching: splits inside nodes and across pages."""
        for page_size in [1, 2]:
            with self.subTest(page_size=page_size):
                cache = RadixCache.create_simulated(page_size=page_size)

                # Insert a long sequence that will be split later.
                seq1 = [1, 2, 3, 4, 5, 6, 7, 8]
                val1 = torch.tensor([x * 10 for x in seq1], dtype=torch.int64)
                cache.insert(InsertParams(key=RadixKey(array("q", seq1)), value=val1))

                # Insert a diverging branch to create an internal node on the path.
                seq2 = [1, 2, 9, 10]
                val2 = torch.tensor([x * 10 for x in seq2], dtype=torch.int64)
                cache.insert(InsertParams(key=RadixKey(array("q", seq2)), value=val2))
                print(cache.pretty_print())

                baseline_total = cache.total_size()
                expected_total = 10  # 8 + 2
                self.assertEqual(baseline_total, expected_total)

                # Match that causes a split inside an existing node:
                # take first 4 tokens of seq1, then diverge.
                query1 = [1, 2, 3, 4, 999, 1000]
                result1 = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", query1)))
                )
                torch.testing.assert_close(result1.device_indices, val1[:4])
                # No data change after structural split during matching.
                self.assertEqual(cache.total_size(), baseline_total)

                # Full match of the long sequence still returns the full indices.
                result_full = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", seq1)))
                )
                torch.testing.assert_close(result_full.device_indices, val1)

                # Another split deeper on the path (after matching 6 tokens, then diverge).
                query2 = [1, 2, 3, 4, 5, 6, 777, 888]
                result2 = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", query2)))
                )
                torch.testing.assert_close(result2.device_indices, val1[:6])
                self.assertEqual(cache.total_size(), baseline_total)

                # Matching the short diverging branch should return exactly its indices.
                result_branch = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", seq2)))
                )
                torch.testing.assert_close(result_branch.device_indices, val2)

    def test_hash_value_storage(self):
        """Test that hash_value is stored correctly after insert operations."""
        cache = RadixCache.create_simulated(
            page_size=4,
            enable_kv_cache_events=True,
        )

        # Insert a sequence
        cache.insert(
            InsertParams(key=RadixKey(array("q", [1, 2, 3, 4, 5, 6, 7, 8])), value=None)
        )

        # Trigger event emission to compute hash_value lazily
        cache.take_events()

        # Find the inserted node (traverse from root)
        node = cache.root_node
        for i in range(0, 8, 4):  # page_size=4, so 2 pages
            child_key = tuple([1, 2, 3, 4][:4]) if i == 0 else tuple([5, 6, 7, 8][:4])
            if child_key in node.children:
                node = node.children[child_key]
                break

        # Verify hash_value is set (computed lazily during event emission)
        self.assertIsNotNone(node.hash_value)
        # Should have 2 pages (8 tokens / 4 page_size)
        self.assertEqual(len(node.hash_value), 2)

    def test_hash_value_repeating_tokens(self):
        """Test that repeating token patterns get different hash values."""
        cache = RadixCache.create_simulated(
            page_size=4,
            enable_kv_cache_events=True,
        )

        # Insert a sequence with repeating token pattern: [1,2,3,4, 1,2,3,4]
        cache.insert(
            InsertParams(key=RadixKey(array("q", [1, 2, 3, 4, 1, 2, 3, 4])), value=None)
        )

        events = cache.take_events()
        block_stored_events = [e for e in events if isinstance(e, BlockStored)]

        # The two pages should be represented by one parent-linked store event.
        self.assertEqual(len(block_stored_events), 1)
        self.assertEqual(len(block_stored_events[0].block_hashes), 2)

        # Extract block hashes
        block_hash_1, block_hash_2 = block_stored_events[0].block_hashes

        # The two blocks should have DIFFERENT hashes despite same content
        # because they are at different positions (sequence-aware hashing)
        self.assertNotEqual(
            block_hash_1,
            block_hash_2,
            "Repeating token patterns should get different sequence-aware hashes",
        )

        # The coalesced event keeps the original root parent and ordered hashes.
        self.assertIsNone(block_stored_events[0].parent_block_hash)

    def test_hash_value_split(self):
        """Test that hash_value is split correctly when nodes are split."""
        cache = RadixCache.create_simulated(
            page_size=2,
            enable_kv_cache_events=True,
        )

        # Insert a sequence that will cause a split
        cache.insert(InsertParams(key=RadixKey(array("q", [1, 2, 3, 4])), value=None))
        cache.take_events()  # Clear events and compute hash_value for first node

        # Insert a diverging sequence that will cause a split at page boundary
        cache.insert(InsertParams(key=RadixKey(array("q", [1, 2, 5, 6])), value=None))
        cache.take_events()  # Trigger event emission to compute hash_value

        # Find the split node
        node = cache.root_node
        child_key = tuple([1, 2])
        if child_key in node.children:
            node = node.children[child_key]
            # After split and event emission, hash_value should be computed
            # Note: If hash_value wasn't set before split, it will be computed lazily
            # during event emission. If it was set, it will be split.
            # Either way, after events are emitted, it should be set.
            self.assertIsNotNone(node.hash_value)
            # Should have 1 page (split at page_size=2)
            self.assertEqual(len(node.hash_value), 1)

    def test_memory_allocated(self):
        keys, values = [], []

        num_seqs = 10000
        vocab_size = 1000
        base_prefix_len = 10000
        suffix_len = 100

        torch_allocated_before = torch.get_device_module().memory_allocated()

        # build dataset with common prefix
        common_prefix = [
            random.randint(1, vocab_size - 1) for _ in range(base_prefix_len)
        ]
        for _ in range(num_seqs):
            suffix = [random.randint(1, vocab_size - 1) for _ in range(suffix_len)]
            seq = common_prefix + suffix
            keys.append(seq)
            values.append(torch.zeros(len(seq), device=get_device(), dtype=torch.int32))

        cache: RadixCache = RadixCache.create_simulated()

        for key, value in zip(keys, values):
            cache.insert(InsertParams(key=RadixKey(array("q", key)), value=value))

        del values

        torch_allocated = (
            torch.get_device_module().memory_allocated() - torch_allocated_before
        )
        cache_size_bytes = cache.total_size() * 4
        print(f"\nCache size (MB): {cache_size_bytes / (1024 * 1024)}")
        print(f"Torch allocated (MB): {torch_allocated / (1024 * 1024)}")

        # The cache size should be within reasonable bounds of the actual allocated memory.
        self.assertLess(torch_allocated, cache_size_bytes * 2)


if __name__ == "__main__":
    unittest.main()
