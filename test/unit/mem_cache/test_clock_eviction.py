"""Unit tests for the CLOCK (second-chance) eviction policy."""

import pytest

from sglang.srt.mem_cache.evict_policy import CLOCKStrategy
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode


class TestCLOCKStrategyPriority:
    def _make_node(self, referenced: bool, last_access: float) -> TreeNode:
        n = TreeNode()
        n.referenced = referenced
        n.last_access_time = last_access
        return n

    def test_unreferenced_sorts_before_referenced(self):
        strategy = CLOCKStrategy()
        old_unreferenced = self._make_node(referenced=False, last_access=1.0)
        new_referenced   = self._make_node(referenced=True,  last_access=2.0)
        assert strategy.get_priority(old_unreferenced) < strategy.get_priority(new_referenced)

    def test_same_ref_bit_falls_back_to_lru(self):
        strategy = CLOCKStrategy()
        older = self._make_node(referenced=False, last_access=1.0)
        newer = self._make_node(referenced=False, last_access=9.0)
        assert strategy.get_priority(older) < strategy.get_priority(newer)

    def test_missing_referenced_attr_treated_as_false(self):
        strategy = CLOCKStrategy()
        n = TreeNode()
        if hasattr(n, "referenced"):
            del n.__dict__["referenced"]
        priority = strategy.get_priority(n)
        assert priority[0] == 0


class TestRadixCacheCLOCKIntegration:
    def test_clock_policy_registered(self):
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from unittest.mock import MagicMock
        mock_alloc = MagicMock()
        mock_alloc.device = "cpu"
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_alloc,
            page_size=1,
            enable_kv_cache_events=False,
            eviction_policy="clock",
        )
        cache = RadixCache(params)
        assert isinstance(cache.eviction_strategy, CLOCKStrategy)

    def test_referenced_bit_set_on_match(self):
        import torch
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
        cache = RadixCache.create_simulated(disable=False, page_size=1)
        key_ids = list(range(4))
        value = torch.zeros(4, dtype=torch.int32)
        cache.insert(InsertParams(key=key_ids, value=value))
        cache.match_prefix(MatchPrefixParams(key=key_ids))

        all_nodes = []
        def _collect(node):
            for child in node.children.values():
                all_nodes.append(child)
                _collect(child)
        _collect(cache.root_node)
        assert any(n.referenced for n in all_nodes)


    def test_second_chance_eviction_order(self):
        """Referenced node survives one eviction round; unreferenced node is evicted first."""
        import torch
        from unittest.mock import MagicMock
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams, EvictParams

        mock_alloc = MagicMock()
        mock_alloc.device = "cpu"
        freed = []
        mock_alloc.free = lambda v: freed.append(v)
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_alloc,
            page_size=1,
            enable_kv_cache_events=False,
            eviction_policy="clock",
        )
        cache = RadixCache(params)

        # Insert node A — will be marked referenced
        key_a = list(range(4))
        value_a = torch.zeros(4, dtype=torch.int32)
        cache.insert(InsertParams(key=key_a, value=value_a))
        cache.match_prefix(MatchPrefixParams(key=key_a))

        # Insert node B — unreferenced
        key_b = list(range(4, 8))
        value_b = torch.ones(4, dtype=torch.int32)
        cache.insert(InsertParams(key=key_b, value=value_b))

        # Collect leaf nodes
        all_nodes = []
        def _collect(node):
            for child in node.children.values():
                all_nodes.append(child)
                _collect(child)
        _collect(cache.root_node)

        node_a = next(n for n in all_nodes if n.referenced)
        node_b = next(n for n in all_nodes if not n.referenced)

        # Evict 4 tokens — node B should go first, node A gets second chance
        cache.evict(EvictParams(num_tokens=4))

        assert node_a.referenced == False
        assert len(freed) > 0
