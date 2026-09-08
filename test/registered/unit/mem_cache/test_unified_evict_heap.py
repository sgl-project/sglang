"""Eviction-order equivalence of the persistent lazy leaf heaps in FullComponent."""

import heapq
import os
import random
import sys
import types
import unittest
from typing import Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full_component import EvictLayer
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_unified_radix_cache_unittest import CacheConfig, build_fixture  # noqa: E402


# Reference: the per-walk rebuild these heaps replace.
def _ref_evict_device_start(self, request_cnt: int) -> None:
    self._ensure_eviction_strategy()
    self._evict_device_request_cnt = request_cnt
    self._ref_last_node = None
    self._ref_heap = [
        (self.session_ref_eviction_strategy(n), n)
        for n in self.tree_core.evictable_device_leaves
    ]
    heapq.heapify(self._ref_heap)


def _ref_evict_device_next_node(
    self, tracker, device_frees, host_frees
) -> Optional[int]:
    ct = self.component_type
    lv = self._ref_last_node
    if (
        lv is not None
        and lv.parent is not None
        and lv.parent in self.tree_core.evictable_device_leaves
    ):
        heapq.heappush(
            self._ref_heap, (self.session_ref_eviction_strategy(lv.parent), lv.parent)
        )
    self._ref_last_node = None
    while tracker[ct] < self._evict_device_request_cnt and self._ref_heap:
        _, x = heapq.heappop(self._ref_heap)
        if x not in self.tree_core.evictable_device_leaves:
            continue
        self._ref_last_node = x
        return x.id
    return None


def _ref_evict_device_end(self) -> None:
    self._ref_heap = []
    self._ref_last_node = None


def _full_component(cache):
    for comp in cache.tree_core.components:
        if comp.component_type == ComponentType.FULL:
            return comp
    raise AssertionError("no FULL component")


def _make(reference: bool):
    cache, allocator, _ = build_fixture(CacheConfig(kv_size=4096, max_context_len=8192))
    if reference:
        comp = _full_component(cache)
        comp._evict_device_start = types.MethodType(_ref_evict_device_start, comp)
        comp._evict_device_next_node = types.MethodType(
            _ref_evict_device_next_node, comp
        )
        comp._evict_device_end = types.MethodType(_ref_evict_device_end, comp)
    return cache, allocator


def _run_trace(cache, allocator, seed: int, steps: int = 3000):
    order = []
    core = cache.tree_core
    orig = core.evict_device_leaf

    def spy(node_id, *args, **kwargs):
        order.append(tuple(core.node_by_id(node_id).key.token_ids))
        return orig(node_id, *args, **kwargs)

    core.evict_device_leaf = spy
    rng = random.Random(seed)
    tok = 0
    locked = []
    for _ in range(steps):
        op = rng.random()
        if op < 0.55:
            length = rng.randint(1, 12)
            key = RadixKey(
                token_ids=[rng.randint(0, 30) for _ in range(length)],
                extra_key=str(rng.randint(0, 40)),
            )
            value = torch.arange(
                tok, tok + length, dtype=torch.int64, device=allocator.device
            )
            tok += length
            cache.insert(InsertParams(key=key, value=value))
        elif op < 0.68:
            length = rng.randint(1, 12)
            key = RadixKey(
                token_ids=[rng.randint(0, 30) for _ in range(length)],
                extra_key=str(rng.randint(0, 40)),
            )
            cache.match_prefix(MatchPrefixParams(key=key))
        elif op < 0.76 and core.evictable_device_leaves:
            node = rng.choice(sorted(core.evictable_device_leaves, key=lambda n: n.id))
            cache.inc_lock_ref(node.id)
            locked.append(node.id)
        elif op < 0.84 and locked:
            cache.dec_lock_ref(locked.pop(rng.randrange(len(locked))))
        else:
            cache.evict(EvictParams(num_tokens=rng.randint(1, 40)))
    cache.evict(EvictParams(num_tokens=1 << 20))
    return order


class TestUnifiedEvictHeap(unittest.TestCase):
    def test_eviction_order_matches_reference(self):
        for seed in (1, 2, 3):
            with self.subTest(seed=seed):
                ref = _run_trace(*_make(reference=True), seed)
                got = _run_trace(*_make(reference=False), seed)
                self.assertGreater(len(ref), 0)
                self.assertEqual(ref, got)

    def test_heap_bounded_and_drained(self):
        cache, allocator = _make(reference=False)
        comp = _full_component(cache)
        core = cache.tree_core
        rng = random.Random(0)
        for i in range(3000):
            key = RadixKey(
                token_ids=[rng.randint(0, 50) for _ in range(8)], extra_key=str(i)
            )
            cache.insert(
                InsertParams(
                    key=key,
                    value=torch.arange(8, dtype=torch.int64, device=allocator.device),
                )
            )
            if i % 3 == 0:
                cache.evict(EvictParams(num_tokens=8))
            self.assertLessEqual(
                len(comp._device_leaf_heap),
                max(1024, 4 * len(core.evictable_device_leaves)),
            )
        cache.evict(EvictParams(num_tokens=1 << 30))
        self.assertEqual(len(core.evictable_device_leaves), 0)
        comp._refresh_leaf_heap(EvictLayer.DEVICE)
        self.assertEqual(len(comp._device_leaf_heap), 0)


if __name__ == "__main__":
    unittest.main()
