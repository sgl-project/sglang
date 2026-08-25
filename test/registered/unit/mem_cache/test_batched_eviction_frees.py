"""Unit tests for batched eviction frees.

Freeing the concatenation of several disjoint page-exact segments must leave
the SWA/full allocators and the full->swa mapping in the same state as freeing
the segments one at a time; the unified-cache component handlers must batch a
multi-tensor FreeComponentDeviceSlot into a single allocator call; and
UnifiedRadixCache must defer device frees only on tracker-driven eviction
walks, freeing immediately when a capacity target is being checked.

    python -m pytest test/registered/unit/mem_cache/test_batched_eviction_frees.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.unified_cache.cache_action import FreeComponentDeviceSlot
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 128
FULL_PAGES = 64
SWA_PAGES = 32


def _make_allocator() -> SWATokenToKVPoolAllocator:
    kvcache = MagicMock(spec=BaseSWAKVPool)
    kvcache.full_kv_pool = None
    kvcache.swa_kv_pool = None
    kvcache.swa_req_ring_size = None
    return SWATokenToKVPoolAllocator(
        size=FULL_PAGES * PAGE,
        size_swa=SWA_PAGES * PAGE,
        page_size=PAGE,
        dtype=torch.float16,
        device="cpu",
        kvcache=kvcache,
        need_sort=False,
    )


def _segments(seed: int, n_segments: int) -> list[torch.Tensor]:
    """Disjoint page-exact full-index segments, like per-node tree values."""
    g = torch.Generator().manual_seed(seed)
    pages = torch.randperm(FULL_PAGES, generator=g)[: n_segments * 2] + 1
    segs = []
    for i in range(n_segments):
        page_pair = pages[2 * i : 2 * i + 2]
        idx = (page_pair[:, None] * PAGE + torch.arange(PAGE)[None, :]).reshape(-1)
        segs.append(idx.to(torch.int64))
    return segs


def _allocate_segments(
    alloc: SWATokenToKVPoolAllocator, seed: int, n_segments: int, pages: int = 2
) -> list[torch.Tensor]:
    """Allocate node-sized page-exact segments with mapped SWA peers, returned
    in shuffled order like unrelated tree nodes met by one eviction walk."""
    n = n_segments * pages * PAGE
    full_idx = alloc.full_attn_allocator.alloc(n)
    swa_idx = alloc.swa_attn_allocator.alloc(n)
    alloc.set_full_to_swa_mapping(full_idx, swa_idx)
    segs = list(full_idx.to(torch.int64).split(pages * PAGE))
    g = torch.Generator().manual_seed(seed)
    return [segs[i] for i in torch.randperm(n_segments, generator=g).tolist()]


def _allocator_state(alloc: SWATokenToKVPoolAllocator):
    return (
        set(alloc.full_attn_allocator.get_all_free_pages().tolist()),
        set(alloc.swa_attn_allocator.get_all_free_pages().tolist()),
        alloc.full_to_swa_index_mapping.clone(),
    )


class TestBatchedEvictionFrees(CustomTestCase):
    def test_free_swa_segment_cat_equivalent_to_sequential(self):
        a = _make_allocator()
        for seg in _allocate_segments(a, seed=7, n_segments=5):
            a.free_swa_segment(seg, start_pos=0)

        b = _make_allocator()
        b.free_swa_segment(
            torch.cat(_allocate_segments(b, seed=7, n_segments=5)), start_pos=0
        )

        full_a, swa_a, map_a = _allocator_state(a)
        full_b, swa_b, map_b = _allocator_state(b)
        self.assertEqual(swa_a, swa_b)
        self.assertTrue(torch.equal(map_a, map_b))
        self.assertEqual(full_a, full_b)

    def test_full_free_segment_cat_equivalent_to_sequential(self):
        a = _make_allocator()
        for seg in _allocate_segments(a, seed=13, n_segments=5):
            a.full_attn_allocator.free_segment(seg, start_pos=0)

        b = _make_allocator()
        b.full_attn_allocator.free_segment(
            torch.cat(_allocate_segments(b, seed=13, n_segments=5)), start_pos=0
        )

        full_a, _, _ = _allocator_state(a)
        full_b, _, _ = _allocator_state(b)
        self.assertEqual(full_a, full_b)

    def _component_cache(self):
        alloc = MagicMock()
        return SimpleNamespace(token_to_kv_pool_allocator=alloc, is_swa_enabled=True)

    def test_swa_component_batches_multi_tensor_action(self):
        comp = object.__new__(SWAComponent)
        comp.cache = self._component_cache()
        segs = _segments(seed=3, n_segments=4)
        comp.apply_component_action(
            FreeComponentDeviceSlot(
                indices=list(segs), component_type=ComponentType.SWA
            )
        )
        alloc = comp.cache.token_to_kv_pool_allocator
        self.assertEqual(alloc.free_swa_segment.call_count, 1)
        (freed,), kwargs = alloc.free_swa_segment.call_args
        self.assertTrue(torch.equal(freed, torch.cat(segs)))
        self.assertEqual(kwargs, {"start_pos": 0})

    def test_full_component_batches_multi_tensor_action(self):
        comp = object.__new__(FullComponent)
        comp.cache = self._component_cache()
        segs = _segments(seed=5, n_segments=3)
        comp.apply_component_action(
            FreeComponentDeviceSlot(
                indices=list(segs), component_type=ComponentType.FULL
            )
        )
        alloc = comp.cache.token_to_kv_pool_allocator
        self.assertEqual(alloc.full_attn_allocator.free_segment.call_count, 1)
        (freed,), kwargs = alloc.full_attn_allocator.free_segment.call_args
        self.assertTrue(torch.equal(freed, torch.cat(segs)))
        self.assertEqual(kwargs, {"start_pos": 0})

    def test_single_tensor_action_not_concatenated(self):
        comp = object.__new__(SWAComponent)
        comp.cache = self._component_cache()
        (seg,) = _segments(seed=11, n_segments=1)
        comp.apply_component_action(
            FreeComponentDeviceSlot(indices=[seg], component_type=ComponentType.SWA)
        )
        alloc = comp.cache.token_to_kv_pool_allocator
        self.assertEqual(alloc.free_swa_segment.call_count, 1)
        (freed,), _ = alloc.free_swa_segment.call_args
        self.assertIs(freed, seg)


class TestEvictionWalkDeferral(CustomTestCase):
    """Drive UnifiedRadixCache._evict_components with a stubbed walk that frees
    one segment per node, and observe when the device drain runs."""

    REQUEST = {ComponentType.FULL: 4 * PAGE}
    TARGETS = {ComponentType.FULL: (ComponentType.FULL, 8 * PAGE)}

    def _cache(self):
        cache = object.__new__(UnifiedRadixCache)
        cache._pending_frees = None
        cache.drains: list[list[torch.Tensor]] = []
        cache.pending_during_walk: list[bool] = []

        def drain_device(device_frees):
            cache.drains.append(list(device_frees.pop(ComponentType.FULL, [])))
            device_frees.clear()

        cache._drain_device_frees = drain_device
        cache._drain_host_frees = MagicMock()
        return cache

    @staticmethod
    def _walk(cache, segs: list[torch.Tensor]):
        def walk(request_by_type, tracker, available_size_targets):
            for seg in segs:
                cache.pending_during_walk.append(cache._pending_frees is not None)
                cache._free_values({ComponentType.FULL: [seg]}, {})

        return walk

    def test_tracker_walk_defers_into_one_drain(self):
        segs = _segments(seed=17, n_segments=3)
        cache = self._cache()
        cache._evict_components_inner = self._walk(cache, segs)
        cache._evict_components(self.REQUEST, {ComponentType.FULL: 0})
        self.assertEqual(cache.pending_during_walk, [True, True, True])
        self.assertEqual(len(cache.drains), 1)
        self.assertEqual([id(t) for t in cache.drains[0]], [id(t) for t in segs])
        self.assertIsNone(cache._pending_frees)

    def test_targeted_walk_frees_per_node(self):
        segs = _segments(seed=19, n_segments=3)
        cache = self._cache()
        cache._evict_components_inner = self._walk(cache, segs)
        cache._evict_components(
            self.REQUEST, {ComponentType.FULL: 0}, available_size_targets=self.TARGETS
        )
        self.assertEqual(cache.pending_during_walk, [False, False, False])
        self.assertEqual(
            [[id(t) for t in d] for d in cache.drains], [[id(t)] for t in segs]
        )
        self.assertIsNone(cache._pending_frees)

    def test_targeted_walk_settles_enclosing_deferred_frees(self):
        outer_segs = _segments(seed=23, n_segments=3)
        inner_segs = _segments(seed=29, n_segments=2)
        cache = self._cache()
        outer_walk = self._walk(cache, outer_segs[:2])
        inner_walk = self._walk(cache, inner_segs)

        def nested_walk(request_by_type, tracker, available_size_targets):
            outer_walk(request_by_type, tracker, available_size_targets)
            cache._evict_components_inner = inner_walk
            cache._evict_components(
                request_by_type, tracker, available_size_targets=self.TARGETS
            )
            cache._evict_components_inner = nested_walk
            # Back in the outer walk: deferral resumes.
            cache.pending_during_walk.append(cache._pending_frees is not None)
            cache._free_values({ComponentType.FULL: [outer_segs[2]]}, {})

        cache._evict_components_inner = nested_walk
        cache._evict_components(self.REQUEST, {ComponentType.FULL: 0})
        # The outer walk's two deferred frees settle before the targeted walk
        # starts, the targeted walk frees per node, and the outer walk's
        # trailing free is deferred to the final drain.
        self.assertEqual(
            [[id(t) for t in d] for d in cache.drains],
            [
                [id(outer_segs[0]), id(outer_segs[1])],
                [id(inner_segs[0])],
                [id(inner_segs[1])],
                [id(outer_segs[2])],
            ],
        )
        self.assertEqual(cache.pending_during_walk, [True, True, False, False, True])
        self.assertIsNone(cache._pending_frees)


if __name__ == "__main__":
    unittest.main()
