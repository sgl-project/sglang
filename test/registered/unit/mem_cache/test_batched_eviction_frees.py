"""Unit tests for batched eviction frees.

Asserts that freeing the concatenation of several disjoint page-aligned
segments leaves the SWA/full allocators and the full->swa mapping in a state
equivalent to freeing the segments one at a time, and that the unified-cache
component handlers batch multi-tensor FreeComponentDeviceSlot actions into a
single allocator call.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.unified_cache.cache_action import FreeComponentDeviceSlot
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
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
    """Disjoint page-aligned full-index segments, like per-node tree values."""
    g = torch.Generator().manual_seed(seed)
    pages = torch.randperm(FULL_PAGES, generator=g)[: n_segments * 2] + 1
    segs = []
    for i in range(n_segments):
        page_pair = pages[2 * i : 2 * i + 2]
        idx = (page_pair[:, None] * PAGE + torch.arange(PAGE)[None, :]).reshape(-1)
        segs.append(idx.to(torch.int64))
    return segs


def _install_mapping(alloc: SWATokenToKVPoolAllocator, segs: list[torch.Tensor]):
    swa_slot = PAGE
    for seg in segs:
        n = seg.numel()
        swa_idx = torch.arange(swa_slot, swa_slot + n, dtype=torch.int64)
        alloc.set_full_to_swa_mapping(seg, swa_idx)
        swa_slot += n


def _allocator_state(alloc: SWATokenToKVPoolAllocator):
    def pages_set(a):
        parts = []
        if a.free_pages is not None:
            parts.append(a.free_pages.to(torch.int64))
        if a.release_pages is not None:
            parts.append(a.release_pages.to(torch.int64))
        return set(torch.cat(parts).tolist()) if parts else set()

    return (
        pages_set(alloc.full_attn_allocator),
        pages_set(alloc.swa_attn_allocator),
        alloc.full_to_swa_index_mapping.clone(),
    )


class TestBatchedEvictionFrees(CustomTestCase):
    def test_free_swa_cat_equivalent_to_sequential(self):
        segs = _segments(seed=7, n_segments=5)

        a = _make_allocator()
        _install_mapping(a, segs)
        for seg in segs:
            a.free_swa(seg)

        b = _make_allocator()
        _install_mapping(b, segs)
        b.free_swa(torch.cat(segs))

        full_a, swa_a, map_a = _allocator_state(a)
        full_b, swa_b, map_b = _allocator_state(b)
        self.assertEqual(swa_a, swa_b)
        self.assertTrue(torch.equal(map_a, map_b))
        self.assertEqual(full_a, full_b)

    def test_full_free_cat_equivalent_to_sequential(self):
        segs = _segments(seed=13, n_segments=5)

        a = _make_allocator()
        for seg in segs:
            a.full_attn_allocator.free(seg)

        b = _make_allocator()
        b.full_attn_allocator.free(torch.cat(segs))

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
            FreeComponentDeviceSlot(indices=list(segs), component_type=ComponentType.SWA)
        )
        alloc = comp.cache.token_to_kv_pool_allocator
        self.assertEqual(alloc.free_swa.call_count, 1)
        (freed,), _ = alloc.free_swa.call_args
        self.assertTrue(torch.equal(freed, torch.cat(segs)))

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
        self.assertEqual(alloc.full_attn_allocator.free.call_count, 1)
        (freed,), _ = alloc.full_attn_allocator.free.call_args
        self.assertTrue(torch.equal(freed, torch.cat(segs)))

    def test_single_tensor_action_not_concatenated(self):
        comp = object.__new__(SWAComponent)
        comp.cache = self._component_cache()
        (seg,) = _segments(seed=11, n_segments=1)
        comp.apply_component_action(
            FreeComponentDeviceSlot(indices=[seg], component_type=ComponentType.SWA)
        )
        alloc = comp.cache.token_to_kv_pool_allocator
        self.assertEqual(alloc.free_swa.call_count, 1)
        (freed,), _ = alloc.free_swa.call_args
        self.assertIs(freed, seg)


if __name__ == "__main__":
    unittest.main()
