"""A fixed page keeps its band's extent over it, free or not.

A band held where it stands fixes its live pages; freeing some of them leaves
fixed pages on the free list. No path that shrinks a band past free pages may
pass one of those: a lazy end's compaction (urgent or not), a float's boundary
absorption, or a float parking once it holds no live page. Capacity prices a
band's extent over its fixed pages, so a band that shrank past one would lose
pages.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_test_unified_pools,
    hold_band,
    release_band,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _tri():
    """state end + window float + full end, every end compacting lazily."""
    setup_allocator_context()
    allocator = build_test_unified_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=1,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=1,
        start_layer=0,
        end_layer=2,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        full_max_total_num_tokens=48,
        swa_max_total_num_tokens=48,
        enable_memory_saver=False,
        need_sort=False,
        mamba_layer_ids=[0],
        mamba2_cache_params=SimpleNamespace(
            shape=SimpleNamespace(conv=[(3, 8)], temporal=(1, 4, 8)),
            dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
            layers=[0],
        ),
        max_mamba_cache_size=8,
        model_context_len=256,
        extra_max_context_len=1,
        max_num_reqs=4,
        enable_mamba_extra_buffer=False,
        enable_mamba_extra_buffer_lazy=False,
        disable_overlap_schedule=True,
        sliding_window_size=32,
    ).token_to_kv_pool_allocator
    (window,) = allocator._windows
    assert window.grow_direction == "float"
    return allocator, window


def _holes(band):
    return set(band._free_pages_snapshot())


def _virtual_ids(band, pages):
    p2v = band.ids.physical_to_virtual
    return torch.tensor([int(p2v[p]) for p in pages], dtype=torch.int64)


class TestEndCompactionKeepsFixedFreePages(CustomTestCase):
    def _held_end_with_a_free_fixed_frontier(self):
        """The full end (grows down) holds four groups; it is held in place,
        then an interior group and the group at its frontier are freed."""
        allocator, _ = _tri()
        full = allocator.full_attn_allocator
        groups = [allocator.alloc(4) for _ in range(4)]
        self.assertTrue(all(g is not None for g in groups))
        hold_band(full)
        fixed = full.fixed_pages()
        allocator.free(groups[1])
        allocator.free(groups[3])
        frontier = full._span_range()[0]
        self.assertIn(frontier, _holes(full))
        self.assertIn(frontier, fixed)
        return allocator, full, fixed

    def _check_kept(self, allocator, full, fixed):
        first, end = full._span_range()
        self.assertLessEqual(first, min(fixed))
        self.assertGreaterEqual(end, max(fixed) + 1)
        self.assertEqual(len(_holes(full)), 8)
        self.assertEqual(allocator.verify_byte_accounting(), [])
        # Every hole is still there to take.
        self.assertIsNotNone(allocator.alloc(8))
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_an_urgent_compaction(self):
        allocator, full, fixed = self._held_end_with_a_free_fixed_frontier()
        full.flush_for_allocation()
        self._check_kept(allocator, full, fixed)

    def test_an_opportunistic_compaction(self):
        allocator, full, fixed = self._held_end_with_a_free_fixed_frontier()
        full.flush_opportunistic()
        self._check_kept(allocator, full, fixed)


class TestFloatKeepsFixedFreePages(CustomTestCase):
    def _held_float(self):
        allocator, window = _tri()
        for _ in range(3):
            self.assertIsNotNone(allocator.alloc(4))
        hold_band(window)
        return allocator, window

    def test_boundary_absorption(self):
        allocator, window = self._held_float()
        low, high = window._span_range()
        # Free the pages at both boundaries: fixed, now holes.
        allocator.free_swa(_virtual_ids(window, [low, high - 1]))
        self.assertEqual(_holes(window), {low, high - 1})
        window.flush_opportunistic()
        self.assertEqual(window._span_range(), (low, high))
        self.assertEqual(_holes(window), {low, high - 1})
        self.assertEqual(allocator.verify_byte_accounting(), [])
        # Released, the boundary holes go back to the gap.
        release_band(window)
        window.flush_opportunistic()
        self.assertEqual(window._span_range(), (low + 1, high - 1))
        self.assertEqual(_holes(window), set())
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_an_empty_float_with_a_fixed_page_does_not_park(self):
        allocator, window = self._held_float()
        low, high = window._span_range()
        allocator.free_swa(_virtual_ids(window, range(low, high)))
        self.assertEqual(window._live_pages(), 0)
        # Still opaque over its fixed pages, to the band and to the planner.
        self.assertEqual(window._span_range(), (low, high))
        self.assertFalse(window._is_frontier_transparent())
        self.assertFalse(
            allocator.pricing_snapshot().band(window.sub_pool_name).transparent
        )
        window.flush_opportunistic()
        self.assertEqual(window._span_range(), (low, high))
        self.assertEqual(allocator.verify_byte_accounting(), [])
        # Released, it parks.
        release_band(window)
        self.assertTrue(window._is_frontier_transparent())
        self.assertEqual(window._span_pages(), 0)
        self.assertEqual(allocator.verify_byte_accounting(), [])


if __name__ == "__main__":
    unittest.main()
