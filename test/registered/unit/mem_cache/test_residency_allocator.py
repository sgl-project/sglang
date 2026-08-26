import unittest
from types import SimpleNamespace

import torch

from sglang.srt.attn_parallel import KvResidency
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator.residency import (
    ResidencyAwarePagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestResidencyAwareAllocator(CustomTestCase):
    def _allocator(self):
        return ResidencyAwarePagedTokenToKVPoolAllocator(
            physical_size=128,
            physical_page_size=1,
            dcp_size=4,
            replicated_fraction=0.5,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=object(),
            need_sort=False,
        )

    def test_regions_map_to_disjoint_physical_rows(self):
        allocator = self._allocator()
        initial_available = allocator.total_available_size()

        allocator.set_active_residency(KvResidency.REPLICATED)
        replicated = allocator.alloc(allocator.page_size)
        allocator.set_active_residency(KvResidency.STRIPED)
        striped = allocator.alloc(allocator.page_size)

        self.assertTrue((replicated < allocator.striped_virtual_start).all())
        self.assertTrue((striped >= allocator.striped_virtual_start).all())
        replicated_physical = set(replicated.tolist())
        striped_physical = set((striped // allocator.dcp_size).tolist())
        self.assertTrue(replicated_physical.isdisjoint(striped_physical))

        allocator.free(torch.cat((replicated, striped)))
        self.assertEqual(allocator.total_available_size(), initial_available)

    def test_last_replicated_page_does_not_alias_first_striped_page(self):
        allocator = self._allocator()
        allocator.set_active_residency(KvResidency.REPLICATED)
        replicated_pages = []
        while allocator.available_size():
            replicated_pages.append(allocator.alloc(allocator.page_size))
        last_replicated = replicated_pages[-1]

        allocator.set_active_residency(KvResidency.STRIPED)
        first_striped = allocator.alloc(allocator.page_size)

        replicated_physical = set(last_replicated.tolist())
        striped_physical = set((first_striped // allocator.dcp_size).tolist())
        self.assertTrue(replicated_physical.isdisjoint(striped_physical))

    def test_free_segments_deduplicates_shared_boundary_page(self):
        allocator = self._allocator()
        calls = []
        allocator.free_segment = lambda indices, *, start_pos: calls.append(
            (indices.clone(), start_pos)
        )

        allocator.free_segments(
            [
                (torch.tensor([4, 5, 6]), 0),
                (torch.tensor([7, 8, 9]), 3),
            ]
        )

        self.assertEqual([len(indices) for indices, _ in calls], [3, 2])
        self.assertEqual([start for _, start in calls], [0, 4])

    def test_transitioning_residency_cannot_allocate(self):
        allocator = self._allocator()
        with self.assertRaises(RuntimeError):
            allocator.set_active_residency(KvResidency.TRANSITIONING)

    def test_pool_stats_use_active_residency_capacity(self):
        allocator = self._allocator()
        observer = SchedulerPoolStatsObserver(
            tree_cache=SimpleNamespace(evictable_size=lambda: 0),
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=SimpleNamespace(),
            session_controller=SimpleNamespace(sessions={}),
            hisparse_coordinator=None,
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            enable_hisparse=False,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=allocator.physical_size,
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )

        stats = observer._get_token_info()
        self.assertEqual(stats.full_num_used, 0)
        self.assertEqual(stats.full_token_usage, 0.0)

        allocator.set_active_residency(KvResidency.STRIPED)
        allocator.alloc(allocator.page_size)
        stats = observer._get_token_info()
        self.assertEqual(stats.full_num_used, allocator.page_size)
        self.assertGreater(stats.full_token_usage, 0.0)


class TestResidencyLayoutTag(CustomTestCase):
    def test_scheduler_stamps_sticky_layout_namespace(self):
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(
            dynamic_attn_parallel_enable_dcp=True,
            dynamic_attn_parallel_dcp_min_context=8,
            dynamic_attn_parallel_striped_min_context=8,
        )
        scheduler.ps = SimpleNamespace(attn_dcp_size=4)
        req = SimpleNamespace(
            kv_residency=None,
            kv_layout_tagged=False,
            full_untruncated_fill_ids=list(range(16)),
            origin_input_ids=list(range(16)),
            extra_key="tenant",
        )

        scheduler._ensure_req_kv_residency(req)
        first_key = req.extra_key
        self.assertEqual(req.kv_residency, KvResidency.STRIPED)
        self.assertIn("striped:dcp4:epoch0", first_key)

        req.full_untruncated_fill_ids = [1]
        scheduler._ensure_req_kv_residency(req)
        self.assertEqual(req.kv_residency, KvResidency.STRIPED)
        self.assertEqual(req.extra_key, first_key)

    def test_scheduler_defaults_dynamic_dcp_to_replicated_residency(self):
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(
            dynamic_attn_parallel_enable_dcp=True,
            dynamic_attn_parallel_dcp_min_context=8,
            dynamic_attn_parallel_striped_min_context=None,
        )
        scheduler.ps = SimpleNamespace(attn_dcp_size=4)
        req = SimpleNamespace(
            kv_residency=None,
            kv_layout_tagged=False,
            full_untruncated_fill_ids=list(range(16)),
            origin_input_ids=list(range(16)),
            extra_key=None,
        )

        scheduler._ensure_req_kv_residency(req)

        self.assertEqual(req.kv_residency, KvResidency.REPLICATED)
        self.assertIn("replicated:dcp4:epoch0", req.extra_key)

    def test_radix_keys_reject_cross_layout_match(self):
        tokens = torch.arange(8).tolist()
        from array import array

        replicated = RadixKey(
            array("q", tokens), extra_key="tenant|sglang-kv:replicated:dcp4:epoch0"
        )
        striped = RadixKey(
            array("q", tokens), extra_key="tenant|sglang-kv:striped:dcp4:epoch0"
        )
        with self.assertRaises(ValueError):
            replicated.match(striped)


if __name__ == "__main__":
    unittest.main()
