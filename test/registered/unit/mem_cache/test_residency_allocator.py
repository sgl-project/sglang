import unittest
from types import SimpleNamespace

import torch

from sglang.srt.attn_parallel import KvResidency
from sglang.srt.managers.scheduler import Scheduler
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

    def test_transitioning_residency_cannot_allocate(self):
        allocator = self._allocator()
        with self.assertRaises(RuntimeError):
            allocator.set_active_residency(KvResidency.TRANSITIONING)


class TestResidencyLayoutTag(CustomTestCase):
    def test_scheduler_stamps_sticky_layout_namespace(self):
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(
            dynamic_attn_parallel_enable_dcp=True,
            dynamic_attn_parallel_dcp_min_context=8,
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
