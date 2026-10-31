"""A band tracks the pages a host copy touches by the ids the copy reads.

A host copy reads its controller ids through the transfer engine's rule
(collapsed by the host pool's DCP width, then paged as the storage's
``host_ids`` says), while the band tracks transfers in its local virtual ids
over physical-row pages. These tests pin that both name the same physical
pages, that ids a direct I/O backend hands on the host translate on the band's
device, that retirement keeps the event map bounded, and that a reservation's
taken pages come from the pages it took, not from the whole span.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.memory.virtual_slot import translate_token_ids
from sglang.srt.mem_cache.kv.alloc.band import Band, _span_growth
from sglang.srt.mem_cache.kv.alloc.chain import ChainRoles
from sglang.srt.mem_cache.kv.alloc.id_space import UnitIds
from sglang.srt.mem_cache.kv.alloc.page_usage import PageUsage
from sglang.srt.mem_cache.kv.storage import KVStorage
from sglang.srt.mem_cache.l2_transfer import L2Transfer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _band_stub(*, page_size, pool_page_size, device="cpu"):
    ids = UnitIds(
        num_virtual_ids=16,
        num_physical_units=16,
        unit_tokens=pool_page_size,
        shards_under_dcp=False,
        device=device,
    )
    return SimpleNamespace(
        ids=ids,
        page_size=page_size,
        pool_page_size=pool_page_size,
        shards_under_dcp=False,
    )


def _storage_stub(band, *, owner="full"):
    return SimpleNamespace(
        _allocator=SimpleNamespace(roles=ChainRoles(owner=owner)),
        _member=lambda _name: band,
    )


def _transfer(ids, *, host_dcp_size=1):
    return L2Transfer(
        host_pool=SimpleNamespace(dcp_size=host_dcp_size),
        device_pool=None,
        host_indices=None,
        device_indices=ids,
    )


def _tracked_pages(storage, band, sub_pool, transfer):
    ids = KVStorage._host_copy_ids(storage, sub_pool, transfer)
    return Band._physical_pages_of(band, ids, widened=False)


class TestHostCopyIds(CustomTestCase):
    def test_widened_envelope_pages_match_the_copy(self):
        """DCP 2 over an envelope: 4 physical rows a page, 8 logical ids a
        page; ids 24..31 name virtual page 3, bound to physical page 9."""
        band = _band_stub(page_size=8, pool_page_size=4)
        band.ids.virtual_to_physical[3] = 9
        ids = torch.arange(24, 32, dtype=torch.int64)
        copied = translate_token_ids(
            loc=ids,
            v2p=band.ids.virtual_to_physical,
            page_size=band.page_size,
            dcp_size=1,
            dcp_rank=0,
        )
        self.assertEqual(set((copied // band.page_size).tolist()), {9})
        tracked = _tracked_pages(_storage_stub(band), band, "full", _transfer(ids))
        self.assertEqual(tracked, {9})

    def test_host_pool_dcp_collapses_before_paging(self):
        """A host pool that shards under DCP hands widened ids: the copy
        translates them collapsed, and so does the tracking."""
        band = _band_stub(page_size=4, pool_page_size=4)
        band.ids.virtual_to_physical[2] = 11
        ids = torch.arange(16, 24, dtype=torch.int64)  # widened, DCP 2
        tracked = _tracked_pages(
            _storage_stub(band), band, "full", _transfer(ids, host_dcp_size=2)
        )
        self.assertEqual(tracked, {11})

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_host_resident_ids_translate_on_the_band_device(self):
        """A direct I/O backend hands the ids on the host; the band's tables
        live on the device."""
        band = _band_stub(page_size=4, pool_page_size=4, device="cuda")
        band.ids.virtual_to_physical[5] = 7
        ids = torch.arange(20, 24, dtype=torch.int64)  # on the host
        tracked = _tracked_pages(_storage_stub(band), band, "full", _transfer(ids))
        self.assertEqual(tracked, {7})


class TestRetirementBookkeeping(CustomTestCase):
    def test_urgent_retirements_keep_the_event_map_bounded(self):
        usage = PageUsage()
        for _ in range(10_000):
            event = object()
            usage.register_forward(event)
            usage.retire_through(event)  # an urgent wait retires it first
            usage.retire_forward(event)  # then the scheduler processes it
        self.assertEqual(len(usage.event_seq), 0)
        for _ in range(10_000):
            usage.register_forward(object())
            usage.retire_all()  # a settle waits the newest forward
        self.assertEqual(len(usage.event_seq), 0)


class TestSpanGrowth(CustomTestCase):
    def test_growth_is_the_pages_taken_at_either_end(self):
        self.assertEqual(_span_growth((10, 20), (10, 22)), [20, 21])
        self.assertEqual(_span_growth((10, 20), (8, 20)), [8, 9])
        self.assertEqual(_span_growth((10, 20), (9, 21)), [9, 20])
        self.assertEqual(_span_growth((10, 20), (10, 20)), [])

    def test_growth_from_an_empty_span(self):
        self.assertEqual(_span_growth((15, 15), (4, 7)), [4, 5, 6])
        self.assertEqual(_span_growth((3, 3), (4, 7)), [4, 5, 6])
        self.assertEqual(_span_growth((5, 5), (4, 7)), [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
