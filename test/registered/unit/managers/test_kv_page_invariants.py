"""Unit tests for SGLANG_CHECK_KV_PAGE_INVARIANTS: watermark + double-free checks."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.scheduler_components import (
    invariant_checker as invariant_checker_module,
)
from sglang.srt.managers.scheduler_components.invariant_checker import (
    _assert_kv_allocation_watermark_page_aligned,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE_SIZE = 256


def _make_checker(page_size=_PAGE_SIZE, row_width=4096, num_reqs=8, free_pages=None):
    rtt = torch.zeros((num_reqs, row_width), dtype=torch.int32)
    rtp = SimpleNamespace(req_to_token=rtt)
    if free_pages is None:
        free_pages = torch.arange(num_reqs * 4, dtype=torch.int64)
    alloc = SimpleNamespace(
        page_size=page_size,
        free_pages=free_pages,
        release_pages=torch.empty(0, dtype=torch.int64),
    )
    tc = SimpleNamespace(slots={})
    _ps, _rtp, _alloc, _tc = page_size, rtp, alloc, tc

    class _FakeChecker:
        page_size = _ps
        req_to_token_pool = _rtp
        token_to_kv_pool_allocator = _alloc
        tree_cache = _tc
        get_last_batch = lambda self: None
        count_memory_leak_warnings = 0

        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker as _RIC,
        )

        _check_kv_page_invariants = _RIC._check_kv_page_invariants

    return _FakeChecker(), rtt, tc, alloc


class _FakeReq:
    def __init__(self, rid, rpi, committed, allocated):
        self.rid = rid
        self.req_pool_idx = rpi
        self.kv_committed_len = committed
        self.kv = SimpleNamespace(kv_allocated_len=allocated, swa_evicted_seqlen=0)


class _FakeSlot:
    def __init__(self, rpi, committed, allocated):
        self.req_pool_idx = rpi
        self.kv_committed_len = committed
        self.kv = SimpleNamespace(kv_allocated_len=allocated, swa_evicted_seqlen=0)
        self.is_holding_kv = True


class TestKVPageInvariants(CustomTestCase):
    def test_clean_layout_no_warning(self):
        chk, rtt, tc, alloc = _make_checker(
            free_pages=torch.arange(100, 200, dtype=torch.int64)
        )
        rtt[0, :256] = torch.arange(_PAGE_SIZE)  # req 0 owns page 0
        rtt[1, :256] = torch.arange(_PAGE_SIZE, 2 * _PAGE_SIZE)  # req 1 owns page 1
        chk.get_last_batch = lambda: SimpleNamespace(
            reqs=[_FakeReq("a", 0, 256, 256), _FakeReq("b", 1, 200, 256)]
        )
        chk._check_kv_page_invariants()
        self.assertEqual(chk.count_memory_leak_warnings, 0)

    def test_committed_gt_allocated_raises(self):
        chk, rtt, tc, alloc = _make_checker()
        chk.get_last_batch = lambda: SimpleNamespace(reqs=[_FakeReq("a", 0, 145, 144)])
        with self.assertRaises(AssertionError):
            chk._check_kv_page_invariants()

    def test_slot_committed_gt_allocated_raises(self):
        chk, rtt, tc, alloc = _make_checker()
        chk.get_last_batch = lambda: None
        tc.slots = {"s1": _FakeSlot(0, 145, 144)}
        with self.assertRaises(AssertionError):
            chk._check_kv_page_invariants()

    def test_owner_references_free_page_raises(self):
        # req 0 owns page 5, but page 5 is in the free pool -> use-after-free.
        chk, rtt, tc, alloc = _make_checker(free_pages=torch.tensor([5, 6, 7]))
        rtt[0, :3] = torch.tensor(
            [5 * _PAGE_SIZE, 5 * _PAGE_SIZE + 1, 5 * _PAGE_SIZE + 2]
        )
        chk.get_last_batch = lambda: SimpleNamespace(
            reqs=[_FakeReq("a", 0, 3, _PAGE_SIZE)]
        )
        with self.assertRaises(ValueError):
            chk._check_kv_page_invariants()

    def test_free_pool_duplicate_raises(self):
        chk, rtt, tc, alloc = _make_checker(free_pages=torch.tensor([3, 3, 4]))
        rtt[0, :1] = torch.tensor([10 * _PAGE_SIZE])  # owner page 10, not in free
        chk.get_last_batch = lambda: SimpleNamespace(
            reqs=[_FakeReq("a", 0, 1, _PAGE_SIZE)]
        )
        with self.assertRaises(ValueError):
            chk._check_kv_page_invariants()

    def test_misaligned_watermark_raises(self) -> None:
        """A non-NPU request owner must publish page-aligned capacity."""
        chk, rtt, tc, alloc = _make_checker()
        chk.get_last_batch = lambda: SimpleNamespace(reqs=[_FakeReq("a", 0, 1, 3)])

        with self.assertRaisesRegex(AssertionError, "must be page-aligned"):
            chk._check_kv_page_invariants()

    def test_watermark_must_cover_committed_page(self) -> None:
        """A request watermark must cover the page containing committed KV."""
        with self.assertRaisesRegex(AssertionError, "must cover committed KV"):
            _assert_kv_allocation_watermark_page_aligned(
                allocated_len=_PAGE_SIZE,
                committed_len=_PAGE_SIZE + 1,
                alloc_page_size=_PAGE_SIZE,
                label="req a",
            )

    def test_npu_watermark_uses_legacy_real_length(self) -> None:
        """NPU watermark validation preserves the legacy real-length carve-out."""
        with mock.patch.object(invariant_checker_module, "_is_npu", True):
            _assert_kv_allocation_watermark_page_aligned(
                allocated_len=3,
                committed_len=3,
                alloc_page_size=_PAGE_SIZE,
                label="req a",
            )


if __name__ == "__main__":
    unittest.main()
