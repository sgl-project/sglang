"""Addressing invariant for the unified_kv c4 overlap-state ring.

``translate_from_swa_loc_to_state_loc`` divides the SWA location by
``swa_page_size`` to get the row group. That is exact while SWA is paged at
``swa_page_size`` (the non-unified paged mode, where the quotient is the SWA page
id), but unified_kv strides the device SWA ring by ``unified_swa_ring_size``
(sliding_window, plus the draft tokens under speculative decode). Divisor and
stride are independent knobs, and when the stride is the smaller of the two,
several request slots land in one row group and share c4 state rows -- which are
per-request state.

kv_cache_configurator pins swa_page_size to the schedule page size (asserted
== 256) while DeepSeek-V4's window is 128, so the shipped config aliases 2:1.
That is why the HiCache capture/restore helpers address state by (request slot,
position) instead; the two tests below pin both halves of that.

Run:
  PYTHONPATH=<worktree>/python python -m pytest \
      test/srt/mem_cache/test_swa_state_loc_addressing.py -q
"""

import types
import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import get_compress_state_ring_size
from sglang.srt.mem_cache.unified_cache.components.swa_component import (
    _state_locs_for_window,
)

SCHEDULE_PAGE_SIZE = 256  # kv_cache_configurator asserts this in paged swa mode
DSV4_SLIDING_WINDOW = 128  # window_size in the model config
REQ_SLOTS = 64


def _state_rows(swa_page_size, ring_size, slot, swa_ring):
    """State rows the c4 ride touches for request slot ``slot``, through the
    production arithmetic. The translator reads nothing but those two attrs, so a
    stub keeps this off the GPU and out of the pool allocator."""
    stub = types.SimpleNamespace(swa_page_size=swa_page_size, ring_size=ring_size)
    swa_loc = torch.arange(slot * swa_ring, (slot + 1) * swa_ring, dtype=torch.int64)
    return set(
        CompressStatePool.translate_from_swa_loc_to_state_loc(stub, swa_loc).tolist()
    )


def _restore_rows(slot, ring_size, B, ratio):
    """Rows ``_restore_state_windows`` writes for request slot ``slot`` -- through
    the production helper, so a revert to the swa_loc route fails this."""
    stub = types.SimpleNamespace(ring_size=ring_size)
    stub.translate_from_req_position_to_state_loc = lambda r, pos: (
        CompressStatePool.translate_from_req_position_to_state_loc(stub, r, pos)
    )
    locs = _state_locs_for_window(stub, slot, B, ratio, torch.device("cpu"))
    return set(locs.tolist())


class TestRestoreRowsAreRequestScoped(unittest.TestCase):
    def test_no_two_request_slots_share_a_restore_row(self):
        ring_size = get_compress_state_ring_size(4, is_speculative=False)
        rows = [
            _restore_rows(slot, ring_size, B=SCHEDULE_PAGE_SIZE, ratio=4)
            for slot in range(REQ_SLOTS)
        ]
        shared = [
            (a, b)
            for a in range(REQ_SLOTS)
            for b in range(a + 1, REQ_SLOTS)
            if rows[a] & rows[b]
        ]
        self.assertEqual(shared, [], f"restore rows alias, e.g. {shared[:4]}")


class TestUnifiedSwaStateLocAddressing(unittest.TestCase):
    def _shared_slots(self, swa_page_size, swa_ring):
        ring_size = get_compress_state_ring_size(4, is_speculative=False)
        rows = [
            _state_rows(swa_page_size, ring_size, slot, swa_ring)
            for slot in range(REQ_SLOTS)
        ]
        return [
            (a, b)
            for a in range(REQ_SLOTS)
            for b in range(a + 1, REQ_SLOTS)
            if rows[a] & rows[b]
        ]

    def test_swa_loc_route_aliases_under_the_shipped_config(self):
        # Why the capture/restore helpers cannot go through the SWA slot: with
        # divisor 256 and ring stride 128, slot pairs (0,1), (2,3), ... land on
        # one row group. Kept as an assertion because the primitive still ships
        # for the non-unified paged mode, where the quotient really is the page.
        shared = self._shared_slots(SCHEDULE_PAGE_SIZE, DSV4_SLIDING_WINDOW)
        self.assertEqual(len(shared), REQ_SLOTS // 2)

    def test_no_sharing_once_divisor_matches_the_ring_stride(self):
        # control: the invariant holds as soon as the divisor is the same quantity
        # the device ring is strided by
        self.assertEqual(
            self._shared_slots(DSV4_SLIDING_WINDOW, DSV4_SLIDING_WINDOW), []
        )


if __name__ == "__main__":
    unittest.main()
