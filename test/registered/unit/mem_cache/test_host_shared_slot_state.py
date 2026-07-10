from __future__ import annotations

import unittest

import torch

from sglang.srt.mem_cache.storage.tensorcast_store.host_shared_slot_state import (
    HostSharedPageSlotManager,
    HostSharedPageSlotStaleTokenError,
    HostSharedPageSlotState,
    HostSharedPageSlotStateError,
    HostSharedPageSlotTracker,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class HostSharedPageSlotTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = HostSharedPageSlotTracker(page_size=32, page_num=4)

    def test_reserve_get_commit_and_retire_bumps_generation(self) -> None:
        slot_tokens = self.tracker.reserve_slots([1], logical_keys=["page-a"])
        reserved_snapshot = self.tracker.snapshot(1)
        self.assertEqual(reserved_snapshot.state, HostSharedPageSlotState.SLOT_RESERVED)
        self.assertEqual(reserved_snapshot.pin_count, 1)
        self.assertEqual(reserved_snapshot.page_start, 32)
        self.assertEqual(reserved_snapshot.logical_key, "page-a")

        self.tracker.mark_get_inflight(slot_tokens)
        inflight_snapshot = self.tracker.snapshot(1)
        self.assertEqual(inflight_snapshot.state, HostSharedPageSlotState.GET_IN_FLIGHT)
        self.assertEqual(inflight_snapshot.pin_count, 1)

        self.tracker.commit_get_success(slot_tokens)
        resident_snapshot = self.tracker.snapshot(1)
        self.assertEqual(resident_snapshot.state, HostSharedPageSlotState.SLOT_RESIDENT)
        self.assertEqual(resident_snapshot.pin_count, 0)
        self.assertEqual(resident_snapshot.slot_generation, 0)

        retired_tokens = self.tracker.retire_slots(slot_tokens)
        free_snapshot = self.tracker.snapshot(1)
        self.assertEqual(free_snapshot.state, HostSharedPageSlotState.SLOT_FREE)
        self.assertEqual(free_snapshot.pin_count, 0)
        self.assertEqual(free_snapshot.slot_generation, 1)
        self.assertIsNone(free_snapshot.logical_key)
        self.assertEqual(retired_tokens[0].slot_generation, 1)

    def test_stale_completion_is_rejected_after_retire_and_reuse(self) -> None:
        old_slot_tokens = self.tracker.reserve_slots([0], logical_keys=["page-a"])
        self.tracker.mark_get_inflight(old_slot_tokens)
        self.tracker.fail_get(old_slot_tokens)
        recycled_tokens = self.tracker.retire_slots(old_slot_tokens)
        new_slot_tokens = self.tracker.reserve_slots([0], logical_keys=["page-b"])

        self.assertEqual(recycled_tokens[0].slot_generation, 1)
        self.assertEqual(new_slot_tokens[0].slot_generation, 1)
        with self.assertRaises(HostSharedPageSlotStaleTokenError):
            self.tracker.commit_get_success(old_slot_tokens)

    def test_put_round_trip_keeps_resident_slot_valid(self) -> None:
        slot_tokens = self.tracker.reserve_slots([2], logical_keys=["page-c"])
        self.tracker.commit_get_success(slot_tokens)

        self.tracker.begin_put(slot_tokens)
        inflight_snapshot = self.tracker.snapshot(2)
        self.assertEqual(inflight_snapshot.state, HostSharedPageSlotState.PUT_IN_FLIGHT)
        self.assertEqual(inflight_snapshot.pin_count, 1)

        self.tracker.finish_put(slot_tokens)
        resident_snapshot = self.tracker.snapshot(2)
        self.assertEqual(resident_snapshot.state, HostSharedPageSlotState.SLOT_RESIDENT)
        self.assertEqual(resident_snapshot.pin_count, 0)
        self.assertEqual(resident_snapshot.logical_key, "page-c")

    def test_invalid_slot_must_retire_before_reuse(self) -> None:
        slot_tokens = self.tracker.reserve_slots([3], logical_keys=["page-d"])
        self.tracker.mark_get_inflight(slot_tokens)
        self.tracker.fail_get(slot_tokens)

        invalid_snapshot = self.tracker.snapshot(3)
        self.assertEqual(invalid_snapshot.state, HostSharedPageSlotState.SLOT_INVALID)
        self.assertEqual(invalid_snapshot.pin_count, 0)

        with self.assertRaises(HostSharedPageSlotStateError):
            self.tracker.reserve_slots([3], logical_keys=["page-e"])

        recycled_tokens = self.tracker.retire_slots(slot_tokens)
        self.assertEqual(recycled_tokens[0].slot_generation, 1)
        reused_tokens = self.tracker.reserve_slots([3], logical_keys=["page-e"])
        self.assertEqual(reused_tokens[0].slot_generation, 1)

    def test_put_inflight_slot_cannot_be_retired(self) -> None:
        slot_tokens = self.tracker.reserve_slots([0], logical_keys=["page-f"])
        self.tracker.commit_get_success(slot_tokens)
        self.tracker.begin_put(slot_tokens)

        with self.assertRaises(HostSharedPageSlotStateError):
            self.tracker.retire_slots(slot_tokens)

    def test_page_start_conversion_is_page_granular(self) -> None:
        self.assertEqual(self.tracker.slot_index_for_page_start(64), 2)
        self.assertEqual(self.tracker.page_start_for_slot_index(2), 64)
        with self.assertRaises(ValueError):
            self.tracker.slot_index_for_page_start(48)


class HostSharedPageSlotManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = HostSharedPageSlotManager(page_size=2, page_num=4)

    def test_retire_released_page_slots_retires_resident_slot(self) -> None:
        slot_tokens = self.manager.reserve_page_slots([0], logical_keys=["page-a"])
        self.manager.mark_page_get_inflight(slot_tokens)
        self.manager.commit_page_get_success(slot_tokens, logical_keys=["page-a"])

        self.manager.retire_released_page_slots(torch.tensor([0, 1], dtype=torch.int64))

        snapshot = self.manager.describe_page_slot(0)
        self.assertEqual(snapshot.state, HostSharedPageSlotState.SLOT_FREE)
        self.assertEqual(snapshot.slot_generation, 1)
        self.assertIsNone(snapshot.logical_key)

    def test_retire_released_page_slots_retires_invalid_slot(self) -> None:
        slot_tokens = self.manager.reserve_page_slots([2], logical_keys=["page-b"])
        self.manager.mark_page_get_inflight(slot_tokens)
        self.manager.fail_page_get(slot_tokens)

        self.manager.retire_released_page_slots(torch.tensor([2, 3], dtype=torch.int64))

        snapshot = self.manager.describe_page_slot(2)
        self.assertEqual(snapshot.state, HostSharedPageSlotState.SLOT_FREE)
        self.assertEqual(snapshot.slot_generation, 1)

    def test_retire_released_page_slots_rejects_inflight_slot(self) -> None:
        slot_tokens = self.manager.reserve_page_slots([4], logical_keys=["page-c"])
        self.manager.mark_page_get_inflight(slot_tokens)

        with self.assertRaises(HostSharedPageSlotStateError):
            self.manager.retire_released_page_slots(
                torch.tensor([4, 5], dtype=torch.int64)
            )


if __name__ == "__main__":
    unittest.main()
