"""Unit tests: scheduler + DPC wiring for Mooncake-native scale-down.

These tests validate:
  * ``ElasticScaleUpdateReq.direction`` field round-trips through the IO struct.
  * DPC's ``_dispatch_elastic_scale_update`` routes grow vs. shrink into the
    correct slot mutator (``add_elastic_workers`` vs. ``remove_elastic_workers``).
  * ``ElasticEPStateManager.begin_scale`` accepts a shrink target and
    ``get_shrink_direction`` / ``get_pending_shrink_ranks`` return the right
    values; ``commit_scale`` transitions to ``serving_shrunk``.

No distributed backend is required.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.managers.io_struct import ElasticScaleUpdateReq


class TestElasticScaleUpdateReqSchema(unittest.TestCase):
    """The DPC's dispatcher relies on ``direction`` being defaulted and
    round-trippable through the io_struct pickling path."""

    def test_default_direction_is_grow(self):
        msg = ElasticScaleUpdateReq(
            success=True,
            effective_ep_size=4,
            slot_offset=3,
            slot_count=1,
        )
        self.assertEqual(msg.direction, "grow")

    def test_shrink_direction_carries(self):
        msg = ElasticScaleUpdateReq(
            success=True,
            effective_ep_size=3,
            slot_offset=3,
            slot_count=1,
            direction="shrink",
        )
        self.assertEqual(msg.direction, "shrink")
        self.assertEqual(msg.slot_offset, 3)
        self.assertEqual(msg.slot_count, 1)


class _FakeDPC:
    """Minimal shim for DPC methods invoked by _dispatch_elastic_scale_update."""

    def __init__(self):
        self.added = []
        self.removed = []

    def add_elastic_workers(self, slot_offset, slot_count):
        self.added.append((slot_offset, slot_count))

    def remove_elastic_workers(self, slot_offset, slot_count):
        self.removed.append((slot_offset, slot_count))


class TestDpcDispatchRouting(unittest.TestCase):
    def test_grow_routes_to_add(self):
        from sglang.srt.managers.data_parallel_controller import (
            DataParallelController,
        )

        fake = _FakeDPC()
        DataParallelController._dispatch_elastic_scale_update(
            fake,
            ElasticScaleUpdateReq(
                success=True,
                effective_ep_size=5,
                slot_offset=4,
                slot_count=1,
                direction="grow",
            ),
        )
        self.assertEqual(fake.added, [(4, 1)])
        self.assertEqual(fake.removed, [])

    def test_shrink_routes_to_remove(self):
        from sglang.srt.managers.data_parallel_controller import (
            DataParallelController,
        )

        fake = _FakeDPC()
        DataParallelController._dispatch_elastic_scale_update(
            fake,
            ElasticScaleUpdateReq(
                success=True,
                effective_ep_size=3,
                slot_offset=3,
                slot_count=1,
                direction="shrink",
            ),
        )
        self.assertEqual(fake.added, [])
        self.assertEqual(fake.removed, [(3, 1)])

    def test_failed_scale_is_ignored(self):
        from sglang.srt.managers.data_parallel_controller import (
            DataParallelController,
        )

        fake = _FakeDPC()
        DataParallelController._dispatch_elastic_scale_update(
            fake,
            ElasticScaleUpdateReq(
                success=False,
                effective_ep_size=4,
                slot_offset=3,
                slot_count=1,
                direction="shrink",
                error="mock failure",
            ),
        )
        self.assertEqual(fake.added, [])
        self.assertEqual(fake.removed, [])


class TestElasticEPStateManagerShrinkFsm(unittest.TestCase):
    """The state-manager FSM has to accept shrink targets symmetrically with
    grow (same ``begin_scale`` entrypoint, direction discovered by comparing
    ``pending_ep_size`` vs. ``effective_ep_size``)."""

    def setUp(self):
        # Reset singleton so each test builds a fresh manager.
        ElasticEPStateManager._instance = None
        ElasticEPStateManager._on_scale = None

    def _install_state(self, effective_size: int, max_size: int = 4):
        """Build a manager instance without depending on torch.distributed."""
        active = torch.ones(max_size, dtype=torch.int32)
        active[effective_size:] = 0
        inst = SimpleNamespace(
            active_ranks=active,
            last_active_ranks=active.clone(),
            active_ranks_cpu=active.detach().cpu().clone(),
            effective_ep_size=effective_size,
            pending_ep_size=None,
            scale_phase="idle",
            last_error=None,
            pending_since=None,
            original_ep_size=effective_size,
            has_scaled=False,
            ep_join_rank_offset=0,
        )
        # Attach minimal state-mutating methods used by commit_scale/reset.
        def _snapshot():
            inst.last_active_ranks = inst.active_ranks.clone()

        def _sync_cpu():
            inst.active_ranks_cpu = inst.active_ranks.detach().cpu().clone()

        def _reset():
            inst.active_ranks.zero_()
            inst.active_ranks[: inst.effective_ep_size] = 1
            _snapshot()
            _sync_cpu()

        inst.snapshot_active_to_last = _snapshot
        inst.sync_active_to_cpu = _sync_cpu
        inst.reset = _reset
        inst.is_active_equal_last = lambda: torch.equal(
            inst.active_ranks, inst.last_active_ranks
        )
        ElasticEPStateManager._instance = inst
        return inst

    def test_begin_scale_shrink_direction(self):
        self._install_state(effective_size=4)
        ok = ElasticEPStateManager.begin_scale(3)
        self.assertTrue(ok)
        self.assertEqual(ElasticEPStateManager.get_pending_ep_size(), 3)
        self.assertEqual(ElasticEPStateManager.get_shrink_direction(), "shrink")
        self.assertEqual(ElasticEPStateManager.get_pending_shrink_ranks(), [3])

    def test_begin_scale_grow_direction(self):
        self._install_state(effective_size=3, max_size=4)
        ok = ElasticEPStateManager.begin_scale(4)
        self.assertTrue(ok)
        self.assertEqual(ElasticEPStateManager.get_shrink_direction(), "grow")
        self.assertEqual(ElasticEPStateManager.get_pending_shrink_ranks(), [])

    def test_commit_scale_shrink_sets_serving_shrunk(self):
        self._install_state(effective_size=4)
        ElasticEPStateManager.begin_scale(3)
        ElasticEPStateManager.mark_draining()
        self.assertEqual(ElasticEPStateManager.get_scale_phase(), "draining")
        ElasticEPStateManager.mark_retiring()
        self.assertEqual(ElasticEPStateManager.get_scale_phase(), "retiring")
        ElasticEPStateManager.mark_reconfiguring()
        self.assertEqual(ElasticEPStateManager.get_scale_phase(), "reconfiguring")
        ElasticEPStateManager.commit_scale()
        self.assertEqual(ElasticEPStateManager.get_scale_phase(), "serving_shrunk")
        self.assertEqual(ElasticEPStateManager.get_effective_ep_size(), 3)
        self.assertIsNone(ElasticEPStateManager.get_pending_ep_size())

    def test_commit_scale_grow_sets_serving_expanded(self):
        self._install_state(effective_size=3, max_size=4)
        ElasticEPStateManager.begin_scale(4)
        ElasticEPStateManager.mark_joining()
        ElasticEPStateManager.commit_scale()
        self.assertEqual(ElasticEPStateManager.get_scale_phase(), "serving_expanded")
        self.assertEqual(ElasticEPStateManager.get_effective_ep_size(), 4)


class TestSchedulerShrinkGuardRails(unittest.TestCase):
    """``handle_scale_elastic_ep`` must reject non-mooncake shrink requests
    and same-size / zero-size requests without corrupting state."""

    def _fake_scheduler(self, effective_ep_size=4, elastic_ep_backend="mooncake"):
        ElasticEPStateManager._instance = SimpleNamespace(
            effective_ep_size=effective_ep_size,
            pending_ep_size=None,
            scale_phase="idle",
            active_ranks_cpu=torch.ones(effective_ep_size, dtype=torch.int32),
            last_error=None,
        )

        sched = SimpleNamespace()
        sched.server_args = SimpleNamespace(
            max_ep_size=effective_ep_size,
            elastic_ep_backend=elastic_ep_backend,
        )
        return sched

    def _call_handler(self, sched, new_ep_size):
        from sglang.srt.managers.io_struct import ScaleElasticEPReqInput
        from sglang.srt.managers.scheduler import Scheduler

        with mock.patch.object(
            ElasticEPStateManager, "get_effective_ep_size", return_value=4
        ), mock.patch.object(
            ElasticEPStateManager, "is_scaling", return_value=False
        ):
            return Scheduler.handle_scale_elastic_ep(
                sched, ScaleElasticEPReqInput(new_ep_size=new_ep_size)
            )

    def test_shrink_on_nixl_rejected(self):
        sched = self._fake_scheduler(elastic_ep_backend="nixl")
        out = self._call_handler(sched, new_ep_size=3)
        self.assertFalse(out.success)
        self.assertIn("requires --elastic-ep-backend mooncake", out.message)

    def test_zero_target_rejected(self):
        sched = self._fake_scheduler()
        out = self._call_handler(sched, new_ep_size=0)
        self.assertFalse(out.success)
        self.assertIn("must be >= 1", out.message)

    def test_same_size_rejected(self):
        sched = self._fake_scheduler()
        out = self._call_handler(sched, new_ep_size=4)
        self.assertFalse(out.success)
        self.assertIn("nothing to do", out.message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
