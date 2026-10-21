"""Readback must expose staging capacity without advancing its CUDA fences."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.deferred_draft_prefill import DeferredDraftPrefillPool
from sglang.srt.disaggregation.deferred_draft_prefill_lifecycle import (
    DeferredDraftPrefillSourceFence,
    DeferredDraftPrefillSourceRetirements,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import GetInternalStateReq
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def pool(size=4):
    return DeferredDraftPrefillPool(
        size, 8, 4, "torch.bfloat16", "cpu", allocate_storage=False
    )


def request(rid="request"):
    return SimpleNamespace(
        rid=rid,
        bootstrap_room=100,
        pd_deferred_draft_source_generation="generation",
        origin_input_ids=[1, 2, 3],
        metadata_buffer_index=0,
        disagg_kv_sender=SimpleNamespace(
            deferred_draft_prefill_fence=DeferredDraftPrefillSourceFence()
        ),
    )


class TestDeferredDraftPrefillStats(unittest.TestCase):
    def controller(self, hidden, mode=DisaggregationMode.PREFILL, retirements=None):
        scheduler = SimpleNamespace(
            disaggregation_mode=mode,
            deferred_draft_prefill_pool=hidden,
            deferred_draft_source_retirements=retirements,
            tp_worker=SimpleNamespace(
                get_decode_cuda_graph_bs=Mock(return_value=[1, 2])
            ),
        )
        return SchedulerInternalStateController(scheduler, record_step_time=False)

    def read_internal_state(self, controller):
        # Exercise the real public controller entry and disaggregation builder;
        # other contributors are independent of deferred staging ownership.
        graph = dict.fromkeys(
            ("decode", "target_verify", "draft_decode", "draft_extend"), 0.0
        )
        context = SimpleNamespace(
            resolved_server_args_dict=lambda: {"memory_usage": {"graph": graph}}
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "sglang.srt.managers.scheduler_components.internal_state.get_context",
                    return_value=context,
                )
            )
            for name in (
                "_add_topology",
                "_add_memory_usage",
                "_add_elastic_ep",
                "_add_speculative",
                "_add_step_times",
                "_add_rust_server",
                "_add_env_vars",
            ):
                stack.enter_context(patch.object(controller, name))
            return controller.get_internal_state(GetInternalStateReq()).internal_state

    def test_hidden_slot_conservation_and_snapshot_has_no_cuda_side_effects(self):
        hidden = pool()
        active, pending, unknown = [request(rid) for rid in ("a", "b", "c")]
        for req in (active, pending, unknown):
            hidden.alloc(req)
        event = SimpleNamespace(query=Mock(side_effect=AssertionError("CUDA query")))
        hidden.free(pending, completion_event=event)
        hidden.quarantine(unknown, reason="late RDMA completion unknown")
        hidden.reclaim = Mock(side_effect=AssertionError("stats attempted reclaim"))
        expected = {
            "total": 4,
            "free": 1,
            "active": 1,
            "pending_cuda": 1,
            "quarantined": 1,
        }
        first = hidden.get_slot_stats()
        self.assertEqual(first, expected)
        self.assertEqual(
            first["total"], sum(v for k, v in first.items() if k != "total")
        )
        first["free"] = 99
        self.assertEqual(hidden.get_slot_stats(), expected)
        controller = self.controller(hidden, mode=DisaggregationMode.DECODE)
        state = self.read_internal_state(controller)
        self.assertEqual(state["deferred_draft_prefill"], {"hidden_slots": expected})
        hidden.reclaim.assert_not_called()
        event.query.assert_not_called()

    def test_prefill_readback_reports_pending_source_retirements(self):
        hidden = pool(1)
        req = request()
        hidden.alloc(req)
        req.deferred_draft_prefill_pool = hidden
        allocator = SimpleNamespace(free=Mock())
        retirements = DeferredDraftPrefillSourceRetirements(allocator)
        req.disagg_kv_sender.deferred_draft_prefill_fence.submit()
        retirements.register(req)
        retirements.snapshot = Mock(side_effect=AssertionError("stats polled fence"))
        retirements.resolve = Mock(side_effect=AssertionError("stats retired owner"))
        hidden.reclaim = Mock(side_effect=AssertionError("stats attempted reclaim"))
        stats = self.read_internal_state(
            self.controller(hidden, retirements=retirements)
        )["deferred_draft_prefill"]
        self.assertEqual(
            stats["hidden_slots"],
            {
                "total": 1,
                "free": 0,
                "active": 1,
                "pending_cuda": 0,
                "quarantined": 0,
            },
        )
        self.assertEqual(stats["pending_source_retirements"], 1)
        self.assertEqual(stats["quarantined_source_metadata_slots"], 0)
        retirements.snapshot.assert_not_called()
        retirements.resolve.assert_not_called()
        allocator.free.assert_not_called()
        hidden.reclaim.assert_not_called()

    def test_decode_idle_includes_completed_but_unreclaimed_cuda_slots(self):
        hidden = pool(8)
        req = request()
        hidden.alloc(req)
        event = SimpleNamespace(query=Mock(return_value=True))
        hidden.free(req, completion_event=event)
        hidden.reclaim = Mock(side_effect=AssertionError("stats attempted reclaim"))
        stats = self.read_internal_state(
            self.controller(hidden, mode=DisaggregationMode.DECODE)
        )["deferred_draft_prefill"]["hidden_slots"]
        self.assertEqual(
            stats,
            {
                "total": 8,
                "free": 7,
                "active": 0,
                "pending_cuda": 1,
                "quarantined": 0,
            },
        )
        self.assertEqual(stats["free"] + stats["pending_cuda"], stats["total"])
        event.query.assert_not_called()
        hidden.reclaim.assert_not_called()

    def test_unknown_drain_reports_quarantine_even_when_pending_registry_is_empty(self):
        hidden = pool(1)
        req = request()
        hidden.alloc(req)
        req.deferred_draft_prefill_pool = hidden
        allocator = SimpleNamespace(free=Mock())
        retirements = DeferredDraftPrefillSourceRetirements(allocator)
        fence = req.disagg_kv_sender.deferred_draft_prefill_fence
        fence.submit()
        fence.poison("backend lost an unsettled batch handle")
        retirements.register(req)
        retirements.resolve()
        self.assertFalse(retirements)
        stats = self.read_internal_state(
            self.controller(hidden, retirements=retirements)
        )["deferred_draft_prefill"]
        self.assertEqual(
            stats["hidden_slots"],
            {
                "total": 1,
                "free": 0,
                "active": 0,
                "pending_cuda": 0,
                "quarantined": 1,
            },
        )
        self.assertEqual(stats["pending_source_retirements"], 0)
        self.assertEqual(stats["quarantined_source_metadata_slots"], 1)
        allocator.free.assert_not_called()

    def test_prefill_without_retirement_owner_reports_zero_counts(self):
        stats = self.read_internal_state(self.controller(pool(1)))[
            "deferred_draft_prefill"
        ]
        self.assertEqual(stats["pending_source_retirements"], 0)
        self.assertEqual(stats["quarantined_source_metadata_slots"], 0)

    def test_disabled_or_missing_pool_preserves_the_original_readback_schema(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                controller = self.controller(None)
                if missing:
                    del controller.scheduler.deferred_draft_prefill_pool
                state = self.read_internal_state(controller)
                self.assertEqual(
                    set(state),
                    {
                        "memory_usage",
                        "disaggregation_mode",
                        "decode_cuda_graph_bs",
                        "decode_cuda_graph_memory_gb",
                    },
                )
                self.assertNotIn("deferred_draft_prefill", state)


if __name__ == "__main__":
    unittest.main()
