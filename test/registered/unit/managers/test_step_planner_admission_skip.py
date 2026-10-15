"""Prefill admission keeps no state across steps: it is skipped only when the
current lists and pools prove it would admit nothing, and runs otherwise."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler_components.engine_state import (
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_NUM_REQ_SLOTS = 4


class _Req(SimpleNamespace):
    # Hashed by identity, as Req is.
    __hash__ = object.__hash__


def _req(rid, ring_slot=None):
    return _Req(
        rid=rid,
        ring_slot=ring_slot,
        is_prefill_only=False,
        beam_group=None,
        token_indices_to_pool=None,
        lora_id=None,
        output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=8),
        kv=SimpleNamespace(holds_mamba=False),
        init_next_round_input=Mock(),
    )


class _Adder:
    """Finds no KV room for any request."""

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = None
        self.offered = []

    def add_one_req(self, req, **kwargs):
        self.offered.append(req)
        return AddReqResult.NO_TOKEN


class _AdmissionCase(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology(pp_max_micro_batch_size=64))
        self.req_to_token_pool = SimpleNamespace(available_size=lambda: self.free_rows)
        self.free_rows = _NUM_REQ_SLOTS
        self.adders = []

        def make_adder(*args, **kwargs):
            adder = _Adder()
            self.adders.append(adder)
            return adder

        adder_patch = patch(
            "sglang.srt.managers.scheduler_components.step_planner.PrefillAdder",
            side_effect=make_adder,
        )
        adder_patch.start()
        self.addCleanup(adder_patch.stop)
        self.planner = self._planner()

    def _planner(self) -> StepPlanner:
        s = StepPlanner.__new__(StepPlanner)
        s.counters = LaunchCounters()
        s.counters.processed_tokens_counter = 0
        s.flags = EngineFlags()
        s.core_lists = CoreLists()
        s.core_lists.decoding = []
        s.core_lists.waiting_queue = []
        s.core_lists.chunked_req = None
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.is_mixed_chunk = False
        s.min_free_slots_delayer = None
        s.dynamic_chunk_sizer = None
        s.chunked_prefill_size = None
        s.max_prefill_tokens = 1 << 20
        s.max_running_requests = _NUM_REQ_SLOTS
        s.max_prefill_bs = 0
        s.truncation_align_size = None
        s.new_token_ratio_tracker = SimpleNamespace(current=0.5)
        s.policy = SimpleNamespace(calc_priority=Mock())
        s.req_to_token_pool = self.req_to_token_pool
        s.token_to_kv_pool_allocator = None
        s.tree_cache = SimpleNamespace(buffer_pipeline=None)
        s.hisparse_coordinator = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.beam_coordinator = BeamCoordinator(
            model_config=None,
            spec_algorithm=None,
            max_req_len=0,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            future_map=None,
            core_lists=s.core_lists,
        )
        return s

    def _admit(self, slot=None):
        """Run one admission pass; return the requests it offered the adder,
        or None when it was skipped before the adder."""
        num_adders = len(self.adders)
        self.planner._get_new_batch_prefill_raw(
            slot or ScheduleSlot(), prefill_delayer_single_pass=None
        )
        if len(self.adders) == num_adders:
            return None
        return [req.rid for req in self.adders[-1].offered]


class TestAdmissionSkipReason(_AdmissionCase):
    def test_no_waiting_request(self):
        self.assertEqual(self.planner.admission_skip_reason([]), "no waiting request")
        self.assertIsNone(self._admit())

    def test_no_request_slot(self):
        self.planner.core_lists.waiting_queue = [_req("w")]
        self.free_rows = 0

        self.assertEqual(self.planner.admission_skip_reason([]), "no request slot")
        self.assertIsNone(self._admit())

    def test_a_chunked_request_always_runs(self):
        self.planner.core_lists.chunked_req = _req("c")
        self.free_rows = 0

        self.assertIsNone(self.planner.admission_skip_reason([]))

    def test_priority_preemption_runs_without_a_request_slot(self):
        # Preempting a running request frees its slot.
        self.planner.core_lists.waiting_queue = [_req("w")]
        self.planner.enable_priority_preemption = True
        self.free_rows = 0

        self.assertIsNone(self.planner.admission_skip_reason([_req("r")]))

    def test_request_cap_counts_the_scheduled_pp_slot_only(self):
        enter_scope(self, published_topology(pp_size=2, pp_max_micro_batch_size=2))
        decoding = [_req("a", 0), _req("b", 0), _req("c", 1)]
        self.planner.core_lists.decoding = decoding
        self.planner.core_lists.waiting_queue = [_req("w")]

        # Slot 0 holds its micro-batch cap; slot 1 has room.
        self.assertEqual(self._admit(ScheduleSlot(index=0)), None)
        self.assertEqual(self._admit(ScheduleSlot(index=1)), ["w"])


class TestFailedAdmissionKeepsNoState(_AdmissionCase):
    """A pass that admits nothing does not close the next one: admission runs
    again whatever changed in between, so no event has to reopen it."""

    def setUp(self):
        super().setUp()
        self.running = [_req("r0"), _req("r1")]
        self.planner.core_lists.decoding = list(self.running)
        self.planner.core_lists.waiting_queue = [_req("w0"), _req("w1")]

    def test_no_token_then_nothing_changes(self):
        self.assertEqual(self._admit(), ["w0"])
        self.assertEqual(self._admit(), ["w0"])

    def test_former_reopen_events(self):
        """Each event that used to reopen a closed latch: admission after it
        offers the head request, as it does after no event at all."""
        planner = self.planner
        lists = planner.core_lists

        def request_finishes():
            lists.decoding = lists.decoding[1:]

        def retract():
            retracted = lists.decoding.pop()
            lists.waiting_queue.append(retracted)

        def hisparse_ready():
            lists.decoding.append(_req("staged"))

        def flush():
            lists.decoding = []

        events = {
            "none": lambda: None,
            "request finishes": request_finishes,
            "retract": retract,
            "hisparse ready": hisparse_ready,
            "flush": flush,
        }
        for name, event in events.items():
            with self.subTest(event=name):
                lists.decoding = list(self.running)
                lists.waiting_queue = [_req("w0"), _req("w1")]
                self.assertEqual(self._admit(), ["w0"])
                event()
                self.assertEqual(self._admit(), ["w0"])

    def test_prefill_decode_interval_defers_then_admits(self):
        self.planner.prefill_decode_interval = 1
        self.planner._prefill_decode_interval_remaining = 1

        self.assertTrue(self.planner.should_defer_prefill())
        self.assertFalse(self.planner.should_defer_prefill())
        self.assertEqual(self._admit(), ["w0"])

    def test_pp_slot_failure_does_not_close_another_slot(self):
        enter_scope(self, published_topology(pp_size=2, pp_max_micro_batch_size=64))
        for req, ring_slot in zip(self.running, (0, 1)):
            req.ring_slot = ring_slot

        self.assertEqual(self._admit(ScheduleSlot(index=0)), ["w0"])
        self.assertEqual(self._admit(ScheduleSlot(index=1)), ["w0"])
        self.assertEqual(self._admit(ScheduleSlot(index=0)), ["w0"])

    def test_nothing_running_retries_every_pass(self):
        # With nothing decoding no release is coming; a failure must not
        # stop the passes that admit once the pool frees.
        self.planner.core_lists.decoding = []
        self.free_rows = 0
        self.assertIsNone(self._admit())

        self.free_rows = _NUM_REQ_SLOTS
        self.assertEqual(self._admit(), ["w0"])


if __name__ == "__main__":
    unittest.main()
