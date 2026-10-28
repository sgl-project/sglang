"""When prefill admission is skipped: the request cap and an empty queue are
read each pass; a pass that found no KV room for the head is remembered until
a request leaves, the slot's requests change or the waiting queue changes."""

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
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.step_planner import (
    DecodeFirst,
    StepPlanner,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

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
        return_logprob=False,
        beam_group=None,
        token_indices_to_pool=None,
        lora_id=None,
        output_ids=[],
        seqlen=8,
        sampling_params=SimpleNamespace(max_new_tokens=8),
        kv=SimpleNamespace(holds_mamba=False, holds_kv=False, kv_allocated_len=8),
        init_next_round_input=Mock(),
    )


class _Adder:
    """Admits the first num_fit requests offered, then finds no KV room."""

    num_fit = 0
    # The verdict after an admission: NO_TOKEN when it used up the budget.
    verdict_after_fit = AddReqResult.CONTINUE

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = None
        self.offered = []

    def budget_state(self):
        return AddReqResult.CONTINUE

    def add_one_req(self, req, **kwargs):
        self.offered.append(req)
        if len(self.can_run_list) < self.num_fit:
            self.can_run_list.append(req)
            return self.verdict_after_fit
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

        self.make_adder = make_adder
        self.planner = self._planner()

    def _planner(self) -> StepPlanner:
        s = StepPlanner.__new__(StepPlanner)
        s.prefill_adder_class = self.make_adder
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
        s.policy = SimpleNamespace(
            calc_priority=Mock(), enable_priority_scheduling=False
        )
        s.req_to_token_pool = self.req_to_token_pool
        s.token_to_kv_pool_allocator = None
        s.tree_cache = SimpleNamespace(buffer_pipeline=None)
        s.hisparse_coordinator = None
        s.mode_owners = ModeOwners()
        s.request_retirer = Mock()
        s._no_kv_room = {}
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
        self.assertEqual(
            self.planner.admission_skip_reason(ScheduleSlot(), []),
            "no waiting request",
        )
        self.assertIsNone(self._admit())

    def test_no_request_slot(self):
        self.planner.core_lists.waiting_queue = [_req("w")]
        self.free_rows = 0

        self.assertEqual(
            self.planner.admission_skip_reason(ScheduleSlot(), []), "no request slot"
        )
        self.assertIsNone(self._admit())

    def test_no_request_slot_reads_no_waiting_request(self):
        # Without streaming sessions no waiting request holds a row, so a
        # pass with every row spent skips without reading the queue.
        reads = []

        class _Kv(SimpleNamespace):
            @property
            def holds_kv(self):
                reads.append(1)
                return False

        waiting = [_req(f"w{i}") for i in range(3)]
        for req in waiting:
            req.kv = _Kv(holds_mamba=False, kv_allocated_len=8)
        self.planner.core_lists.waiting_queue = waiting
        self.free_rows = 0

        self.assertEqual(
            self.planner.admission_skip_reason(ScheduleSlot(), []), "no request slot"
        )
        self.assertEqual(reads, [])

    def test_a_chunked_request_always_runs(self):
        self.planner.core_lists.chunked_req = _req("c")
        self.free_rows = 0

        self.assertIsNone(self.planner.admission_skip_reason(ScheduleSlot(), []))

    def test_priority_preemption_runs_without_a_request_slot(self):
        # Preempting a running request frees its slot.
        self.planner.core_lists.waiting_queue = [_req("w")]
        self.planner.enable_priority_preemption = True
        self.free_rows = 0

        self.assertIsNone(
            self.planner.admission_skip_reason(ScheduleSlot(), [_req("r")])
        )

    def test_request_cap_counts_the_scheduled_pp_slot_only(self):
        enter_scope(self, published_topology(pp_size=2, pp_max_micro_batch_size=2))
        decoding = [_req("a", 0), _req("b", 0), _req("c", 1)]
        self.planner.core_lists.decoding = decoding
        self.planner.core_lists.waiting_queue = [_req("w")]

        # Slot 0 holds its micro-batch cap; slot 1 has room.
        self.assertEqual(self._admit(ScheduleSlot(index=0)), None)
        self.assertEqual(self._admit(ScheduleSlot(index=1)), ["w"])


class TestNoKvRoom(_AdmissionCase):
    """A pass that stopped on a request finding no KV room is not repeated
    until a request leaves, a request slot frees or the queue head changes;
    each change is made alone below."""

    def setUp(self):
        super().setUp()
        self.running = [_req("r0"), _req("r1")]
        self.planner.core_lists.decoding = list(self.running)
        self.planner.core_lists.waiting_queue = [_req("w0"), _req("w1")]
        self.assertEqual(self._admit(), ["w0"])

    def test_skipped_while_nothing_changes(self):
        # The new token ratio's decay is not waited for.
        self.planner.new_token_ratio_tracker.current = 0.1

        self.assertEqual(
            self.planner.admission_skip_reason(ScheduleSlot(), self.running),
            "no KV room",
        )
        self.assertIsNone(self._admit())
        self.assertIsNone(self._admit())

    def test_each_input_reopens(self):
        lists = self.planner.core_lists

        def request_leaves():
            lists.decoding = lists.decoding[1:]
            self.free_rows += 1

        def request_finishes_in_another_slot():
            self.free_rows += 1

        def retract():
            lists.waiting_queue.append(lists.decoding.pop())
            self.free_rows += 1

        def head_aborted():
            lists.waiting_queue.pop(0)

        def retracted_request_requeued_at_head():
            lists.waiting_queue.insert(0, lists.decoding.pop())
            self.free_rows += 1

        def flush():
            lists.decoding = []
            self.free_rows = _NUM_REQ_SLOTS

        events = {
            "request leaves": request_leaves,
            "request finishes in another slot": request_finishes_in_another_slot,
            "retract": retract,
            "head aborted": head_aborted,
            "retracted request requeued at head": retracted_request_requeued_at_head,
            "flush": flush,
        }
        for name, event in events.items():
            with self.subTest(event=name):
                lists.decoding = list(self.running)
                lists.waiting_queue = [_req("w0"), _req("w1"), _req("w2")]
                self.free_rows = _NUM_REQ_SLOTS - len(self.running)
                self.assertEqual(self._admit(), ["w0"])
                self.assertIsNone(self._admit())
                event()
                self.assertIsNotNone(self._admit())

    def test_kept_while_nothing_is_released(self):
        lists = self.planner.core_lists

        def request_joins_decoding():
            # An admitted request joins behind the running ones.
            lists.decoding.append(_req("admitted"))
            self.free_rows -= 1

        def arrival_behind_the_head():
            lists.waiting_queue.append(_req("new"))

        def request_aborted_behind_the_head():
            lists.waiting_queue.pop()

        events = {
            "request joins decoding": request_joins_decoding,
            "arrival behind the head": arrival_behind_the_head,
            "request aborted behind the head": request_aborted_behind_the_head,
        }
        for name, event in events.items():
            with self.subTest(event=name):
                lists.decoding = list(self.running)
                lists.waiting_queue = [_req("w0"), _req("w1"), _req("w2")]
                self.free_rows = _NUM_REQ_SLOTS - len(self.running)
                self.assertEqual(self._admit(), ["w0"])
                event()
                self.assertIsNone(self._admit())

    def test_arrival_reopens_only_under_priority_scheduling(self):
        lists = self.planner.core_lists
        for priority_scheduling in (False, True):
            with self.subTest(priority_scheduling=priority_scheduling):
                self.planner.policy.enable_priority_scheduling = priority_scheduling
                lists.waiting_queue = [_req("w0"), _req("w1")]
                self.assertEqual(self._admit(), ["w0"])
                self.assertIsNone(self._admit())
                # A priority arrival is sorted first only by the next pass.
                lists.waiting_queue.append(_req("urgent"))
                offered = self._admit()
                self.assertEqual(offered is not None, priority_scheduling)

    def test_a_chunked_request_runs(self):
        self.planner.core_lists.chunked_req = _req("c")

        self.assertIsNone(
            self.planner.admission_skip_reason(ScheduleSlot(), self.running)
        )

    def test_remembered_per_pp_slot(self):
        enter_scope(self, published_topology(pp_size=2, pp_max_micro_batch_size=64))
        for req, ring_slot in zip(self.running, (0, 1)):
            req.ring_slot = ring_slot
        slots = [ScheduleSlot(index=0), ScheduleSlot(index=1)]

        self.assertEqual(self._admit(slots[0]), ["w0"])
        self.assertEqual(self._admit(slots[1]), ["w0"])
        self.assertIsNone(self._admit(slots[0]))
        self.assertIsNone(self._admit(slots[1]))
        # Slot 1's request leaves; its request slot frees a step later.
        self.planner.core_lists.decoding.remove(self.running[1])
        self.assertIsNone(self._admit(slots[0]))
        self.assertEqual(self._admit(slots[1]), ["w0"])
        self.free_rows += 1
        self.assertEqual(self._admit(slots[0]), ["w0"])

    def test_remembered_under_priority_preemption(self):
        # Preemption runs only at the request cap, which the running requests
        # and the free request slots in the key decide.
        self.planner._no_kv_room.clear()
        self.planner.enable_priority_preemption = True

        self.assertEqual(self._admit(), ["w0"])
        self.assertIsNone(self._admit())

    def test_hybrid_swa_runs_every_pass(self):
        # Its KV frees as windows slide, with no request leaving.
        self.planner._no_kv_room.clear()
        self.planner.is_hybrid_swa = True

        self.assertEqual(self._admit(), ["w0"])
        self.assertEqual(self._admit(), ["w0"])

    def test_hisparse_runs_every_pass(self):
        # Its requests join decoding as their host copies land.
        self.planner._no_kv_room.clear()
        self.planner.hisparse_coordinator = Mock()

        self.assertEqual(self._admit(), ["w0"])
        self.assertEqual(self._admit(), ["w0"])

    def test_nothing_running_is_not_remembered(self):
        # With nothing running no request release is coming: a failure stays
        # unremembered, so the pass that fits once the pool frees runs.
        self.planner._no_kv_room.clear()
        self.planner.core_lists.decoding = []

        self.assertEqual(self._admit(), ["w0"])
        self.assertEqual(self._admit(), ["w0"])

    def test_prefill_decode_interval_defers_then_admits(self):
        self.planner._no_kv_room.clear()
        self.planner.prefill_decode_interval = 1
        self.planner._prefill_decode_interval_remaining = 1
        self.planner.prefill_decode_interval_min_bs = 0

        self.assertTrue(self.planner.should_defer_prefill(ScheduleSlot()))
        self.assertFalse(self.planner.should_defer_prefill(ScheduleSlot()))
        self.assertEqual(self._admit(), ["w0"])


class TestNoKvRoomMixed(_AdmissionCase):
    """Under mixed chunked prefill the pass reserves the decode part's KV; a
    remembered pass reruns once that reservation shrinks."""

    def setUp(self):
        super().setUp()
        s = self.planner
        s.is_mixed_chunk = True
        s.composition_policy = DecodeFirst()
        s.pending = []
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s.token_to_kv_pool_allocator = SimpleNamespace(page_size=4)
        # Both decode rows sit at a page boundary: each needs a new page.
        self.running = [_req("r0"), _req("r1")]
        s.core_lists.decoding = list(self.running)
        s.core_lists.waiting_queue = [_req("w0")]
        self.assertEqual(self._admit(), ["w0"])
        self.assertEqual(s._no_kv_room[None].decode_kv_tokens, 8)

    def test_skipped_while_the_reservation_holds(self):
        self.assertIsNone(self._admit())

    def test_a_smaller_reservation_reopens(self):
        # A commit moves r1 off the page boundary: its next token needs no
        # new page, so the decode part reserves less.
        self.running[1].kv.kv_allocated_len = 9

        self.assertEqual(self._admit(), ["w0"])


if __name__ == "__main__":
    unittest.main()


class TestNoKvRoomAfterAdmitting(_AdmissionCase):
    """A pass that admits some requests and then finds no KV room for the next
    is remembered with that request as the head the next pass sees."""

    def setUp(self):
        super().setUp()
        for target in ("set_time_batch", "PrefillStats"):
            p = patch(f"sglang.srt.managers.scheduler_components.step_planner.{target}")
            p.start()
            self.addCleanup(p.stop)
        self.planner.step_builder = Mock()
        self.planner.tree_cache.storage_prefetch_retries = None
        _Adder.num_fit = 1
        self.addCleanup(setattr, _Adder, "num_fit", 0)
        self.running = [_req("r0")]
        self.planner.core_lists.decoding = list(self.running)
        self.planner.core_lists.waiting_queue = [_req("w0"), _req("w1")]

    def test_skipped_after_the_launch(self):
        lists = self.planner.core_lists
        self.assertEqual(self._admit(), ["w0", "w1"])
        # The launch: w0 leaves the queue, joins decoding and takes a slot.
        admitted = lists.waiting_queue.pop(0)
        lists.decoding.append(admitted)
        self.free_rows -= 1

        self.assertIsNone(self._admit())
        # The admitted request finishes: its slot frees.
        lists.decoding.remove(admitted)
        self.free_rows += 1
        self.assertEqual(self._admit(), ["w1"])

    def test_a_beam_launch_counts_its_member_rows(self):
        # The admitted beam takes a row for itself and one for its member;
        # once both are taken, a row that frees leaves the head room.
        lists = self.planner.core_lists
        beam = _req("beam")
        beam.beam_group = SimpleNamespace(beam_width=2, member_rows=None)
        lists.waiting_queue = [beam, _req("w1")]
        self.assertEqual(self._admit(), ["beam", "w1"])
        # The launch: the beam joins decoding and spawns its member's row.
        lists.decoding.append(lists.waiting_queue.pop(0))
        beam.beam_group.member_rows = [9]
        self.free_rows -= 2

        self.assertIsNone(self._admit())
        # A request finishes in another slot: its row frees.
        self.free_rows += 1
        self.assertEqual(self._admit(), ["w1"])

    def test_no_record_when_the_last_waiting_request_used_up_the_budget(self):
        # The admission that ends the pass on NO_TOKEN took the only waiting
        # request; no request is left to head the queue.
        _Adder.verdict_after_fit = AddReqResult.NO_TOKEN
        self.addCleanup(setattr, _Adder, "verdict_after_fit", AddReqResult.CONTINUE)
        lists = self.planner.core_lists
        lists.waiting_queue = [_req("w0")]

        self.assertEqual(self._admit(), ["w0"])
        self.assertEqual(self.planner._no_kv_room, {})
