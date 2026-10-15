"""What one StepPlanner admission pass admits: preemption sizes a request by
its current prefix match."""

import unittest
from array import array
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
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler_components.engine_state import (
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_PLANNER = "sglang.srt.managers.scheduler_components.step_planner"


class _Req(SimpleNamespace):
    # Hashed by identity, as Req is.
    __hash__ = object.__hash__

    def finished(self):
        return False


def _req(rid, *, priority=0, max_new_tokens=8, holds_row=False):
    return _Req(
        rid=rid,
        ring_slot=None,
        priority=priority,
        is_prefill_only=False,
        beam_group=None,
        token_indices_to_pool=None,
        lora_id=None,
        output_ids=[],
        seqlen=8,
        prefix_len=0,
        full_untruncated_fill_ids=array("q"),
        sampling_params=SimpleNamespace(
            max_new_tokens=max_new_tokens, ignore_eos=False
        ),
        time_stats=SimpleNamespace(wait_queue_entry_time=0),
        kv=SimpleNamespace(holds_mamba=False, holds_kv=holds_row),
        init_next_round_input=Mock(),
    )


class _Adder(PrefillAdder):
    """Admits every request offered while free_tokens is unspent; preemption
    is PrefillAdder's own, against free_tokens."""

    free_tokens = 0

    def __init__(self, page_size, tree_cache, allocator, running_reqs, *a, **kw):
        self.running_reqs = running_reqs
        self.new_token_ratio = 1.0
        self.priority_scheduling_preemption_threshold = 0
        self.memory_budget = SimpleNamespace(
            remaining_total=self.free_tokens, total_offset=0
        )
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None
        self.tree_cache = None
        self.hisparse_coordinator = None
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = None

    def add_one_req(self, req, **kwargs):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _PlannerCase(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology(pp_max_micro_batch_size=64))
        self.free_rows = 4
        self.adders = []

        def make_adder(*args, **kwargs):
            adder = _Adder(*args, **kwargs)
            self.adders.append(adder)
            return adder

        for target, kwargs in (
            ("PrefillAdder", dict(side_effect=make_adder)),
            ("set_time_batch", {}),
            ("PrefillStats", {}),
        ):
            p = patch(f"{_PLANNER}.{target}", **kwargs)
            p.start()
            self.addCleanup(p.stop)
        # Preemption releases a running request through this function.
        p = patch("sglang.srt.managers.schedule_policy.release_req")
        p.start()
        self.addCleanup(p.stop)
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
        s.max_running_requests = 64
        s.max_prefill_bs = 0
        s.truncation_align_size = None
        s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        s.policy = SimpleNamespace(
            calc_priority=Mock(), enable_priority_scheduling=False
        )
        s.req_to_token_pool = SimpleNamespace(available_size=lambda: self.free_rows)
        s.token_to_kv_pool_allocator = None
        s.tree_cache = SimpleNamespace(
            buffer_pipeline=None, storage_prefetch_retries=None
        )
        s.hisparse_coordinator = None
        s._no_kv_room = {}
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.step_builder = Mock()
        s.request_intake = Mock()
        s.beam_coordinator = BeamCoordinator(
            model_config=None,
            spec_algorithm=None,
            max_req_len=0,
            req_to_token_pool=s.req_to_token_pool,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            future_map=None,
            core_lists=s.core_lists,
        )
        return s

    def _admit(self):
        """Run one admission pass; return the adder it used, or None when it
        was skipped before the adder."""
        num_adders = len(self.adders)
        self.planner._get_new_batch_prefill_raw(
            ScheduleSlot(), prefill_delayer_single_pass=None
        )
        return self.adders[-1] if len(self.adders) > num_adders else None


class TestPreemptionSizesTheMatchedRequest(_PlannerCase):
    """At the request cap, a higher-priority request preempts only when its
    matched input and output fit what the preempted requests free."""

    def _verdict(self, matched_before: bool):
        # The pool is full and its last row is held by one running request
        # that frees 50 tokens; the candidate needs 1000 input tokens plus 40
        # output tokens.
        self.free_rows = 0
        self.planner.enable_priority_preemption = True
        running = _req("run", priority=0, max_new_tokens=50)
        self.planner.core_lists.decoding = [running]
        new = _req("new", priority=1, max_new_tokens=40)

        def match(tree_cache=None):
            new.full_untruncated_fill_ids = array("q", range(1000))
            new.prefix_len = 0

        new.init_next_round_input.side_effect = match
        if matched_before:
            match()
        self.planner.core_lists.waiting_queue = [new]

        adder = self._admit()
        return adder.preempt_list, adder.can_run_list

    def test_verdict_does_not_depend_on_an_earlier_match(self):
        for matched_before in (True, False):
            with self.subTest(matched_before=matched_before):
                preempted, admitted = self._verdict(matched_before)
                self.assertEqual(preempted, [])
                self.assertEqual(admitted, [])


if __name__ == "__main__":
    unittest.main()
