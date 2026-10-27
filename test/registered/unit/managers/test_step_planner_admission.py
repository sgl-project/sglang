"""What one StepPlanner admission pass admits: preemption sizes a request by
its current prefix match, the row budget counts the new request rows the
admitted requests take, not the requests, a spent pass budget stops the pass
before its first candidate, a request HiCache keeps waiting gives back its
match, and a mixed step reserves its decode part before admission."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

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
from sglang.srt.managers.scheduler_components.step_planner import (
    DecodeFirst,
    StepPlanner,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

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
        return_logprob=False,
        input_embeds=None,
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
        time_stats=SimpleNamespace(
            wait_queue_entry_time=0,
            observe_storage_prefetch_ready=Mock(),
            observe_storage_staging_ready=Mock(),
        ),
        kv=SimpleNamespace(
            holds_mamba=False,
            holds_kv=holds_row,
            req_pool_idx=0 if holds_row else None,
            kv_allocated_len=8,
        ),
        init_next_round_input=Mock(),
    )


class _Adder(PrefillAdder):
    """Admits every request offered; preemption is PrefillAdder's own,
    against free_tokens free KV tokens."""

    free_tokens = 0
    # Whether the pass budget is spent before the first candidate.
    spent = False

    def __init__(self, page_size, tree_cache, allocator, running_reqs, *a, **kw):
        # The decode part's forward tokens and KV, reserved ahead of admission.
        self.reserved = (a[3], kw["mixed_decode_kv_tokens"])
        self.running_reqs = running_reqs
        self.new_token_ratio = 1.0
        self.priority_scheduling_preemption_threshold = 0
        self.memory_budget = SimpleNamespace(
            remaining_total=self.free_tokens, total_offset=0
        )
        self.req_to_token_pool = None
        # No request-owned workspace beyond prefill.
        self.token_to_kv_pool_allocator = SimpleNamespace(
            request_slot_reserve=lambda **_: 0
        )
        self.tree_cache = None
        self.hisparse_coordinator = None
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = None
        self.request_retirer = kw["request_retirer"]

    def budget_state(self):
        return AddReqResult.OTHER if self.spent else AddReqResult.CONTINUE

    def add_chunked_req(self, req):
        # The final chunk: it joins and finishes chunking.
        self.can_run_list.append(req)
        return None

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

        self.make_adder = make_adder
        for target, kwargs in (
            ("set_time_batch", {}),
            ("PrefillStats", {}),
        ):
            p = patch(f"{_PLANNER}.{target}", **kwargs)
            p.start()
            self.addCleanup(p.stop)
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
        s.max_running_requests = 64
        s.max_prefill_bs = 0
        s.truncation_align_size = None
        s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        s.policy = SimpleNamespace(
            calc_priority=Mock(),
            enable_priority_scheduling=False,
            shortest_prefill_chunk_limit=lambda *args: None,
        )
        s.req_to_token_pool = SimpleNamespace(available_size=lambda: self.free_rows)
        s.token_to_kv_pool_allocator = None
        s.tree_cache = SimpleNamespace(
            buffer_pipeline=None, storage_prefetch_retries=None
        )
        s.hisparse_coordinator = None
        # Preemption releases a running request through it.
        s.request_retirer = Mock()
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


class TestPreemptionWaitsForHeldRows(_PlannerCase):
    """A preempted request's row is held until the pending step that carries
    it commits; the request it made room for waits for that row instead of
    being admitted against a row that is not free yet."""

    def test_preempted_row_still_held_admits_nothing(self):
        self.free_rows = 0
        self.planner.enable_priority_preemption = True
        _Adder.free_tokens = 10_000
        self.addCleanup(setattr, _Adder, "free_tokens", 0)
        running = _req("run", priority=0, max_new_tokens=50)
        self.planner.core_lists.decoding = [running]
        new = _req("new", priority=1, max_new_tokens=40)
        self.planner.core_lists.waiting_queue = [new]

        adder = self._admit()

        # The retirer held the preempted row: the free rows stay at zero.
        self.assertEqual(adder.preempt_list, [running])
        self.planner.request_retirer.retract.assert_called_once()
        self.assertEqual(adder.can_run_list, [])


class TestHeldRowAdmission(_PlannerCase):
    """A request that already holds a request row (#38118, #35537) and a beam
    candidate's member rows are counted in rows against the free rows."""

    _NUM_ROWS = 4

    def test_final_chunk_leaves_the_free_row_to_a_waiting_request(self):
        for mode in ("null", "prefill"):
            with self.subTest(mode=mode):
                enter_scope(
                    self,
                    published_topology(
                        disaggregation_mode=mode,
                        pp_max_micro_batch_size=self._NUM_ROWS,
                    ),
                )
                running = [_req("r0", holds_row=True), _req("r1", holds_row=True)]
                final_chunk = _req("chunk", holds_row=True)
                waiting = _req("w")
                self.planner.core_lists.decoding = list(running)
                self.planner.core_lists.chunked_req = final_chunk
                self.planner.core_lists.waiting_queue = [waiting]
                # The chunk and the running requests hold rows; one is free.
                self.free_rows = self._NUM_ROWS - len(running) - 1

                adder = self._admit()

                self.assertEqual(adder.can_run_list, [final_chunk, waiting])

    def test_a_session_request_holding_a_row_runs_without_a_free_one(self):
        enter_scope(
            self,
            published_topology(
                pp_max_micro_batch_size=64, enable_streaming_session=True
            ),
        )
        self.planner.core_lists.decoding = [_req("r", holds_row=True)]
        self.planner.core_lists.waiting_queue = [_req("s", holds_row=True)]
        self.free_rows = 0

        adder = self._admit()

        self.assertEqual([r.rid for r in adder.can_run_list], ["s"])

    def test_a_beam_candidate_takes_a_row_per_member(self):
        # With five free rows, a request and a 4-wide beam fit; a third does
        # not.
        beam = _req("beam")
        beam.beam_group = SimpleNamespace(beam_width=4, member_rows=None)
        self.planner.core_lists.waiting_queue = [_req("a"), beam, _req("b")]
        self.free_rows = 5

        adder = self._admit()

        self.assertEqual([r.rid for r in adder.can_run_list], ["a", "beam"])


class TestSpentBudget(_PlannerCase):
    def test_a_chunk_that_spends_the_budget_stops_at_the_head(self):
        # The resumed chunk spends the pass budget: the head is neither
        # matched nor offered, and the pass leaves no KV-room record.
        self.planner.core_lists.decoding = [_req("r", holds_row=True)]
        self.planner.core_lists.chunked_req = _req("chunk", holds_row=True)
        head = _req("w")
        self.planner.core_lists.waiting_queue = [head]
        _Adder.spent = True
        self.addCleanup(setattr, _Adder, "spent", False)

        adder = self._admit()

        self.assertEqual([r.rid for r in adder.can_run_list], ["chunk"])
        head.init_next_round_input.assert_not_called()
        self.assertEqual(self.planner._no_kv_room, {})


class TestHiCacheSkipRevertsTheMatch(_PlannerCase):
    """A request HiCache keeps waiting after its match gives back the mamba
    slot and copy source the match staged, as any request the pass does not
    admit does."""

    def _skipped_req(self):
        mamba_allocator = Mock()
        self.planner.tree_cache = SimpleNamespace(
            buffer_pipeline=None,
            req_to_token_pool=SimpleNamespace(mamba_allocator=mamba_allocator),
            check_prefetch_progress=lambda handle: True,
            pop_prefetch_loaded_span=lambda handle: (0, None),
        )
        req = _req("w")
        req.cache_request_handle = None
        req.kv = SimpleNamespace(
            holds_kv=False,
            kv_allocated_len=8,
            mamba_pool_idx=None,
            mamba_cow_src_index=None,
            mamba_needs_clear=False,
        )

        def match(tree_cache=None):
            # The match copies a cached state into a fresh slot.
            req.kv.mamba_pool_idx = torch.tensor(7)
            req.kv.mamba_cow_src_index = torch.tensor(3)
            req.kv.holds_mamba = True

        req.kv.holds_mamba = False
        req.init_next_round_input.side_effect = match
        self.planner.core_lists.waiting_queue = [req]
        return req, mamba_allocator

    def _assert_reverted(self, req, mamba_allocator, adder):
        self.assertEqual(adder.can_run_list, [])
        req.init_next_round_input.assert_called_once()
        mamba_allocator.free.assert_called_once()
        self.assertEqual(mamba_allocator.free.call_args.args[0].tolist(), [7])
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)

    def test_device_hit_loss_reverts_the_match(self):
        req, mamba_allocator = self._skipped_req()
        self.planner.request_intake.prefetch_after_device_hit_loss.return_value = True

        with patch(f"{_PLANNER}.hicache_storage_enabled", return_value=True):
            adder = self._admit()

        self._assert_reverted(req, mamba_allocator, adder)

    def test_deferred_staged_prefetch_reverts_the_match(self):
        req, mamba_allocator = self._skipped_req()
        self.planner.request_intake.prefetch_after_device_hit_loss.return_value = False
        self.planner.tree_cache.buffer_pipeline = SimpleNamespace(
            prepare_staged_prefetch=lambda req: False
        )

        with patch(f"{_PLANNER}.hicache_storage_enabled", return_value=True):
            adder = self._admit()

        self._assert_reverted(req, mamba_allocator, adder)


class TestMixedStep(_PlannerCase):
    def setUp(self):
        super().setUp()
        s = self.planner
        s.is_mixed_chunk = True
        s.composition_policy = DecodeFirst()
        s.pending = []
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        p = patch(f"{_PLANNER}.check_decode_mem", return_value=True)
        p.start()
        self.addCleanup(p.stop)
        prefill_only = _req("prefill-only", holds_row=True)
        prefill_only.is_prefill_only = True
        self.decode = [_req("d0", holds_row=True), _req("d1", holds_row=True)]
        s.core_lists.decoding = [*self.decode, prefill_only]
        self.waiting = _req("w")
        s.core_lists.waiting_queue = [self.waiting]

    def _mixed_rows(self):
        """The requests of the decode step mixed into the built step."""
        built = self.planner.step_builder
        if not built.build_extend_step.return_value.mix_decode_rows.called:
            return None
        return built.new_decode_step.call_args.args[0]

    def test_the_decode_part_is_reserved_before_admission(self):
        # Its forward tokens are its rows and its KV the tokens they
        # allocate; a prefill-only request decodes no token.
        adder = self._admit()

        self.assertEqual(adder.reserved, (2, 2))
        self.assertEqual(self._mixed_rows(), self.decode)

    def test_a_decode_row_with_logprobs_is_not_reserved(self):
        self.decode[0].return_logprob = True

        adder = self._admit()

        self.assertEqual(adder.reserved, (0, 0))
        self.assertIsNone(self._mixed_rows())

    def test_a_prefill_with_input_embeds_keeps_the_decode_part_out(self):
        self.waiting.input_embeds = [[0.0]]

        adder = self._admit()

        self.assertEqual(adder.reserved, (2, 2))
        self.assertIsNone(self._mixed_rows())


if __name__ == "__main__":
    unittest.main()
