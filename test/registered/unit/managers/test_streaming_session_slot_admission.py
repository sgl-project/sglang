import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ReqPool:
    def __init__(self, available: int):
        self._available = available

    def available_size(self) -> int:
        return self._available


class _ScanReq:
    def __init__(self, rid: str, session=None):
        self.rid = rid
        self.session = session
        self.to_finish = None
        self.kv = ReqKvInfo()
        self.beam_group = None
        self.lora_id = None
        self.init_calls = 0
        self.time_stats = SimpleNamespace(set_wait_queue_entry_time=lambda: None)

    def init_next_round_input(self, _tree_cache):
        self.init_calls += 1
        if self.session is not None:
            self.kv.req_pool_idx = self.rid


class _FakePrefillAdder:
    def __init__(self, *_args, **_kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None

    def add_one_req(self, req, **_kwargs):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return req

    def preempt_to_schedule(self, _req):
        return False


class _FakeNewBatch:
    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.contains_last_prefill_chunk = None
        self.prefill_stats = None
        self.decoding_reqs = None
        self.prepare_for_extend_calls = 0

    def prepare_for_extend(self):
        self.prepare_for_extend_calls += 1


class _ChunkedReq(_ScanReq):
    def __init__(self, rid):
        super().__init__(rid)
        self.kv.req_pool_idx = rid
        self.inflight_middle_chunks = 0
        self.extend_range = SimpleNamespace(length=1, end=1)

    def init_next_round_input(self):
        self.init_calls += 1


class _FallbackSessionReq(_ScanReq):
    """A retained slot that falls back to a fresh row during prefix matching."""

    def __init__(self, rid, req_pool):
        super().__init__(rid, session=SimpleNamespace(session_id="a", streaming=True))
        self.req_pool = req_pool

    def init_next_round_input(self, _tree_cache):
        self.init_calls += 1
        self.req_pool._available += 1


@contextmanager
def _patch_scan_dependencies(pp_max_micro_batch_size=4):
    memory = SimpleNamespace(
        enable_flexkv=False,
        hicache_host_memory_mode="write_through",
    )
    patches = (
        (scheduler_module, "get_memory", lambda: memory),
        (
            scheduler_module,
            "get_parallel",
            lambda: SimpleNamespace(pp_max_micro_batch_size=pp_max_micro_batch_size),
        ),
        (
            scheduler_module,
            "get_schedule",
            lambda: SimpleNamespace(prefill_max_requests=None),
        ),
        (scheduler_module, "PrefillAdder", _FakePrefillAdder),
        (scheduler_module, "set_time_batch", lambda *_args: None),
        (
            scheduler_module.ScheduleBatch,
            "init_new",
            staticmethod(lambda reqs, *_args, **_kwargs: _FakeNewBatch(reqs)),
        ),
        (
            scheduler_module.PrefillStats,
            "from_adder",
            staticmethod(lambda *_args, **_kwargs: object()),
        ),
    )
    with ExitStack() as stack:
        for target, name, value in patches:
            stack.enter_context(patch.object(target, name, value))
        yield


def _make_scheduler(
    waiting_queue, available=0, pending_beam_rows=0, reusable_session_ids=None
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.req_to_token_pool = _ReqPool(available=available)
    scheduler.ps = SimpleNamespace(pp_size=1)
    reusable_session_ids = (
        {"a"} if reusable_session_ids is None else reusable_session_ids
    )
    scheduler.tree_cache = SimpleNamespace(
        supports_streaming_session=lambda: True,
        has_reusable_streaming_session_slot=lambda session_id: (
            session_id in reusable_session_ids
        ),
    )
    scheduler.beam_coordinator = SimpleNamespace(
        pending_member_rows=lambda _batch: pending_beam_rows
    )
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_streaming_session = True
    scheduler.enable_priority_preemption = False
    scheduler.is_hybrid_swa = False
    scheduler.chunked_req = None
    scheduler.waiting_queue = list(waiting_queue)
    scheduler.min_free_slots_delayer = None
    scheduler.policy = SimpleNamespace(calc_priority=lambda *_args: None)
    scheduler.chunked_prefill_size = None
    scheduler.enable_dynamic_chunking = False
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=object(),
            prefill_aware_swa=False,
        )
    )
    scheduler.page_size = 1
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
    scheduler.max_prefill_tokens = 1024
    scheduler.is_mixed_chunk = False
    scheduler.priority_scheduling_preemption_threshold = 0
    scheduler.max_prefill_bs = 4
    scheduler.max_running_requests = 4
    scheduler.dllm_config = None
    scheduler.enable_lora = False
    scheduler.enable_hicache_storage = False
    scheduler.truncation_align_size = None
    scheduler.model_config = object()
    scheduler.enable_overlap = False
    scheduler.spec_algorithm = object()
    scheduler.enable_priority_scheduling = False
    scheduler.load_inquirer = SimpleNamespace(
        _get_num_pending_tokens=lambda **_kwargs: 0
    )
    return scheduler


def _running_batch(reqs=None):
    return SimpleNamespace(
        batch_is_full=False,
        reqs=[] if reqs is None else list(reqs),
        is_empty=lambda: not reqs,
        is_prefill_only=False,
    )


class TestStreamingSessionSlotAdmission(CustomTestCase):
    def test_non_streaming_server_keeps_full_pool_early_out(self):
        """A full ordinary server must not scan its waiting queue every iteration."""
        ordinary = _ScanReq("ordinary")
        priority_calls = []

        with _patch_scan_dependencies():
            scheduler = _make_scheduler([ordinary])
            scheduler.enable_streaming_session = False
            scheduler.policy = SimpleNamespace(
                calc_priority=lambda *_args: priority_calls.append(True)
            )
            running_batch = _running_batch()
            new_batch, returned_running_batch = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNone(new_batch)
        self.assertIs(returned_running_batch, running_batch)
        self.assertTrue(running_batch.batch_is_full)
        self.assertEqual(priority_calls, [])
        self.assertEqual(ordinary.init_calls, 0)

    def test_blocked_fresh_head_does_not_hide_reusable_session_turn(self):
        """A full request pool must still admit a turn that owns its session slot."""
        ordinary = _ScanReq("ordinary")
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)

        with _patch_scan_dependencies():
            scheduler = _make_scheduler([ordinary, turn])
            running_batch = _running_batch()
            new_batch, returned_running_batch = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNotNone(new_batch)
        self.assertEqual(new_batch.reqs, [turn])
        self.assertIs(returned_running_batch, running_batch)
        self.assertEqual(scheduler.waiting_queue, [ordinary])
        self.assertEqual(ordinary.init_calls, 0)
        self.assertEqual(turn.init_calls, 1)

    def test_noncontinue_retained_turn_preserves_fresh_fifo(self):
        """Stopping after a retained turn must not reorder blocked fresh requests."""
        first = _ScanReq("first")
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)
        second = _ScanReq("second")

        def add_and_stop(adder, req, **_kwargs):
            adder.can_run_list.append(req)
            return AddReqResult.OTHER

        with (
            _patch_scan_dependencies(),
            patch.object(_FakePrefillAdder, "add_one_req", add_and_stop),
        ):
            scheduler = _make_scheduler([first, turn, second])
            running_batch = _running_batch()
            new_batch, returned_running_batch = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNotNone(new_batch)
        self.assertEqual(new_batch.reqs, [turn])
        self.assertIs(returned_running_batch, running_batch)
        self.assertFalse(running_batch.batch_is_full)
        self.assertEqual(scheduler.waiting_queue, [first, second])
        self.assertEqual(first.init_calls, 0)
        self.assertEqual(turn.init_calls, 1)
        self.assertEqual(second.init_calls, 0)

    def test_hicache_deferred_reusable_turn_is_retried(self):
        """A temporary HiCache wait must not permanently latch slot admission full."""
        ordinary = _ScanReq("ordinary")
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)
        ready = {"value": False}
        checks = []

        def check_prefetch_progress(rid):
            checks.append(rid)
            return ready["value"]

        with _patch_scan_dependencies():
            scheduler = _make_scheduler([ordinary, turn])
            scheduler.enable_hicache_storage = True
            scheduler.tree_cache.check_prefetch_progress = check_prefetch_progress
            scheduler.tree_cache.pop_prefetch_loaded_tokens = lambda _rid: 0
            running_batch = _running_batch()

            first_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )
            full_after_first_pass = running_batch.batch_is_full
            ready["value"] = True
            second_batch, returned_running_batch = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNone(first_batch)
        self.assertFalse(full_after_first_pass)
        self.assertIsNotNone(second_batch)
        self.assertEqual(second_batch.reqs, [turn])
        self.assertIs(returned_running_batch, running_batch)
        self.assertEqual(scheduler.waiting_queue, [ordinary])
        self.assertEqual(checks, ["a2", "a2"])
        self.assertEqual(ordinary.init_calls, 0)
        self.assertEqual(turn.init_calls, 1)

    def test_hierarchical_cache_no_token_does_not_latch_reusable_turn(self):
        """An empty HiCache batch intentionally retries after temporary NO_TOKEN."""
        ordinary = _ScanReq("ordinary")
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)
        ready = {"value": False}

        def add_one_req(adder, req, **_kwargs):
            if not ready["value"]:
                return AddReqResult.NO_TOKEN
            adder.can_run_list.append(req)
            return AddReqResult.CONTINUE

        with (
            _patch_scan_dependencies(),
            patch.object(_FakePrefillAdder, "add_one_req", add_one_req),
        ):
            scheduler = _make_scheduler([ordinary, turn])
            scheduler.enable_hierarchical_cache = True
            scheduler.tree_cache.check_hicache_events = lambda: None
            scheduler.tree_cache.ready_to_load_host_cache = lambda: None
            running_batch = _running_batch()

            first_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )
            full_after_first_pass = running_batch.batch_is_full
            ready["value"] = True
            second_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNone(first_batch)
        self.assertFalse(full_after_first_pass)
        self.assertIsNotNone(second_batch)
        self.assertEqual(second_batch.reqs, [turn])
        self.assertEqual(scheduler.waiting_queue, [ordinary])
        self.assertEqual(turn.init_calls, 2)

    def test_lora_deferred_reusable_turn_is_retried(self):
        """A temporary LoRA wait must not permanently latch slot admission full."""
        ordinary = _ScanReq("ordinary")
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)
        ready = {"value": False}
        checks = []

        def can_schedule_lora(req, _running_loras):
            checks.append(req.rid)
            return req is ordinary or ready["value"]

        with _patch_scan_dependencies():
            scheduler = _make_scheduler([ordinary, turn])
            scheduler.enable_lora = True
            scheduler.lora_drainer = None
            scheduler._can_schedule_lora_req = can_schedule_lora
            running_batch = _running_batch()

            first_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )
            full_after_first_pass = running_batch.batch_is_full
            ready["value"] = True
            second_batch, returned_running_batch = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNone(first_batch)
        self.assertFalse(full_after_first_pass)
        self.assertIsNotNone(second_batch)
        self.assertEqual(second_batch.reqs, [turn])
        self.assertIs(returned_running_batch, running_batch)
        self.assertEqual(scheduler.waiting_queue, [ordinary])
        self.assertEqual(checks, ["ordinary", "a2", "ordinary", "a2"])
        self.assertEqual(ordinary.init_calls, 0)
        self.assertEqual(turn.init_calls, 1)

    def test_full_fresh_lora_queue_latches_after_one_scan(self):
        """A full queue with no reusable turn must not spin on LoRA deferral."""
        ordinary = _ScanReq("ordinary")
        checks = []

        def cannot_schedule_lora(req, _running_loras):
            checks.append(req.rid)
            return False

        with _patch_scan_dependencies():
            scheduler = _make_scheduler([ordinary])
            scheduler.enable_lora = True
            scheduler.lora_drainer = None
            scheduler._can_schedule_lora_req = cannot_schedule_lora
            running_batch = _running_batch()
            first_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )
            second_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNone(first_batch)
        self.assertIsNone(second_batch)
        self.assertTrue(running_batch.batch_is_full)
        self.assertEqual(checks, ["ordinary"])

    def test_min_free_slots_delay_preserves_global_prefill_cadence(self):
        """Retained slots must not bypass an enabled global prefill delay."""
        ordinary = _ScanReq("ordinary")
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)
        calls = []

        def should_delay(**kwargs):
            calls.append(kwargs)
            return True

        with _patch_scan_dependencies():
            scheduler = _make_scheduler([ordinary, turn], available=1)
            scheduler.min_free_slots_delayer = SimpleNamespace(
                should_delay=should_delay
            )
            running_batch = _running_batch([object()])
            new_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNone(new_batch)
        self.assertEqual(calls, [{"running_bs": 1, "num_allocatable_reqs": 1}])
        self.assertEqual(scheduler.waiting_queue, [ordinary, turn])
        self.assertEqual(ordinary.init_calls, 0)
        self.assertEqual(turn.init_calls, 0)

    def test_beam_slot_budget_does_not_hide_reusable_session_turn(self):
        """Beam row reservations must not hide a later retained session turn."""
        beam = _ScanReq("beam")
        beam.beam_group = SimpleNamespace(beam_width=2)
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)

        with _patch_scan_dependencies():
            scheduler = _make_scheduler([beam, turn], available=2, pending_beam_rows=1)
            running_batch = _running_batch()
            new_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNotNone(new_batch)
        self.assertEqual(new_batch.reqs, [turn])
        self.assertEqual(scheduler.waiting_queue, [beam])

    def test_priority_preemption_is_tried_before_skipping_fresh_request(self):
        """A blocked fresh request must retain the configured preemption chance."""
        ordinary = _ScanReq("ordinary")
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)
        calls = []

        def preempt_to_schedule(_adder, req):
            calls.append(req.rid)
            return False

        with (
            _patch_scan_dependencies(),
            patch.object(_FakePrefillAdder, "preempt_to_schedule", preempt_to_schedule),
        ):
            scheduler = _make_scheduler([ordinary, turn])
            scheduler.enable_priority_preemption = True
            running_batch = _running_batch()
            new_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertEqual(calls, ["ordinary"])
        self.assertIsNotNone(new_batch)
        self.assertEqual(new_batch.reqs, [turn])

    def test_pipeline_limit_still_blocks_reusable_session_turn(self):
        """A retained row must not bypass pipeline microbatch capacity."""
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)

        with _patch_scan_dependencies(pp_max_micro_batch_size=1):
            scheduler = _make_scheduler([turn], available=8)
            running_batch = _running_batch([object()])
            new_batch, returned_running_batch = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNone(new_batch)
        self.assertIs(returned_running_batch, running_batch)
        self.assertTrue(running_batch.batch_is_full)
        self.assertEqual(turn.init_calls, 0)

    def test_disaggregated_prefill_counts_only_fresh_rows(self):
        """Retained turns must not consume the disaggregated fresh-row budget."""
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)
        ordinary = _ScanReq("ordinary")
        blocked = _ScanReq("blocked")

        with _patch_scan_dependencies():
            scheduler = _make_scheduler([turn, ordinary, blocked], available=1)
            scheduler.disaggregation_mode = DisaggregationMode.PREFILL
            running_batch = _running_batch()
            new_batch, returned_running_batch = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNotNone(new_batch)
        self.assertEqual(new_batch.reqs, [turn, ordinary])
        self.assertIs(returned_running_batch, running_batch)
        self.assertEqual(scheduler.waiting_queue, [blocked])
        self.assertEqual(turn.init_calls, 1)
        self.assertEqual(ordinary.init_calls, 1)
        self.assertEqual(blocked.init_calls, 0)

    def test_full_batch_wakes_only_for_enqueued_reusable_turn(self):
        """Only a runnable retained-row arrival should wake a row-full batch."""
        ordinary = _ScanReq("ordinary")
        ordinary_2 = _ScanReq("ordinary-2")
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)
        scheduler = _make_scheduler([ordinary])
        scheduler.running_batch = _running_batch()
        scheduler._set_or_validate_priority = lambda _req: True
        scheduler._abort_on_queued_limit = lambda _req: False
        scheduler._prefetch_kvcache = lambda _req: None

        with _patch_scan_dependencies():
            first_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=scheduler.running_batch,
            )

        self.assertIsNone(first_batch)
        self.assertTrue(scheduler.running_batch.batch_is_full)
        scheduler._add_request_to_queue(ordinary_2)
        self.assertTrue(scheduler.running_batch.batch_is_full)
        scheduler._add_request_to_queue(turn)

        self.assertFalse(scheduler.running_batch.batch_is_full)
        self.assertEqual(scheduler.waiting_queue, [ordinary, ordinary_2, turn])
        with _patch_scan_dependencies():
            new_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=scheduler.running_batch,
            )
        self.assertIsNotNone(new_batch)
        self.assertEqual(new_batch.reqs, [turn])
        self.assertEqual(scheduler.waiting_queue, [ordinary, ordinary_2])

    def test_released_request_slot_wakes_full_batch(self):
        """A real row release must wake a batch latched full on row capacity."""
        scheduler = _make_scheduler([])
        scheduler.running_batch = _running_batch()
        scheduler.running_batch.batch_is_full = True

        scheduler._wake_batch_if_req_slot_released(available_before=0)
        self.assertTrue(scheduler.running_batch.batch_is_full)
        scheduler.req_to_token_pool._available = 1
        scheduler._wake_batch_if_req_slot_released(available_before=0)

        self.assertFalse(scheduler.running_batch.batch_is_full)

    def test_released_request_slot_wakes_all_pipeline_microbatches(self):
        """A row release must wake every full pipeline microbatch."""
        scheduler = _make_scheduler([])
        scheduler.ps.pp_size = 2
        first = _running_batch()
        second = _running_batch()
        first.batch_is_full = True
        second.batch_is_full = True
        scheduler.running_batch = first
        scheduler.running_mbs = [first, second]
        scheduler.req_to_token_pool._available = 1

        scheduler._wake_batch_if_req_slot_released(available_before=0)

        self.assertFalse(first.batch_is_full)
        self.assertFalse(second.batch_is_full)

    def test_pipeline_wakeup_is_safe_before_loop_initialization(self):
        """A control-plane wake before PP loop entry must not access missing state."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.flush_cache = lambda: True
        scheduler.is_fully_idle = lambda: True
        scheduler.ipc_channels = SimpleNamespace()
        scheduler.tree_cache = SimpleNamespace()
        scheduler.init_running_status()
        scheduler.ps = SimpleNamespace(pp_size=2)
        scheduler.running_batch.batch_is_full = True

        scheduler._wake_full_prefill_batches()

        self.assertFalse(scheduler.running_batch.batch_is_full)
        self.assertEqual(scheduler.running_mbs, [])

    def test_only_live_streaming_session_slot_is_reusable(self):
        """Only unfinished streaming turns with a retained row may bypass the limit."""
        scheduler = _make_scheduler([], reusable_session_ids={"live"})
        cases = (
            (_ScanReq("ordinary"), False),
            (
                _ScanReq(
                    "non-streaming",
                    session=SimpleNamespace(session_id="live", streaming=False),
                ),
                False,
            ),
            (
                _ScanReq(
                    "missing",
                    session=SimpleNamespace(session_id="missing", streaming=True),
                ),
                False,
            ),
            (
                _ScanReq(
                    "live", session=SimpleNamespace(session_id="live", streaming=True)
                ),
                True,
            ),
        )
        prefinished = _ScanReq(
            "prefinished",
            session=SimpleNamespace(session_id="live", streaming=True),
        )
        prefinished.to_finish = object()

        for req, expected in (*cases, (prefinished, False)):
            with self.subTest(rid=req.rid):
                self.assertEqual(
                    scheduler._waiting_req_reuses_streaming_session_slot(req),
                    expected,
                )

    def test_chunked_request_owned_slot_does_not_hide_reusable_turn(self):
        """A chunked request's owned row must not hide a retained session turn."""
        chunked = _ChunkedReq("chunked")
        session = SimpleNamespace(session_id="a", streaming=True)
        turn = _ScanReq("a2", session=session)

        with _patch_scan_dependencies():
            scheduler = _make_scheduler([turn])
            scheduler.chunked_req = chunked
            running_batch = _running_batch()
            new_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNotNone(new_batch)
        self.assertEqual(new_batch.reqs, [chunked, turn])
        self.assertEqual(chunked.init_calls, 1)
        self.assertEqual(turn.init_calls, 1)

    def test_chunked_request_owned_slot_does_not_consume_fresh_capacity(self):
        """A chunked request's owned row must not consume a newly free row."""
        chunked = _ChunkedReq("chunked")
        ordinary = _ScanReq("ordinary")

        with _patch_scan_dependencies():
            scheduler = _make_scheduler([ordinary], available=1)
            scheduler.chunked_req = chunked
            running_batch = _running_batch()
            new_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNotNone(new_batch)
        self.assertEqual(new_batch.reqs, [chunked, ordinary])
        self.assertEqual(scheduler.waiting_queue, [])

    def test_session_fallback_to_fresh_row_is_charged_to_slot_budget(self):
        """A retained turn that loses its row must consume fresh-row capacity."""
        ordinary = _ScanReq("ordinary")

        with _patch_scan_dependencies():
            scheduler = _make_scheduler([])
            fallback = _FallbackSessionReq("a2", scheduler.req_to_token_pool)
            scheduler.waiting_queue = [fallback, ordinary]
            running_batch = _running_batch()
            new_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNotNone(new_batch)
        self.assertEqual(new_batch.reqs, [fallback])
        self.assertEqual(scheduler.waiting_queue, [ordinary])
        self.assertEqual(scheduler.req_to_token_pool.available_size(), 1)


if __name__ == "__main__":
    unittest.main()
