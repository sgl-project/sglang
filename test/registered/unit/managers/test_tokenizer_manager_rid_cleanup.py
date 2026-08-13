"""
Unit tests for rid_to_state cleanup in TokenizerManager.

Verifies that request IDs are properly removed from rid_to_state after
completion or abort, allowing resubmission with the same rid without
triggering "Duplicate request ID detected" errors.

Covers:
  - _handle_abort_req cleans up rid_to_state
  - _handle_batch_output cleans up rid_to_state on finished requests
  - _init_req_state rejects duplicate rids
  - Resubmission succeeds after cleanup
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import msgspec

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    AbortReq,
    BatchStrOutput,
    GenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ReqState,
    TokenizerManager,
)
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    APIServerReqTimeStats,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


_NOT_FINISHED = object()  # Sentinel: request has not finished yet

# ---------------------------------------------------------------------------
# Per-request field defaults for BatchStrOutput construction.
# Categorised by value shape so that _make_batch_str_output can assign
# type-appropriate defaults without hardcoding every field name.
# When a field is renamed upstream, the old name simply won't appear in
# msgspec.structs.fields() and the new name will fall through to the
# pattern-matching or safe fallback — no test breakage.
# ---------------------------------------------------------------------------

_PER_REQUEST_INT_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "retraction_counts",
        # Speculative-decoding int-scalar fields (current and historical names)
        "spec_verify_ct",
        "spec_accepted_drafts",
        "spec_num_correct_drafts",
    }
)

_PER_REQUEST_FLOAT_FIELDS = frozenset(
    {
        "output_token_entropy_val",
    }
)

_PER_REQUEST_NESTED_LIST_FIELDS = frozenset(
    {
        "output_ids",
        # Logprob fields
        "input_token_logprobs_val",
        "input_token_logprobs_idx",
        "output_token_logprobs_val",
        "output_token_logprobs_idx",
        "input_top_logprobs_val",
        "input_top_logprobs_idx",
        "output_top_logprobs_val",
        "output_top_logprobs_idx",
        "input_token_ids_logprobs_val",
        "input_token_ids_logprobs_idx",
        "output_token_ids_logprobs_val",
        "output_token_ids_logprobs_idx",
        # Speculative-decoding histogram fields (current and historical names)
        "spec_acceptance_histogram",
        "spec_correct_drafts_histogram",
    }
)

_PER_REQUEST_OPTIONAL_FIELDS = frozenset(
    {
        "output_hidden_states",
        "routed_experts",
        "indexer_topk",
        "placeholder_tokens_idx",
        "placeholder_tokens_val",
    }
)


def _make_tokenizer_manager() -> TokenizerManager:
    """Create a TokenizerManager with mocked dependencies, bypassing __init__."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm._config_updates = []
    tm.server_args.enable_trace = False
    tm.server_args.enable_metrics = False
    tm.server_args.enable_lora = False
    tm.server_args.speculative_algorithm = None
    tm.server_args.incremental_streaming_output = False
    tm.server_args.skip_tokenizer_init = False
    tm.server_args.batch_notify_size = 1
    tm.server_args.weight_version = "1"
    tm.server_args.crash_dump_folder = ""
    tm.server_args.dp_size = 1
    tm.disaggregation_mode = "none"
    tm.rid_to_state = {}
    tm.enable_metrics = False
    tm.enable_trace = False
    tm.enable_lora = False
    tm.incremental_streaming_output = False
    tm.allow_auto_truncate = False
    tm.skip_tokenizer_init = False
    tm.dump_requests_folder = ""
    tm.crash_dump_folder = ""
    tm.send_to_scheduler = MagicMock()
    return tm


def _make_req_state(rid: str = "test_rid") -> ReqState:
    """Create a minimal ReqState for testing."""
    obj = Mock(spec=GenerateReqInput)
    obj.rid = rid
    obj.stream = False
    obj.return_logprob = False
    obj.lora_path = None
    obj.log_metrics = False
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


def _make_abort_req(rid: str, abort_message: str = "Aborted") -> AbortReq:
    """Create an AbortReq for testing."""
    return AbortReq(
        rid=rid,
        abort_all=False,
        finished_reason={"type": "abort", "message": abort_message},
        abort_message=abort_message,
    )


def _make_batch_obj(rids):
    """Batch request whose obj[i] is stable, mirroring GenerateReqInput's cache.

    The plain helper below hands out a fresh Mock per __getitem__ call, which would
    make identity checks fail for reasons the production code never sees.
    """
    obj = MagicMock(spec=GenerateReqInput)
    obj.rid = list(rids)
    obj.is_single = False
    subs = {}
    for i, rid in enumerate(rids):
        sub = Mock(spec=GenerateReqInput)
        sub.rid = rid
        sub.is_single = True
        subs[i] = sub
    obj.__getitem__.side_effect = subs.__getitem__
    return obj


def _run_background(background_tasks):
    """Run a BackgroundTasks bundle without waiting out its real sleep."""

    async def main():
        with patch("asyncio.sleep", new=AsyncMock()):
            await background_tasks()

    asyncio.run(main())


def _make_batch_str_output(rid: str, finished_reason=None) -> BatchStrOutput:
    """Create a minimal BatchStrOutput for a single request.

    Uses struct field introspection so that new or renamed fields in
    BatchStrOutput don't break this test.  Only the fields that matter for
    test logic (rids, finished_reasons, output_strs) are set explicitly;
    all others receive type-appropriate defaults based on naming patterns.
    Fields with class-level defaults are left alone automatically.
    """
    if finished_reason is _NOT_FINISHED:
        fr = None
    elif finished_reason is None:
        fr = {"type": "length"}
    else:
        fr = finished_reason

    kwargs = {}
    for f in msgspec.structs.fields(BatchStrOutput):
        if f.name == "rids":
            kwargs[f.name] = [rid]
        elif f.name == "finished_reasons":
            kwargs[f.name] = [fr]
        elif f.name == "output_strs":
            kwargs[f.name] = ["hello"]
        elif f.name in _PER_REQUEST_INT_FIELDS:
            kwargs[f.name] = [0]
        elif f.name in _PER_REQUEST_FLOAT_FIELDS:
            kwargs[f.name] = [0.0]
        elif f.name in _PER_REQUEST_NESTED_LIST_FIELDS:
            kwargs[f.name] = [[]]
        elif f.name in _PER_REQUEST_OPTIONAL_FIELDS:
            kwargs[f.name] = [None]
        # Fields with class defaults — skip, let the default be used
        elif (
            f.default is not msgspec.NODEFAULT
            or f.default_factory is not msgspec.NODEFAULT
        ):
            continue
        # Unknown required field — provide a safe per-request default.
        # Most BatchStrOutput fields are per-request lists; [[]] works for
        # List[List[...]] and is unlikely to crash on [i] indexing for
        # List[int] either (the inner [] just means "no data").
        else:
            kwargs[f.name] = [[]]

    return BatchStrOutput(**kwargs)


class TestRidToStateCleanupOnAbort(CustomTestCase):
    """Test that _handle_abort_req removes rid from rid_to_state."""

    def test_abort_removes_rid_from_state(self):
        """After _handle_abort_req, rid should be removed from rid_to_state."""
        tm = _make_tokenizer_manager()
        rid = "abort_test_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertNotIn(rid, tm.rid_to_state)

    def test_abort_allows_resubmit_same_rid(self):
        """After abort, _init_req_state should accept the same rid again."""
        tm = _make_tokenizer_manager()
        rid = "resubmit_after_abort_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        # Resubmit with the same rid — should not raise
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_abort_sets_finished_and_notifies(self):
        """_handle_abort_req should mark state as finished and set the event."""
        tm = _make_tokenizer_manager()
        rid = "abort_notify_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertTrue(state.finished)
        self.assertTrue(state.event.is_set())
        self.assertEqual(len(state.out_list), 1)
        self.assertEqual(
            state.out_list[0]["meta_info"]["finish_reason"]["type"], "abort"
        )


class TestRidToStateCleanupOnBatchOutput(CustomTestCase):
    """Test that _handle_batch_output removes rid from rid_to_state on completion."""

    def test_batch_output_removes_rid_on_finish(self):
        """When a request finishes in _handle_batch_output, rid should be removed."""
        tm = _make_tokenizer_manager()
        rid = "batch_finish_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid)
        asyncio.run(tm._handle_batch_output(batch_output))

        self.assertNotIn(rid, tm.rid_to_state)

    def test_batch_output_allows_resubmit_after_finish(self):
        """After a request finishes, the same rid can be resubmitted."""
        tm = _make_tokenizer_manager()
        rid = "batch_resubmit_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid)
        asyncio.run(tm._handle_batch_output(batch_output))

        # Resubmit with the same rid — should not raise
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_batch_output_keeps_rid_when_not_finished(self):
        """When a request is not yet finished, rid should remain in rid_to_state."""
        tm = _make_tokenizer_manager()
        rid = "batch_ongoing_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        # finished_reason=_NOT_FINISHED means the request is still ongoing
        batch_output = _make_batch_str_output(rid, finished_reason=_NOT_FINISHED)
        asyncio.run(tm._handle_batch_output(batch_output))

        self.assertIn(rid, tm.rid_to_state)


class TestInitReqStateDuplicateDetection(CustomTestCase):
    """Test that _init_req_state raises ValueError for duplicate rids."""

    def test_duplicate_rid_raises_error(self):
        """_init_req_state should raise ValueError if rid already exists."""
        tm = _make_tokenizer_manager()
        rid = "duplicate_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None

        with self.assertRaises(ValueError) as ctx:
            tm._init_req_state(obj)
        self.assertIn("Duplicate request ID", str(ctx.exception))

    def test_unique_rid_succeeds(self):
        """_init_req_state should succeed with a unique rid."""
        tm = _make_tokenizer_manager()
        rid = "unique_rid"

        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None

        tm._init_req_state(obj)
        self.assertIn(rid, tm.rid_to_state)


class TestResubmitAfterCompletion(CustomTestCase):
    """End-to-end test: complete a request, then resubmit with the same rid."""

    def test_complete_then_resubmit_same_rid(self):
        """A request that completes normally should allow resubmission with the same rid."""
        tm = _make_tokenizer_manager()
        rid = "complete_resubmit_rid"

        # Phase 1: simulate a request in rid_to_state, then complete it
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid, finished_reason={"type": "length"})
        asyncio.run(tm._handle_batch_output(batch_output))

        # rid should be cleaned up
        self.assertNotIn(rid, tm.rid_to_state)

        # Phase 2: resubmit with the same rid — should succeed
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_abort_then_resubmit_same_rid(self):
        """An aborted request should allow resubmission with the same rid."""
        tm = _make_tokenizer_manager()
        rid = "abort_resubmit_rid"

        # Phase 1: simulate a request, then abort it
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertNotIn(rid, tm.rid_to_state)

        # Phase 2: resubmit with the same rid — should succeed
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)


class _DummyAsyncCM:
    """Reusable no-op async context manager (stands in for an RW lock)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _make_tm_for_generate() -> TokenizerManager:
    """Augment the mocked TokenizerManager with what generate_request needs."""
    tm = _make_tokenizer_manager()
    tm.server_args.language_only = False
    tm.server_args.tokenizer_worker_num = 1
    tm.server_args.enable_strict_thinking = False
    tm.auto_create_handle_loop = Mock()
    tm._set_default_priority = Mock()
    tm.request_logger = Mock()
    tm.tokenizer = None
    tm.is_pause = False
    tm.is_pause_cond = asyncio.Condition()
    tm.model_update_lock = Mock()
    tm.model_update_lock.reader_lock = _DummyAsyncCM()
    tm._validate_and_resolve_lora = AsyncMock(return_value=None)
    return tm


def _make_generate_obj(rid, is_single):
    obj = MagicMock(spec=GenerateReqInput)
    obj.routed_dp_rank = None
    obj.is_single = is_single
    obj.rid = rid
    obj.received_time = 0.0
    obj.external_trace_header = None
    obj.bootstrap_room = None
    obj.max_thinking_tokens = None
    obj.normalize_batch_and_arguments = Mock()
    if not is_single:
        obj.__getitem__.side_effect = lambda i: Mock()
    return obj


class TestDiscardPendingReqStates(CustomTestCase):
    """Direct tests for _discard_pending_req_states."""

    def test_discard_single(self):
        tm = _make_tokenizer_manager()
        rid = "d_single"
        tm.rid_to_state[rid] = _make_req_state(rid)
        obj = Mock(spec=GenerateReqInput)
        obj.is_single = True
        obj.rid = rid
        tm._discard_pending_req_states(obj)
        self.assertNotIn(rid, tm.rid_to_state)

    def test_discard_batch_removes_all(self):
        tm = _make_tokenizer_manager()
        rids = ["d0", "d1", "d2"]
        for r in rids:
            tm.rid_to_state[r] = _make_req_state(r)
        obj = Mock(spec=GenerateReqInput)
        obj.is_single = False
        obj.rid = list(rids)
        tm._discard_pending_req_states(obj)
        for r in rids:
            self.assertNotIn(r, tm.rid_to_state)

    def test_discard_ignores_already_removed(self):
        """Popping a rid that is no longer present must not raise."""
        tm = _make_tokenizer_manager()
        tm.rid_to_state["p1"] = _make_req_state("p1")
        obj = Mock(spec=GenerateReqInput)
        obj.is_single = False
        obj.rid = ["p1", "already_gone"]
        tm._discard_pending_req_states(obj)  # must not raise
        self.assertNotIn("p1", tm.rid_to_state)


class TestParallelStreamTaskCleanup(CustomTestCase):
    def test_failing_choice_cancels_and_closes_sibling_waiters(self):
        tm = _make_tokenizer_manager()

        async def drive():
            sibling_closed = asyncio.Event()

            async def failing_choice():
                await asyncio.sleep(0)
                raise RuntimeError("choice failed")
                yield  # pragma: no cover

            async def blocked_choice():
                try:
                    await asyncio.Event().wait()
                    yield  # pragma: no cover
                finally:
                    sibling_closed.set()

            stream = tm._stream_batch_responses(
                [failing_choice(), blocked_choice()],
                ["choice-0", "choice-1"],
            )
            with self.assertRaisesRegex(RuntimeError, "choice failed"):
                await stream.__anext__()
            self.assertTrue(sibling_closed.is_set())

        asyncio.run(drive())

    def test_failing_non_stream_choice_cancels_and_closes_sibling_waiters(self):
        tm = _make_tokenizer_manager()

        async def drive():
            sibling_closed = asyncio.Event()

            async def failing_choice():
                await asyncio.sleep(0)
                raise RuntimeError("choice failed")
                yield  # pragma: no cover

            async def blocked_choice():
                try:
                    await asyncio.Event().wait()
                    yield  # pragma: no cover
                finally:
                    sibling_closed.set()

            with self.assertRaisesRegex(RuntimeError, "choice failed"):
                await tm._collect_batch_responses([failing_choice(), blocked_choice()])
            self.assertTrue(sibling_closed.is_set())

        asyncio.run(drive())


class TestGenerateRequestCleanupOnDispatchFailure(CustomTestCase):
    """generate_request must not leak rid_to_state when dispatch fails.

    Regression guard: _init_req_state creates rid_to_state entries up front,
    and the only remover is the scheduler-response path. A failure before the
    request reaches the scheduler (e.g. input-length validation rejecting an
    over-context request) used to leak those entries permanently.
    """

    def test_single_failure_before_dispatch_cleans_up(self):
        tm = _make_tm_for_generate()
        rid = "single_overlen"
        obj = _make_generate_obj(rid, is_single=True)
        # Simulate over-length rejection during tokenization/validation.
        tm._tokenize_one_request = AsyncMock(side_effect=ValueError("input too long"))
        tm._send_one_request = Mock()

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaises(ValueError):
            asyncio.run(drive())

        # Got past _init_req_state (which created the entry) ...
        tm._tokenize_one_request.assert_awaited_once()
        tm._send_one_request.assert_not_called()
        # ... and the entry was cleaned up rather than leaked.
        self.assertNotIn(rid, tm.rid_to_state)

    def test_batch_failure_before_dispatch_cleans_up_all(self):
        tm = _make_tm_for_generate()
        rids = ["b0", "b1", "b2"]
        obj = _make_generate_obj(list(rids), is_single=False)

        # One over-length sub-request makes the whole batch dispatch raise.
        async def _boom(*args, **kwargs):
            raise ValueError("input too long")
            yield  # pragma: no cover  (marks this an async generator)

        tm._handle_batch_request = _boom

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaises(ValueError):
            asyncio.run(drive())

        # All sub-request entries created by _init_req_state are cleaned up.
        for r in rids:
            self.assertNotIn(r, tm.rid_to_state)

    def test_thinking_budget_rejects_runtime_without_strict_thinking(self):
        tm = _make_tm_for_generate()
        obj = GenerateReqInput(
            text="hello",
            rid="thinking-budget",
            sampling_params={},
            max_thinking_tokens=32,
        )

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaisesRegex(ValueError, "--enable-strict-thinking"):
            asyncio.run(drive())

        self.assertFalse(tm.rid_to_state)


class TestDiscardKeepsDispatchedStates(CustomTestCase):
    """A dispatched request must survive teardown so its abort can still fire.

    `generate_request` discards pending states on any BaseException, and a client
    disconnect raises CancelledError *after* the request reached the scheduler.
    Dropping the state there strands the request: the delayed `create_abort_task`
    looks the rid up in rid_to_state, finds nothing, and never tells the scheduler
    to stop -- so the GPU keeps decoding for a client that already left.
    """

    def test_undispatched_state_is_discarded(self):
        tm = _make_tokenizer_manager()
        obj = Mock(spec=GenerateReqInput)
        obj.rid = "never-sent"
        obj.is_single = True
        tm.rid_to_state["never-sent"] = _make_req_state("never-sent")

        tm._discard_pending_req_states(obj)

        self.assertNotIn("never-sent", tm.rid_to_state)

    def test_dispatched_state_is_kept(self):
        tm = _make_tokenizer_manager()
        obj = Mock(spec=GenerateReqInput)
        obj.rid = "in-flight"
        obj.is_single = True
        state = _make_req_state("in-flight")
        state.dispatched = True
        tm.rid_to_state["in-flight"] = state

        tm._discard_pending_req_states(obj)

        self.assertIn("in-flight", tm.rid_to_state)

    def test_batch_discards_only_undispatched(self):
        tm = _make_tokenizer_manager()
        obj = Mock(spec=GenerateReqInput)
        obj.rid = ["sent", "unsent"]
        obj.is_single = False
        sent = _make_req_state("sent")
        sent.dispatched = True
        tm.rid_to_state["sent"] = sent
        tm.rid_to_state["unsent"] = _make_req_state("unsent")

        tm._discard_pending_req_states(obj)

        self.assertIn("sent", tm.rid_to_state)
        self.assertNotIn("unsent", tm.rid_to_state)

    def test_kept_state_lets_the_delayed_abort_reach_the_scheduler(self):
        """End state of the disconnect path: abort_request must now dispatch."""
        tm = _make_tokenizer_manager()
        tm.server_args.tokenizer_worker_num = 1
        tm.tokenizer_ipc_name = None
        tm._dispatch_to_scheduler = Mock()
        obj = Mock(spec=GenerateReqInput)
        obj.rid = "disconnected"
        obj.is_single = True
        state = _make_req_state("disconnected")
        state.dispatched = True
        tm.rid_to_state["disconnected"] = state

        tm._discard_pending_req_states(obj)
        tm.abort_request(rid="disconnected")

        tm._dispatch_to_scheduler.assert_called_once()
        sent = tm._dispatch_to_scheduler.call_args[0][0]
        self.assertIsInstance(sent, AbortReq)
        self.assertEqual(sent.rid, "disconnected")

    def test_completed_request_does_not_trigger_a_late_abort(self):
        """A finished request left no state, so the delayed abort stays a no-op.

        This only covers the rid being *gone*. The harder case -- the rid having
        been handed to a different request in the meantime -- is covered by
        TestDelayedAbortTargetsOneRequest below.
        """
        tm = _make_tokenizer_manager()
        tm.server_args.tokenizer_worker_num = 1
        tm.tokenizer_ipc_name = None
        tm._dispatch_to_scheduler = Mock()
        self.assertNotIn("finished-rid", tm.rid_to_state)

        tm.abort_request(rid="finished-rid")

        tm._dispatch_to_scheduler.assert_not_called()


class TestDelayedAbortTargetsOneRequest(CustomTestCase):
    """The delayed abort must not kill a newer request that reused the rid.

    `_init_req_state` rejects a duplicate rid only while the previous state is
    alive, so a rid becomes reusable the moment its request finishes. Between that
    moment and the 2s `create_abort_task` firing there is a window where the rid
    belongs to somebody else; matching on the rid alone would abort that innocent
    request. `expect_obj` pins each delayed abort to the request that scheduled it.
    """

    @staticmethod
    def _make_tm():
        tm = _make_tokenizer_manager()
        tm.server_args.tokenizer_worker_num = 1
        tm.tokenizer_ipc_name = None
        tm._dispatch_to_scheduler = Mock()
        return tm

    @staticmethod
    def _make_obj(rid, is_single=True):
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = is_single
        return obj

    @staticmethod
    def _place_state(tm, rid, obj):
        """Register a dispatched state owned by *obj*, as _init_req_state would."""
        state = _make_req_state(rid)
        state.obj = obj
        state.dispatched = True
        tm.rid_to_state[rid] = state
        return state

    def test_reused_rid_is_not_aborted_by_the_previous_request(self):
        """Request A finished, B took its rid, then A's delayed abort fired."""
        tm = self._make_tm()
        rid = "shared-rid"
        obj_a = self._make_obj(rid)
        obj_b = self._make_obj(rid)
        self._place_state(tm, rid, obj_b)  # A's state is long gone; B owns the rid

        tm.abort_request(rid, expect_obj=obj_a)

        tm._dispatch_to_scheduler.assert_not_called()
        self.assertIs(tm.rid_to_state[rid].obj, obj_b, "B must be left running")

    def test_delayed_abort_still_fires_for_its_own_request(self):
        """The disconnect path must keep working: same object, so abort proceeds."""
        tm = self._make_tm()
        rid = "disconnected"
        obj = self._make_obj(rid)
        self._place_state(tm, rid, obj)

        tm.abort_request(rid, expect_obj=obj)

        tm._dispatch_to_scheduler.assert_called_once()
        self.assertEqual(tm._dispatch_to_scheduler.call_args[0][0].rid, rid)

    def test_explicit_abort_endpoint_is_unaffected(self):
        """/abort_request passes no expectation and must abort whoever holds the rid."""
        tm = self._make_tm()
        rid = "shared-rid"
        self._place_state(tm, rid, self._make_obj(rid))

        tm.abort_request(rid)

        tm._dispatch_to_scheduler.assert_called_once()

    def test_reused_rid_is_protected_with_multiple_tokenizer_workers(self):
        """The identity check must not be limited to tokenizer_worker_num == 1.

        Multi-tokenizer mode runs N Granian worker processes, each with its own
        rid_to_state. A delayed abort is a background task of the response that
        scheduled it, so it always runs in the process that owns the state --
        the lookup is just as trustworthy there as with a single worker.
        """
        tm = self._make_tm()
        tm.server_args.tokenizer_worker_num = 2
        rid = "shared-rid"
        obj_a = self._make_obj(rid)
        obj_b = self._make_obj(rid)
        self._place_state(tm, rid, obj_b)

        tm.abort_request(rid, expect_obj=obj_a)

        tm._dispatch_to_scheduler.assert_not_called()
        self.assertIs(tm.rid_to_state[rid].obj, obj_b, "B must be left running")

    def test_multi_worker_explicit_abort_still_dispatches_on_local_miss(self):
        """The reverse: /abort_request may land on a worker that never saw the rid.

        With several tokenizer workers a local miss proves nothing, so the abort
        must still go out -- otherwise the endpoint silently stops working.
        """
        tm = self._make_tm()
        tm.server_args.tokenizer_worker_num = 2
        self.assertNotIn("elsewhere", tm.rid_to_state)

        tm.abort_request("elsewhere")

        tm._dispatch_to_scheduler.assert_called_once()

    def test_create_abort_task_pins_the_single_request(self):
        tm = self._make_tm()
        obj = self._make_obj("solo")
        tm.abort_request = Mock()

        _run_background(tm.create_abort_task(obj))

        tm.abort_request.assert_called_once_with("solo", expect_obj=obj)

    def test_create_abort_task_pins_each_batch_entry(self):
        tm = self._make_tm()
        obj = _make_batch_obj(["r0", "r1"])
        tm.abort_request = Mock()

        _run_background(tm.create_abort_task(obj))

        self.assertEqual(
            tm.abort_request.call_args_list,
            [call("r0", expect_obj=obj[0]), call("r1", expect_obj=obj[1])],
        )

    def test_sub_object_identity_is_stable(self):
        """Guards the invariant the batch path relies on: obj[i] is cached.

        If upstream ever stops caching sub-objects, `state.obj is expect_obj` would
        never match for batches and every batched disconnect would silently stop
        being aborted -- the exact bug this file exists to prevent.
        """
        obj = GenerateReqInput(text=["a", "b"], rid=["r0", "r1"])
        obj.normalize_batch_and_arguments()

        self.assertIs(obj[0], obj[0])
        self.assertIsNot(obj[0], obj[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
