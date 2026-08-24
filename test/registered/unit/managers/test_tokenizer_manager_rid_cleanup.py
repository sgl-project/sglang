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
  - Every cleanup path hands the request's LoRA reference back to the registry
"""

import asyncio
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import msgspec

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import AbortReq, BatchStrOutput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

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
    tm.request_logger = Mock()
    tm.request_metrics_exporter_manager = Mock()
    tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
    tm.send_to_scheduler = MagicMock()
    tm._lora_release_tasks = set()
    return tm


def _make_req_state(rid: str = "test_rid") -> ReqState:
    """Create a minimal ReqState for testing."""
    obj = Mock(spec=GenerateReqInput)
    obj.rid = rid
    obj.stream = False
    obj.return_logprob = False
    obj.lora_path = None
    obj.lora_id = None
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

    def test_single_tokenizer_held_abort_is_returned_without_key_error(self):
        tm = _make_tm_for_generate()
        rid = "single-held-abort"
        obj = _make_generate_obj(rid, is_single=True)
        obj.return_prompt_token_ids = False
        obj.return_logprob = False
        obj.stream = False

        async def tokenize(req):
            tm.rid_to_state[req.rid].abort_before_dispatch = True
            return SimpleNamespace(rid=req.rid)

        tm._tokenize_one_request = tokenize

        async def drive():
            return [out async for out in tm.generate_request(obj)]

        outputs = asyncio.run(drive())
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["meta_info"]["finish_reason"]["type"], "abort")
        self.assertNotIn(rid, tm.rid_to_state)

    def test_batch_tokenizer_held_aborts_keep_detached_states_for_waiters(self):
        tm = _make_tokenizer_manager()
        obj = GenerateReqInput(text=["a", "b"], rid=["a", "b"])
        obj.received_time = 0.0
        obj.normalize_batch_and_arguments()
        tm._init_req_state(obj)
        for state in tm.rid_to_state.values():
            state.abort_before_dispatch = True

        tokenized_objs = [SimpleNamespace(rid=rid) for rid in obj.rid]
        tm._should_use_batch_tokenization = Mock(return_value=True)
        tm._batch_tokenize_and_process = AsyncMock(return_value=tokenized_objs)

        async def drive():
            return [out async for out in tm._handle_batch_request(obj)]

        outputs = asyncio.run(drive())
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0]), 2)
        self.assertTrue(
            all(
                out["meta_info"]["finish_reason"]["type"] == "abort"
                for out in outputs[0]
            )
        )
        self.assertEqual(tm.rid_to_state, {})


class _RecordingLoRARegistry:
    """Stands in for LoRARegistry, recording acquired and released IDs."""

    def __init__(self):
        self.acquired = []
        self.released = []

    async def get_unregistered_loras(self, lora_paths):
        return []

    async def acquire(self, lora_paths):
        if isinstance(lora_paths, list):
            lora_ids = [
                f"id:{path}" if path is not None else None for path in lora_paths
            ]
            self.acquired.extend(lora_id for lora_id in lora_ids if lora_id is not None)
            return lora_ids

        lora_id = f"id:{lora_paths}"
        self.acquired.append(lora_id)
        return lora_id

    async def release(self, lora_id):
        if isinstance(lora_id, list):
            self.released.extend(item for item in lora_id if item is not None)
        else:
            self.released.append(lora_id)


def _enable_lora(tm: TokenizerManager) -> TokenizerManager:
    tm.enable_lora = True
    tm.lora_registry = _RecordingLoRARegistry()
    return tm


def _make_lora_req_state(rid: str, lora_id: str = "lora-1") -> ReqState:
    state = _make_req_state(rid)
    state.obj.lora_path = "adapter"
    state.obj.lora_id = lora_id
    return state


class TestLoraReferenceRelease(CustomTestCase):
    """Every path that drops a request must release its LoRA reference.

    The registry counts in-flight requests per adapter and
    /unload_lora_adapter waits for that count to reach zero, so a single
    reference that is never handed back wedges every later adapter swap.
    """

    def test_abort_releases_reference(self):
        tm = _enable_lora(_make_tokenizer_manager())
        rid = "lora_abort_rid"
        tm.rid_to_state[rid] = _make_lora_req_state(rid)

        async def drive():
            tm._handle_abort_req(_make_abort_req(rid))
            await asyncio.sleep(0)

        asyncio.run(drive())

        self.assertNotIn(rid, tm.rid_to_state)
        self.assertEqual(tm.lora_registry.released, ["lora-1"])

    def test_bad_request_abort_releases_reference(self):
        tm = _enable_lora(_make_tokenizer_manager())
        rid = "lora_bad_request_rid"
        state = _make_lora_req_state(rid)
        tm.rid_to_state[rid] = state
        out = {
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "status_code": HTTPStatus.BAD_REQUEST,
                    "message": "rejected by the scheduler",
                }
            }
        }

        async def drive():
            with self.assertRaises(ValueError):
                await tm._handle_abort_finish_reason(out, state, is_stream=False)
            await asyncio.sleep(0)

        asyncio.run(drive())

        self.assertNotIn(rid, tm.rid_to_state)
        self.assertEqual(tm.lora_registry.released, ["lora-1"])

    def test_dispatch_failure_releases_reference(self):
        tm = _enable_lora(_make_tm_for_generate())
        rid = "lora_overlen_rid"
        obj = _make_generate_obj(rid, is_single=True)
        obj.lora_path = "adapter"
        obj.lora_id = "lora-1"
        tm._tokenize_one_request = AsyncMock(side_effect=ValueError("input too long"))
        tm._send_one_request = Mock()

        async def drive():
            with self.assertRaises(ValueError):
                await tm.generate_request(obj).__anext__()
            await asyncio.sleep(0)

        asyncio.run(drive())

        self.assertNotIn(rid, tm.rid_to_state)
        self.assertEqual(tm.lora_registry.released, ["lora-1"])

    def test_partial_batch_keeps_dispatched_sibling_until_terminal_ack(self):
        """A later sibling failure must not release an in-flight adapter."""

        async def drive():
            tm = _enable_lora(_make_tokenizer_manager())
            active = _make_lora_req_state("active", "id:active")
            active.dispatched_to_scheduler = True
            pending = _make_lora_req_state("pending", "id:pending")
            tm.rid_to_state = {"active": active, "pending": pending}
            obj = SimpleNamespace(is_single=False, rid=["active", "pending"])

            tm._discard_pending_req_states(obj)
            await asyncio.gather(*list(tm._lora_release_tasks))

            self.assertIn("active", tm.rid_to_state)
            self.assertNotIn("pending", tm.rid_to_state)
            self.assertEqual(tm.lora_registry.released, ["id:pending"])

            tm._handle_abort_req(_make_abort_req("active"))
            await asyncio.gather(*list(tm._lora_release_tasks))
            self.assertEqual(tm.lora_registry.released, ["id:pending", "id:active"])

        asyncio.run(drive())

    def test_reference_released_once_when_cleanup_paths_race(self):
        """A finish and a late abort echo for the same rid release only once."""
        tm = _enable_lora(_make_tokenizer_manager())
        rid = "lora_race_rid"
        tm.rid_to_state[rid] = _make_lora_req_state(rid)

        async def drive():
            await tm._handle_batch_output(_make_batch_str_output(rid))
            tm._handle_abort_req(_make_abort_req(rid))
            await asyncio.sleep(0)

        asyncio.run(drive())

        self.assertEqual(tm.lora_registry.released, ["lora-1"])

    def test_no_release_without_lora(self):
        tm = _enable_lora(_make_tokenizer_manager())
        rid = "no_lora_rid"
        tm.rid_to_state[rid] = _make_req_state(rid)

        async def drive():
            tm._handle_abort_req(_make_abort_req(rid))
            await asyncio.sleep(0)

        asyncio.run(drive())

        self.assertEqual(tm.lora_registry.released, [])

    def test_no_release_when_lora_acquire_never_succeeded(self):
        tm = _enable_lora(_make_tokenizer_manager())
        rid = "unacquired_lora_rid"
        state = _make_req_state(rid)
        state.obj.lora_path = "missing-adapter"
        state.obj.lora_id = None
        tm.rid_to_state[rid] = state

        async def drive():
            tm._handle_abort_req(_make_abort_req(rid))
            await asyncio.sleep(0)

        asyncio.run(drive())

        self.assertEqual(tm.lora_registry.released, [])

    def test_explicit_rids_release_every_parallel_acquire_on_failure(self):
        """One parent rid can own n references until parallel fan-out."""

        async def drive():
            tm = _enable_lora(_make_tokenizer_manager())
            tm.server_args.max_loaded_loras = None
            obj = GenerateReqInput(
                text=["hello", "world"],
                rid=["parent-a", "parent-b"],
                lora_path=["adapter-a", "adapter-b"],
                sampling_params={"n": 3, "max_new_tokens": 8},
            )
            obj.received_time = 0.0
            obj.normalize_batch_and_arguments()
            tm._init_req_state(obj)
            await tm._resolve_lora_path(obj)

            # Simulate tokenization/validation failing before child states are
            # created. Cleanup must balance all batch_size * n acquisitions.
            tm._discard_pending_req_states(obj)
            await asyncio.gather(*list(tm._lora_release_tasks))

            self.assertCountEqual(
                tm.lora_registry.released,
                tm.lora_registry.acquired,
            )
            self.assertEqual(len(tm.lora_registry.released), 6)

        asyncio.run(drive())

    def test_abort_during_lora_acquire_releases_orphaned_reference(self):
        """An abort racing acquire cannot strand the newly acquired ID."""

        async def drive():
            tm = _enable_lora(_make_tokenizer_manager())
            tm.server_args.max_loaded_loras = None
            obj = GenerateReqInput(
                text="hello",
                rid="acquire-race",
                lora_path="adapter",
            )
            obj.received_time = 0.0
            obj.normalize_batch_and_arguments()
            tm._init_req_state(obj)

            acquire_started = asyncio.Event()
            finish_acquire = asyncio.Event()
            original_acquire = tm.lora_registry.acquire

            async def blocked_acquire(lora_path):
                acquire_started.set()
                await finish_acquire.wait()
                return await original_acquire(lora_path)

            tm.lora_registry.acquire = blocked_acquire
            resolve_task = asyncio.create_task(tm._resolve_lora_path(obj))
            await acquire_started.wait()
            tm._handle_abort_req(_make_abort_req(obj.rid))
            finish_acquire.set()

            with self.assertRaisesRegex(ValueError, "aborted while resolving"):
                await resolve_task

            self.assertEqual(tm.lora_registry.acquired, ["id:adapter"])
            self.assertEqual(tm.lora_registry.released, ["id:adapter"])

        asyncio.run(drive())

    def test_cancellation_during_lora_acquire_releases_eventual_reference(self):
        """Cancellation cannot strand a counter increment hidden in acquire."""

        async def drive():
            tm = _enable_lora(_make_tokenizer_manager())
            tm.server_args.max_loaded_loras = None
            obj = GenerateReqInput(
                text="hello", rid="cancel-acquire", lora_path="adapter"
            )
            obj.received_time = 0.0
            obj.normalize_batch_and_arguments()
            tm._init_req_state(obj)

            acquire_started = asyncio.Event()
            finish_acquire = asyncio.Event()
            original_acquire = tm.lora_registry.acquire

            async def blocked_acquire(lora_path):
                acquire_started.set()
                await finish_acquire.wait()
                return await original_acquire(lora_path)

            tm.lora_registry.acquire = blocked_acquire
            resolve_task = asyncio.create_task(tm._resolve_lora_path(obj))
            await acquire_started.wait()
            resolve_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await resolve_task

            finish_acquire.set()
            while tm._lora_release_tasks:
                await asyncio.gather(*list(tm._lora_release_tasks))
            self.assertEqual(tm.lora_registry.acquired, ["id:adapter"])
            self.assertEqual(tm.lora_registry.released, ["id:adapter"])

        asyncio.run(drive())

    def test_parallel_warmup_cancellation_defers_release_until_ack(self):
        """The residual bundle protects an already-dispatched warm-up."""

        async def drive():
            tm = _enable_lora(_make_tokenizer_manager())
            tm.server_args.max_loaded_loras = None
            tm._dispatch_to_scheduler = Mock()
            obj = GenerateReqInput(
                text=["hello"],
                rid=["parallel-parent"],
                lora_path=["adapter"],
                sampling_params={"n": 3, "max_new_tokens": 8},
            )
            obj.received_time = 0.0
            obj.normalize_batch_and_arguments()
            tm._init_req_state(obj)
            await tm._resolve_lora_path(obj)

            warmup_dispatched = asyncio.Event()
            warmup_rid = None

            async def tokenize(req):
                return SimpleNamespace(
                    input_ids=[1],
                    mm_inputs=None,
                    rid=req.rid,
                    sampling_params=SimpleNamespace(max_new_tokens=8),
                    stream=False,
                )

            def send(tokenized):
                nonlocal warmup_rid
                state = tm.rid_to_state[tokenized.rid]
                state.dispatched_to_scheduler = True
                warmup_rid = tokenized.rid
                warmup_dispatched.set()

            async def wait_forever(req, request=None, detached_state=None):
                await asyncio.Event().wait()
                yield  # pragma: no cover

            tm._tokenize_one_request = tokenize
            tm._send_one_request = send
            tm._wait_one_response = wait_forever

            next_output = asyncio.create_task(tm._handle_batch_request(obj).__anext__())
            await warmup_dispatched.wait()
            next_output.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await next_output

            await asyncio.sleep(0)
            self.assertEqual(tm.lora_registry.released, [])
            self.assertIn(warmup_rid, tm.rid_to_state)
            abort_rids = [
                call.args[0].rid
                for call in tm._dispatch_to_scheduler.call_args_list
                if isinstance(call.args[0], AbortReq)
            ]
            self.assertIn(warmup_rid, abort_rids)

            tm._handle_abort_req(_make_abort_req(warmup_rid))
            while tm._lora_release_tasks:
                await asyncio.gather(*list(tm._lora_release_tasks))
            self.assertEqual(len(tm.lora_registry.released), 3)
            self.assertEqual(tm.rid_to_state, {})
            self.assertEqual(tm._parallel_request_groups, {})
            self.assertEqual(tm._parallel_request_group_aliases, {})

        asyncio.run(drive())

    def test_parent_prefix_abort_reaches_parallel_warmup_and_stops_fanout(self):
        """#30912 parent-prefix abort semantics survive regenerated child rids."""

        async def drive():
            tm = _enable_lora(_make_tokenizer_manager())
            tm.server_args.max_loaded_loras = None
            tm.server_args.tokenizer_worker_num = 1
            tm._dispatch_to_scheduler = Mock()
            obj = GenerateReqInput(
                text=["hello"],
                rid=["parallel-parent"],
                lora_path=["adapter"],
                sampling_params={"n": 3, "max_new_tokens": 8},
            )
            obj.received_time = 0.0
            obj.normalize_batch_and_arguments()
            tm._init_req_state(obj)
            await tm._resolve_lora_path(obj)

            warmup_dispatched = asyncio.Event()
            dispatched_rids = []

            async def tokenize(req):
                return SimpleNamespace(
                    input_ids=[1],
                    mm_inputs=None,
                    rid=req.rid,
                    sampling_params=SimpleNamespace(max_new_tokens=8),
                    stream=False,
                )

            def send(tokenized):
                state = tm.rid_to_state[tokenized.rid]
                state.dispatched_to_scheduler = True
                dispatched_rids.append(tokenized.rid)
                warmup_dispatched.set()

            async def wait_for_abort(req, request=None, detached_state=None):
                state = detached_state or tm.rid_to_state[req.rid]
                await state.event.wait()
                yield state.out_list[-1]

            tm._tokenize_one_request = tokenize
            tm._send_one_request = send
            tm._wait_one_response = wait_for_abort

            next_output = asyncio.create_task(tm._handle_batch_request(obj).__anext__())
            await warmup_dispatched.wait()
            await asyncio.sleep(0)
            warmup_rid = dispatched_rids[0]

            tm.abort_request("parallel-parent", prefix=True)
            sent_aborts = [
                call.args[0]
                for call in tm._dispatch_to_scheduler.call_args_list
                if isinstance(call.args[0], AbortReq)
            ]
            self.assertTrue(
                any(
                    abort.abort_all
                    or warmup_rid.startswith(abort.rid)
                    or abort.rid == warmup_rid
                    for abort in sent_aborts
                )
            )
            self.assertEqual(tm.lora_registry.released, [])

            tm._handle_abort_req(_make_abort_req(warmup_rid))
            outputs = await next_output
            while tm._lora_release_tasks:
                await asyncio.gather(*list(tm._lora_release_tasks))

            self.assertEqual(len(outputs), 3)
            self.assertTrue(
                all(
                    out["meta_info"]["finish_reason"]["type"] == "abort"
                    for out in outputs
                )
            )
            self.assertEqual(dispatched_rids, [warmup_rid])
            self.assertEqual(len(tm.lora_registry.released), 3)
            self.assertEqual(tm.rid_to_state, {})
            self.assertEqual(tm._parallel_request_groups, {})
            self.assertEqual(tm._parallel_request_group_aliases, {})

        asyncio.run(drive())

    def test_parallel_abort_isolated_to_matching_prompt_group(self):
        """Aborting prompt A must not cancel prompt B in the same n>1 batch."""

        async def drive():
            tm = _enable_lora(_make_tokenizer_manager())
            tm.server_args.max_loaded_loras = None
            tm.server_args.tokenizer_worker_num = 1
            tm._dispatch_to_scheduler = Mock()
            obj = GenerateReqInput(
                text=["hello-a", "hello-b"],
                rid=["parent-a", "parent-b"],
                lora_path=["adapter-a", "adapter-b"],
                sampling_params={"n": 2, "max_new_tokens": 8},
            )
            obj.received_time = 0.0
            obj.normalize_batch_and_arguments()
            tm._init_req_state(obj)
            await tm._resolve_lora_path(obj)

            dispatched = []
            release_before_a_warmup_ack = None
            release_after_a_warmup_ack = None
            b_release_counts_before_child_terminal = []
            aborted_a = False

            async def tokenize(req):
                return SimpleNamespace(
                    input_ids=[1],
                    mm_inputs=None,
                    rid=req.rid,
                    sampling_params=SimpleNamespace(max_new_tokens=8),
                    stream=False,
                )

            def send(tokenized):
                nonlocal aborted_a, release_before_a_warmup_ack
                state = tm.rid_to_state[tokenized.rid]
                state.dispatched_to_scheduler = True
                dispatched.append(
                    (state.request_group, state.obj.lora_id, tokenized.rid)
                )
                if state.request_group == "parent-a" and not aborted_a:
                    aborted_a = True
                    tm.abort_request("parent-a")
                    release_before_a_warmup_ack = list(tm.lora_registry.released)
                    tm._handle_abort_req(_make_abort_req(tokenized.rid))

            async def finish(req, request=None, detached_state=None):
                nonlocal release_after_a_warmup_ack
                state = detached_state or tm.rid_to_state[req.rid]
                if state.finished:
                    yield state.out_list[-1]
                    return

                if state.request_group == "parent-b" and state.obj.lora_id is None:
                    # Let A's bundle-release task run after its scheduler ACK.
                    await asyncio.sleep(0)
                    release_after_a_warmup_ack = list(tm.lora_registry.released)
                elif state.request_group == "parent-b":
                    b_release_counts_before_child_terminal.append(
                        tm.lora_registry.released.count("id:adapter-b")
                    )

                tm.rid_to_state.pop(req.rid, None)
                state.finished = True
                tm._finalize_lora_lease(state)
                tm._complete_parallel_group_member(state, req.rid)
                yield {
                    "text": "ok",
                    "output_ids": [],
                    "meta_info": {"id": req.rid, "finish_reason": {"type": "length"}},
                }

            tm._tokenize_one_request = tokenize
            tm._send_one_request = send
            tm._wait_one_response = finish

            outputs = [out async for out in tm._handle_batch_request(obj)]
            while tm._lora_release_tasks:
                await asyncio.gather(*list(tm._lora_release_tasks))

            self.assertEqual(len(outputs), 1)
            self.assertEqual(len(outputs[0]), 4)
            self.assertTrue(
                all(
                    out["meta_info"]["finish_reason"]["type"] == "abort"
                    for out in outputs[0][:2]
                )
            )
            self.assertTrue(
                all(
                    out["meta_info"]["finish_reason"]["type"] == "length"
                    for out in outputs[0][2:]
                )
            )

            a_dispatches = [item for item in dispatched if item[0] == "parent-a"]
            b_dispatches = [item for item in dispatched if item[0] == "parent-b"]
            self.assertEqual(len(a_dispatches), 1)
            self.assertIsNone(a_dispatches[0][1])
            self.assertEqual(len(b_dispatches), 3)
            self.assertEqual(sum(item[1] is not None for item in b_dispatches), 2)

            sent_aborts = [
                call.args[0]
                for call in tm._dispatch_to_scheduler.call_args_list
                if isinstance(call.args[0], AbortReq)
            ]
            b_rids = {item[2] for item in b_dispatches}
            self.assertFalse(
                any(abort.abort_all or abort.rid == "parent-b" for abort in sent_aborts)
            )
            self.assertTrue(all(abort.rid not in b_rids for abort in sent_aborts))

            self.assertEqual(release_before_a_warmup_ack, [])
            self.assertEqual(
                release_after_a_warmup_ack, ["id:adapter-a", "id:adapter-a"]
            )
            self.assertEqual(b_release_counts_before_child_terminal, [0, 0])
            self.assertCountEqual(
                tm.lora_registry.released,
                [
                    "id:adapter-a",
                    "id:adapter-a",
                    "id:adapter-b",
                    "id:adapter-b",
                ],
            )
            self.assertEqual(tm.rid_to_state, {})
            self.assertEqual(tm._parallel_request_groups, {})
            self.assertEqual(tm._parallel_request_group_aliases, {})

        asyncio.run(drive())

    def test_parallel_kth_child_failure_keeps_dispatched_children_owned(self):
        """Only never-dispatched children and residual leases release early."""

        async def drive():
            tm = _enable_lora(_make_tokenizer_manager())
            tm.server_args.max_loaded_loras = None
            tm._dispatch_to_scheduler = Mock()
            obj = GenerateReqInput(
                text=["hello"],
                rid=["parallel-parent"],
                lora_path=["adapter"],
                sampling_params={"n": 3, "max_new_tokens": 8},
            )
            obj.received_time = 0.0
            obj.normalize_batch_and_arguments()
            tm._init_req_state(obj)
            await tm._resolve_lora_path(obj)

            send_count = 0
            successful_child_rids = []

            async def tokenize(req):
                return SimpleNamespace(
                    input_ids=[1],
                    mm_inputs=None,
                    rid=req.rid,
                    sampling_params=SimpleNamespace(max_new_tokens=8),
                    stream=False,
                )

            def send(tokenized):
                nonlocal send_count
                send_count += 1
                if send_count == 4:  # warm-up + two children + failing child
                    raise RuntimeError("third child dispatch failed")
                state = tm.rid_to_state[tokenized.rid]
                state.dispatched_to_scheduler = True
                if send_count > 1:
                    successful_child_rids.append(tokenized.rid)

            async def finish_warmup(req, request=None, detached_state=None):
                state = tm.rid_to_state.pop(req.rid)
                tm._finalize_lora_lease(state)
                tm._complete_parallel_group_member(state, req.rid)
                yield {"meta_info": {"id": req.rid}}

            tm._tokenize_one_request = tokenize
            tm._send_one_request = send
            tm._wait_one_response = finish_warmup

            with self.assertRaisesRegex(RuntimeError, "third child"):
                await tm._handle_batch_request(obj).__anext__()
            while tm._lora_release_tasks:
                await asyncio.gather(*list(tm._lora_release_tasks))

            self.assertEqual(len(successful_child_rids), 2)
            self.assertEqual(len(tm.lora_registry.released), 1)
            self.assertTrue(
                all(rid in tm.rid_to_state for rid in successful_child_rids)
            )

            tm._handle_abort_req(_make_abort_req(successful_child_rids[0]))
            while tm._lora_release_tasks:
                await asyncio.gather(*list(tm._lora_release_tasks))
            self.assertEqual(len(tm.lora_registry.released), 2)
            self.assertTrue(tm._parallel_request_groups)

            tm._handle_abort_req(_make_abort_req(successful_child_rids[1]))
            while tm._lora_release_tasks:
                await asyncio.gather(*list(tm._lora_release_tasks))
            self.assertEqual(len(tm.lora_registry.released), 3)
            self.assertEqual(tm.rid_to_state, {})
            self.assertEqual(tm._parallel_request_groups, {})
            self.assertEqual(tm._parallel_request_group_aliases, {})

        asyncio.run(drive())

    def test_parallel_sampling_releases_only_real_samples(self):
        """The prefix-cache warm-up and parent states do not own references."""

        async def drive(text, rid, lora_path, lora_ids):
            tm = _enable_lora(_make_tokenizer_manager())
            obj = GenerateReqInput(
                text=text,
                rid=rid,
                lora_path=lora_path,
                sampling_params={"n": 4, "max_new_tokens": 8},
            )
            obj.received_time = 0.0
            obj.normalize_batch_and_arguments()
            tm._init_req_state(obj)

            # Mirror _resolve_lora_path after acquiring four references.
            obj.lora_id = lora_ids
            for i, sub_obj in obj.__dict__.get("_sub_obj_cache", {}).items():
                sub_obj.lora_id = obj.lora_id[i]
            missing_rids, orphaned_lora_ids = tm._assign_lora_release_ownership(obj)
            self.assertEqual(missing_rids, [])
            self.assertEqual(orphaned_lora_ids, [])

            async def tokenize(req):
                return SimpleNamespace(
                    input_ids=[1],
                    mm_inputs=None,
                    rid=req.rid,
                    sampling_params=SimpleNamespace(max_new_tokens=8),
                    stream=False,
                )

            async def finish(req, request=None, detached_state=None):
                state = tm.rid_to_state.pop(req.rid)
                tm._finalize_lora_lease(state)
                tm._complete_parallel_group_member(state, req.rid)
                yield {"meta_info": {"id": req.rid}}

            tm._tokenize_one_request = tokenize
            tm._send_one_request = Mock()
            tm._wait_one_response = finish

            outputs = [out async for out in tm._handle_batch_request(obj)]
            await asyncio.sleep(0)

            self.assertEqual(len(outputs), 1)
            self.assertCountEqual(
                tm.lora_registry.released,
                lora_ids,
            )
            self.assertEqual(tm.rid_to_state, {})

        # Both accepted rid shapes exercise different parent-state counts.
        asyncio.run(
            drive(
                "hello",
                None,
                "adapter",
                ["stable-upsert-id"] * 4,
            )
        )
        asyncio.run(
            drive(
                ["hello", "world"],
                ["parallel-a", "parallel-b"],
                ["adapter-a", "adapter-b"],
                ["stable-a", "stable-b"] * 4,
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
