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
  - Handler failures clean up pending and dispatched requests
  - _handle_batch_output takes non-streaming first_token_time from the scheduler
"""

import asyncio
import itertools
import pickle
import sys
import time
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import msgspec
import numpy as np
from fastapi import HTTPException
from starlette.requests import Request

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402
import zmq.asyncio  # noqa: E402
from zmq.utils.monitor import recv_monitor_message  # noqa: E402

from sglang.srt.disaggregation.mode import DisaggregationMode  # noqa: E402
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry  # noqa: E402
from sglang.srt.managers import io_struct, mm_utils  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    AbortReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenizedGenerateReqInput,
    EmbeddingReqInput,
    GenerateReqInput,
    RequestOutputStats,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.req_types import (  # noqa: E402
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ReqState,
    TokenizerManager,
    _add_sampling_mask_to_meta_info,
)
from sglang.srt.multimodal.modality import Modality  # noqa: E402
from sglang.srt.multimodal.transport.cuda_ipc import (  # noqa: E402
    CudaIpcTensorTransportProxy,
)
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    APIServerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.observability.request_logger import RequestLogger  # noqa: E402
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_mask import SamplingMaskChunk  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.utils.aio_rwlock import RWLock  # noqa: E402
from sglang.srt.utils.weight_versions import WeightVersionSpan  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestSamplingParamsTruncation(CustomTestCase):
    def test_batch_truncation_is_independent_of_prompt_order(self):
        """A long prompt must not truncate the completion budget of shorter peers."""
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.max_req_len = 100
        manager.max_req_input_len = 94
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = True
        manager.validate_total_tokens = True
        manager.is_generation = True
        manager.model_config = SimpleNamespace(
            joint_head_config=None, pointer_head=None
        )

        for lengths in ((90, 10), (10, 90)):
            for per_prompt in (False, True):
                with self.subTest(lengths=lengths, per_prompt=per_prompt):
                    shared = {"max_new_tokens": 20}
                    req = GenerateReqInput(
                        input_ids=[[1] * length for length in lengths],
                        sampling_params=[shared, shared] if per_prompt else shared,
                    )
                    req.normalize_batch_and_arguments()
                    for i in range(req.batch_size):
                        manager._validate_one_request(req[i], req[i].input_ids)

                    self.assertEqual(
                        [req[i].sampling_params["max_new_tokens"] for i in range(2)],
                        [min(20, 100 - length) for length in lengths],
                    )
                    self.assertEqual(shared, {"max_new_tokens": 20})

    def test_truncated_prompt_keeps_its_reserved_tokens(self):
        """Reserved tokens count in the total after truncation; a multimodal prompt is rejected instead."""
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.max_req_len = 100
        manager.max_req_input_len = 91
        manager.num_reserved_tokens = 3
        manager._embedding_preserve_last_token_id = None
        manager.allow_auto_truncate = True
        manager.validate_total_tokens = True
        manager.is_generation = True
        manager.model_config = SimpleNamespace(
            joint_head_config=None, pointer_head=None
        )
        req = GenerateReqInput(
            input_ids=[1] * 120, sampling_params={"max_new_tokens": 20}
        )
        req.normalize_batch_and_arguments()
        manager._validate_one_request(req, req.input_ids)

        self.assertEqual(len(req.input_ids), 91)
        self.assertEqual(req.sampling_params["max_new_tokens"], 100 - 91 - 3)

        mm_req = GenerateReqInput(
            input_ids=[1] * 120, sampling_params={"max_new_tokens": 20}
        )
        mm_req.normalize_batch_and_arguments()
        with self.assertRaisesRegex(
            ValueError, "Multimodal prompt is too long after expanding"
        ):
            manager._validate_one_request(mm_req, mm_req.input_ids, is_multimodal=True)
        self.assertEqual(len(mm_req.input_ids), 120)
        manager.allow_auto_truncate = False
        long_req = GenerateReqInput(
            input_ids=[1] * 10, sampling_params={"max_new_tokens": 200}
        )
        long_req.normalize_batch_and_arguments()
        with self.assertRaisesRegex(
            ValueError, "This model's maximum context length is 100 tokens"
        ):
            manager._validate_one_request(long_req, long_req.input_ids)


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
        "completion_tokens",
        "reasoning_tokens",
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


def _make_tokenizer_manager(case) -> TokenizerManager:
    """Create a TokenizerManager with mocked dependencies, bypassing __init__.

    The config it reads comes from the bags, so the stand-in needs a published
    config rather than attributes on a mock.
    """
    override = get_context().override_server_args(speculative_algorithm=None)
    override.install()
    case.addCleanup(override.restore)
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm.mm_processor = None
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
    tm.encoder_dispatch_ready = {}
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
    obj.return_routed_experts = False
    obj.return_sampling_mask = False
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


def _request_stats(**overrides) -> RequestOutputStats:
    fields = dict(
        prompt_tokens=0,
        cached_tokens=0,
        cached_tokens_details=None,
        retraction_count=0,
        image_tokens=0,
        audio_tokens=0,
        video_tokens=0,
        time_stats=None,
    )
    return RequestOutputStats(**{**fields, **overrides})


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
        elif f.name == "output_ids_flat":
            kwargs[f.name] = array("q")
        elif f.name == "output_ids_offsets":
            kwargs[f.name] = [0, 0]
        elif f.name == "request_stats":
            kwargs[f.name] = [_request_stats()]
        elif f.name in _PER_REQUEST_INT_FIELDS:
            kwargs[f.name] = [0]
        elif f.name in _PER_REQUEST_FLOAT_FIELDS:
            kwargs[f.name] = [0.0]
        elif f.name in _PER_REQUEST_NESTED_LIST_FIELDS:
            kwargs[f.name] = [[]]
        elif f.name in _PER_REQUEST_OPTIONAL_FIELDS:
            kwargs[f.name] = [None]
        elif f.name == "output_token_sampling_mask":
            kwargs[f.name] = None
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


class TestEngineResponseWait(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tm = _make_tokenizer_manager(self)
        self.tm.incremental_streaming_output = True
        self.tm.request_logger = Mock()
        self.tm.request_metrics_exporter_manager = Mock()
        self.tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
        self.state = _make_req_state("engine_wait")
        self.state.obj.stream = True
        self.state.obj.background = False
        self.tm.rid_to_state[self.state.obj.rid] = self.state

    async def test_available_and_later_outputs_keep_order(self):
        stream = self.tm._wait_one_response(self.state.obj)
        first = {"output_ids": [1], "meta_info": {"finish_reason": None}}
        final = {"output_ids": [2], "meta_info": {"finish_reason": {"type": "length"}}}
        self.state.out_list.append(first)
        self.state.event.set()
        with patch(
            "asyncio.wait_for",
            side_effect=AssertionError("Engine installed an HTTP timeout"),
        ):
            self.assertEqual((await anext(stream))["output_ids"], [1])
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
            self.state.out_list.append(final)
            self.state.finished = True
            self.state.event.set()
            self.assertEqual((await pending)["output_ids"], [2])
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)

    async def test_cumulative_stream_backlog_preserves_customized_info(self):
        """Cumulative streaming returns only the last queued output.

        A customized_info key that arrived on an earlier queued output must
        still reach the client through the accumulated state.
        """
        self.tm.incremental_streaming_output = False
        key = "request_score"
        hidden_key = "hidden_metric"
        stream = self.tm._wait_one_response(self.state.obj)
        first_meta_info = {"finish_reason": None}
        self.tm.update_request_meta_info(
            first_meta_info, self.state, {key: [[386.25]]}, 0, None
        )
        # Accumulated without being written to meta_info, as a subclass may do.
        self.state.customized_info_accumulated[hidden_key] = [1.0, 2.0]
        self.state.out_list.extend(
            [
                {"output_ids": [1], "meta_info": first_meta_info},
                {
                    "output_ids": [1, 2],
                    "meta_info": {"finish_reason": {"type": "length"}},
                },
            ]
        )
        self.state.finished = True
        self.state.event.set()

        out = await anext(stream)

        self.assertEqual(out["output_ids"], [1, 2])
        self.assertEqual(out["meta_info"][key], [386.25])
        self.assertNotIn(hidden_key, out["meta_info"])
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)

    async def test_cancelling_pending_wait_removes_event_waiter(self):
        stream = self.tm._wait_one_response(self.state.obj)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        self.assertTrue(self.state.event._waiters)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertFalse(self.state.event._waiters)
        await stream.aclose()

    async def test_abort_and_shutdown_errors_wake_engine_iterator(self):
        for status in (400, 500, 503):
            with self.subTest(status=status):
                state = _make_req_state(f"engine_abort_{status}")
                state.obj.stream = True
                self.tm.rid_to_state[state.obj.rid] = state
                stream = self.tm._wait_one_response(state.obj)
                pending = asyncio.create_task(anext(stream))
                await asyncio.sleep(0)
                self.tm._handle_abort_req(
                    AbortReq(
                        rid=state.obj.rid,
                        finished_reason={
                            "type": "abort",
                            "status_code": status,
                            "message": "test",
                        },
                    )
                )
                result = await pending
                self.assertEqual(
                    result["meta_info"]["finish_reason"]["status_code"], status
                )
                self.assertNotIn(state.obj.rid, self.tm.rid_to_state)
                with self.assertRaises(StopAsyncIteration):
                    await anext(stream)

    async def test_non_stream_abort_with_gateway_error_is_raised(self):
        """A non-stream abort carrying a gateway status such as 502 must become an HTTP error, not an empty 200."""
        state = _make_req_state("gateway_abort")
        state.obj.stream = False
        self.tm.rid_to_state[state.obj.rid] = state
        out = {
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "status_code": 502,
                    "message": "encoder dispatch failed",
                }
            }
        }
        with self.assertRaises(HTTPException) as ctx:
            await self.tm._handle_abort_finish_reason(out, state, is_stream=False)
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertNotIn(state.obj.rid, self.tm.rid_to_state)

    async def test_http_timeout_still_checks_disconnection(self):
        request = Mock()
        request.is_disconnected = AsyncMock(return_value=True)
        self.tm.abort_request = Mock()
        stream = self.tm._wait_one_response(self.state.obj, request)
        with patch(
            "sglang.srt.managers.tokenizer_manager._REQUEST_STATE_WAIT_TIMEOUT", 0.001
        ):
            with self.assertRaisesRegex(ValueError, "disconnected"):
                await anext(stream)
        request.is_disconnected.assert_awaited_once()
        self.tm.abort_request.assert_called_once_with(self.state.obj.rid)


class TestRidToStateCleanupOnAbort(CustomTestCase):
    """Test that _handle_abort_req removes rid from rid_to_state."""

    def test_abort_removes_rid_from_state(self):
        """An abort must release its adapter even when no response waiter remains."""

        async def drive():
            tm = _make_tm_for_generate(self, enable_lora=True)
            obj = GenerateReqInput(
                text="prompt", rid="abort_test_rid", lora_path="adapter"
            )
            obj.normalize_batch_and_arguments()
            tm._init_req_state(obj)
            await tm._validate_and_resolve_lora(obj)

            tm._handle_abort_req(_make_abort_req(obj.rid))
            await asyncio.gather(*tm._lora_release_tasks)
            self.assertNotIn(obj.rid, tm.rid_to_state)
            self.assertEqual(tm.lora_registry._counters["adapter"].value(), 0)

        asyncio.run(drive())

    def test_abort_allows_resubmit_same_rid(self):
        """After abort, _init_req_state should accept the same rid again."""
        tm = _make_tokenizer_manager(self)
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
        tm = _make_tokenizer_manager(self)
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


class TestAbortOutputPayload(CustomTestCase):
    """An abort chunk is often the only thing a client sees;
    it must carry the same optional fields as a normal finish chunk."""

    def test_abort_returns_the_sampling_masks_of_the_tokens_so_far(self):
        """A retracted request aborted from the waiting queue still has output tokens."""
        chunk = SamplingMaskChunk(
            lengths=np.asarray([3, 1], dtype=np.int32),
            token_ids=np.asarray([5, 7, 9, 2], dtype=np.int32),
            logprobs=np.asarray([-0.5, -1.0], dtype=np.float32),
        )
        for flat, routed_experts in itertools.product((False, True), repeat=2):
            with self.subTest(flat=flat, routed_experts=routed_experts):
                tm = _make_tokenizer_manager(self)
                obj = GenerateReqInput(
                    text="hi",
                    return_sampling_mask=True,
                    return_flat_raw_sampling_mask=flat,
                    return_routed_experts=routed_experts,
                )
                obj.normalize_batch_and_arguments()
                state = ReqState(
                    out_list=[],
                    finished=False,
                    event=asyncio.Event(),
                    obj=obj,
                    time_stats=APIServerReqTimeStats(),
                )
                state.output_ids = [5, 2]
                _add_sampling_mask_to_meta_info({}, state, chunk, finished=False)
                tm.rid_to_state[obj.rid] = state

                tm._handle_abort_req(_make_abort_req(obj.rid))

                meta_info = state.out_list[-1]["meta_info"]
                self.assertEqual(
                    meta_info["output_token_sampling_mask_length"],
                    len(state.out_list[-1]["output_ids"]),
                )

    def test_abort_includes_prompt_token_ids_only_when_requested(self):
        """The abort chunk carries prompt_token_ids captured at tokenization,
        and omits the field when the request did not ask for them."""
        tm = _make_tokenizer_manager(self)
        with_ids = _make_req_state("abort_prompt_ids_rid")
        with_ids.prompt_token_ids = [1, 2, 3]
        without_ids = _make_req_state("abort_no_prompt_ids_rid")

        for state in (with_ids, without_ids):
            tm.rid_to_state[state.obj.rid] = state
            tm._handle_abort_req(_make_abort_req(state.obj.rid))

        self.assertEqual(with_ids.out_list[0]["prompt_token_ids"], [1, 2, 3])
        self.assertNotIn("prompt_token_ids", without_ids.out_list[0])

    def test_abort_output_ids_match_the_streaming_mode(self):
        """Only incremental streaming collapses the abort chunk to the last
        token; cumulative chunks supersede, so they carry the whole generation.
        """
        cases = [
            ("incremental stream", True, True, [7]),
            ("cumulative stream", True, False, [5, 6, 7]),
            ("non-stream", False, False, [5, 6, 7]),
        ]
        for name, is_stream, incremental, expected in cases:
            with self.subTest(name):
                tm = _make_tokenizer_manager(self)
                tm.incremental_streaming_output = incremental
                rid = f"abort_output_ids_{name}"
                state = _make_req_state(rid)
                state.obj.stream = is_stream
                state.output_ids = [5, 6, 7]
                tm.rid_to_state[rid] = state

                tm._handle_abort_req(_make_abort_req(rid))

                out = state.out_list[0]
                self.assertEqual(out["output_ids"], expected)
                self.assertEqual(out["meta_info"]["completion_tokens"], 3)

    def test_abort_with_routed_experts_returns_no_output(self):
        """An abort echo has no routed-experts rows, so a request that asked
        for them gets no output tokens and no output logprobs either."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_routed_experts_rid"
        state = _make_req_state(rid)
        state.obj.return_routed_experts = True
        state.obj.return_logprob = True
        state.obj.top_logprobs_num = 0
        state.obj.token_ids_logprob = None
        state.obj.return_text_in_logprobs = False
        state.output_ids = [5, 6, 7]
        state.output_token_logprobs_val = [-0.1, -0.2, -0.3]
        state.output_token_logprobs_idx = [5, 6, 7]
        tm.rid_to_state[rid] = state

        tm._handle_abort_req(_make_abort_req(rid))

        out = state.out_list[0]
        self.assertEqual(out["output_ids"], [])
        self.assertEqual(out["text"], "")
        meta = out["meta_info"]
        self.assertEqual(meta["completion_tokens"], 0)
        self.assertEqual(meta["routed_experts"], "")
        self.assertEqual(meta["output_token_logprobs"], [])
        self.assertEqual(meta["output_token_logprobs_length"], 0)

    def test_abort_with_routed_experts_returns_no_flat_output_top_logprobs(self):
        """With flat raw output top logprobs, the blanked echo describes 0 rows."""
        for b64 in (False, True):
            with self.subTest(b64=b64):
                tm = _make_tokenizer_manager(self)
                rid = f"abort_routed_experts_flat_rid_{b64}"
                state = _make_req_state(rid)
                state.obj.return_routed_experts = True
                state.obj.return_logprob = True
                state.obj.top_logprobs_num = 2
                state.obj.token_ids_logprob = None
                state.obj.return_text_in_logprobs = False
                state.obj.return_flat_raw_top_logprobs = False
                state.obj.return_flat_raw_output_top_logprobs = True
                state.obj.return_flat_raw_top_logprobs_b64 = b64
                state.output_ids = [5, 6, 7]
                state.output_token_logprobs_val = [-0.1, -0.2, -0.3]
                state.output_token_logprobs_idx = [5, 6, 7]
                state.output_top_logprobs_val = [[-0.1, -2.0]] * 3
                state.output_top_logprobs_idx = [[5, 9]] * 3
                tm.rid_to_state[rid] = state

                tm._handle_abort_req(_make_abort_req(rid))

                meta = state.out_list[0]["meta_info"]
                self.assertEqual(meta["completion_tokens"], 0)
                self.assertEqual(meta["output_top_logprobs_shape"], [0, 2])
                self.assertEqual(meta["output_top_logprobs_null_prefix"], 0)
                suffix = "_flat_b64" if b64 else "_flat"
                empty = "" if b64 else []
                self.assertEqual(meta[f"output_top_logprobs_val{suffix}"], empty)
                self.assertEqual(meta[f"output_top_logprobs_idx{suffix}"], empty)

    def test_blanked_abort_echo_reports_weight_versions_of_no_tokens(self):
        """The weight versions of a blanked echo describe the 0 returned tokens."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_routed_experts_versions_rid"
        state = _make_req_state(rid)
        state.obj.return_routed_experts = True
        state.output_ids = [5, 6, 7]
        tm.rid_to_state[rid] = state
        abort = _make_abort_req(rid)
        abort.weight_versions = [
            WeightVersionSpan(version="v1", start=0, end=2),
            WeightVersionSpan(version="v2", start=2, end=3),
        ]

        tm._handle_abort_req(abort)

        meta = state.out_list[0]["meta_info"]
        self.assertEqual(meta["completion_tokens"], 0)
        self.assertEqual(
            meta["weight_versions"], [{"version": "v1", "start": 0, "end": 0}]
        )
        self.assertEqual(meta["weight_version"], "v1")

    def test_streaming_abort_with_routed_experts_keeps_tokens(self):
        """A streaming client has already received the tokens; the final abort
        chunk is not blanked."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_routed_experts_stream_rid"
        state = _make_req_state(rid)
        state.obj.stream = True
        state.obj.return_routed_experts = True
        state.output_ids = [5, 6, 7]
        tm.rid_to_state[rid] = state

        tm._handle_abort_req(_make_abort_req(rid))

        self.assertEqual(state.out_list[0]["meta_info"]["completion_tokens"], 3)
        self.assertNotIn("routed_experts", state.out_list[0]["meta_info"])


class TestRidToStateCleanupOnBatchOutput(CustomTestCase):
    """Test that _handle_batch_output removes rid from rid_to_state on completion."""

    def test_batch_output_removes_rid_on_finish(self):
        """When a request finishes in _handle_batch_output, rid should be removed."""
        tm = _make_tokenizer_manager(self)
        rid = "batch_finish_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid)
        asyncio.run(tm._handle_batch_output(batch_output))

        self.assertNotIn(rid, tm.rid_to_state)

    def test_batch_output_allows_resubmit_after_finish(self):
        """After a request finishes, the same rid can be resubmitted."""
        tm = _make_tokenizer_manager(self)
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
        tm = _make_tokenizer_manager(self)
        rid = "batch_ongoing_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        # finished_reason=_NOT_FINISHED means the request is still ongoing
        batch_output = _make_batch_str_output(rid, finished_reason=_NOT_FINISHED)
        asyncio.run(tm._handle_batch_output(batch_output))

        self.assertIn(rid, tm.rid_to_state)


class TestStoredRequestStats(CustomTestCase):
    def test_rows_without_request_stats_read_the_latest_request_stats(self):
        """A row without request stats reports the latest received, and a row
        before any is dropped."""
        tm = _make_tokenizer_manager(self)
        rid = "request_stats_rid"
        state = _make_req_state(rid)
        state.obj.stream = True
        tm.rid_to_state[rid] = state
        first = _request_stats(prompt_tokens=5, cached_tokens=2)
        retracted = _request_stats(prompt_tokens=5, cached_tokens=3, retraction_count=1)
        for request_stats, finished_reason in (
            ([first], _NOT_FINISHED),
            (None, _NOT_FINISHED),
            ([retracted], _NOT_FINISHED),
            ([None], None),
        ):
            batch_output = _make_batch_str_output(rid, finished_reason)
            batch_output.request_stats = request_stats
            asyncio.run(tm._handle_batch_output(batch_output))

        self.assertEqual(
            [
                (m["prompt_tokens"], m["cached_tokens"], m["num_retractions"])
                for m in (out["meta_info"] for out in state.out_list)
            ],
            [(5, 2, 0), (5, 2, 0), (5, 3, 1), (5, 3, 1)],
        )

        # A late row of an aborted request whose rid a new request reused.
        reused = tm.rid_to_state[rid] = _make_req_state(rid)
        batch_output = _make_batch_str_output(rid, _NOT_FINISHED)
        batch_output.request_stats = None
        asyncio.run(tm._handle_batch_output(batch_output))
        self.assertEqual(reused.out_list, [])
        self.assertIs(tm.rid_to_state[rid], reused)


class TestRequestTpotGating(CustomTestCase):
    """Request TPOT is observed once, only for requests with a real decode
    interval that ran to completion on a non-prefill worker."""

    def _run(
        self,
        batches,
        disaggregation_mode=DisaggregationMode.NULL,
        *,
        stream=False,
        prefill_finished_time=0.0,
    ):
        tm = _make_tokenizer_manager(self)
        tm.enable_metrics = True
        tm.enable_priority_scheduling = False
        tm.disaggregation_mode = disaggregation_mode
        tm.metrics_collector = MagicMock(labels={})
        rid = "tpot_rid"
        state = _make_req_state(rid)
        state.obj.log_metrics = True
        state.obj.sampling_params = {}
        state.obj.custom_labels = None
        state.obj.stream = stream
        state.time_stats.created_time = time.perf_counter() - 1.0
        tm.rid_to_state[rid] = state
        sched_stats = SchedulerReqTimeStats(
            enable_metrics=True, prefill_finished_time=prefill_finished_time
        )
        for completion_tokens, finished_reason in batches:
            batch_output = _make_batch_str_output(rid, finished_reason)
            batch_output.completion_tokens = [completion_tokens]
            batch_output.request_stats = [
                _request_stats(time_stats=sched_stats.to_output_time_stats())
            ]
            asyncio.run(tm._handle_batch_output(batch_output))
        (call,) = tm.metrics_collector.observe_one_finished_request.call_args_list
        return call.kwargs["time_per_output_token"]

    def test_single_batch_finish(self):
        produced = time.perf_counter() - 0.5
        # Streaming: first and final output in one batch leave no decode interval.
        self.assertIsNone(
            self._run(
                [(40, {"type": "stop"})],
                stream=True,
                prefill_finished_time=produced,
            )
        )
        # Non-streaming: decode runs from the scheduler's first-token time.
        self.assertGreater(
            self._run([(40, {"type": "stop"})], prefill_finished_time=produced),
            0.0,
        )

    def test_multi_batch_records_only_completed_decode(self):
        cases = [
            ("stop", {"type": "stop"}, DisaggregationMode.NULL, True),
            ("abort", {"type": "abort"}, DisaggregationMode.NULL, False),
            ("prefill", {"type": "stop"}, DisaggregationMode.PREFILL, False),
        ]
        for name, finished_reason, mode, recorded in cases:
            with self.subTest(name):
                tpot = self._run(
                    [(1, _NOT_FINISHED), (3, finished_reason)],
                    disaggregation_mode=mode,
                )
                if recorded:
                    self.assertGreater(tpot, 0.0)
                else:
                    self.assertIsNone(tpot)


class TestNonStreamingFirstTokenTime(CustomTestCase):
    def _run(self, *, stream: bool, prefill_finished_time: float):
        tm = _make_tokenizer_manager(self)
        rid = "ttft_rid"
        state = _make_req_state(rid)
        state.obj.stream = stream
        state.time_stats.created_time = time.perf_counter() - 1.0
        tm.rid_to_state[rid] = state
        sched_stats = SchedulerReqTimeStats(
            enable_metrics=True, prefill_finished_time=prefill_finished_time
        )
        batch_output = _make_batch_str_output(rid, finished_reason=_NOT_FINISHED)
        batch_output.request_stats = [
            _request_stats(time_stats=sched_stats.to_output_time_stats())
        ]
        arrival = time.perf_counter()
        asyncio.run(tm._handle_batch_output(batch_output))
        return state.time_stats, arrival

    def test_non_streaming_uses_scheduler_time(self):
        produced = time.perf_counter() - 0.5
        stats, arrival = self._run(stream=False, prefill_finished_time=produced)
        self.assertAlmostEqual(stats.first_token_time, produced, delta=1e-3)
        self.assertGreaterEqual(stats.last_time, arrival)

    def test_streaming_uses_arrival_time(self):
        produced = time.perf_counter() - 0.5
        stats, arrival = self._run(stream=True, prefill_finished_time=produced)
        self.assertGreaterEqual(stats.first_token_time, arrival)

    def test_falls_back_to_arrival_without_scheduler_time(self):
        stats, arrival = self._run(stream=False, prefill_finished_time=0.0)
        self.assertGreaterEqual(stats.first_token_time, arrival)


class TestInitReqStateDuplicateDetection(CustomTestCase):
    """Test that _init_req_state raises ValueError for duplicate rids."""

    def test_duplicate_rid_raises_error(self):
        """_init_req_state should raise ValueError if rid already exists."""
        tm = _make_tokenizer_manager(self)
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
        tm = _make_tokenizer_manager(self)
        rid = "unique_rid"

        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None

        tm._init_req_state(obj)
        self.assertIn(rid, tm.rid_to_state)


class TestInitReqStateAtomicity(CustomTestCase):
    def assert_map_unchanged(self, tm, original_map, original_states):
        self.assertIs(tm.rid_to_state, original_map)
        self.assertEqual(set(tm.rid_to_state), set(original_states))
        for rid, state in original_states.items():
            self.assertIs(tm.rid_to_state[rid], state)

    def test_live_collision_preserves_map_and_adapter_references(self):
        for request_type in (GenerateReqInput, EmbeddingReqInput):
            for collision_index in range(3):
                with self.subTest(
                    request_type=request_type.__name__, position=collision_index
                ):
                    tm = _make_tm_for_generate(self, enable_lora=True)
                    tm.enable_trace = True
                    live_state = _make_req_state("live")
                    tm.rid_to_state = {"live": live_state}
                    original_map = tm.rid_to_state
                    original_states = dict(original_map)
                    tm.abort_request = Mock()
                    tm._release_lora_once = Mock()
                    tm._tokenize_one_request = AsyncMock()
                    tm._send_one_request = AsyncMock()
                    rids = ["fresh0", "fresh1", "fresh2"]
                    rids[collision_index] = "live"
                    obj = request_type(
                        text=["a", "b", "c"], rid=rids, lora_path="adapter"
                    )

                    async def drive():
                        live_state.obj.lora_path = "adapter"
                        live_state.obj.lora_id = await tm.lora_registry.acquire(
                            "adapter"
                        )
                        with self.assertRaisesRegex(
                            ValueError, "Duplicate request ID detected: live"
                        ):
                            await tm.generate_request(obj).__anext__()
                        self.assertEqual(
                            tm.lora_registry._counters["adapter"].value(), 1
                        )

                    with patch.object(APIServerReqTimeStats, "init_trace_ctx") as trace:
                        asyncio.run(drive())
                    self.assert_map_unchanged(tm, original_map, original_states)
                    trace.assert_not_called()
                    tm.abort_request.assert_not_called()
                    tm._release_lora_once.assert_not_called()
                    tm._tokenize_one_request.assert_not_called()
                    tm._send_one_request.assert_not_called()
                    self.assertEqual(tm.send_to_scheduler.method_calls, [])
                    self.assertFalse(live_state.abort_sent)
                    self.assertFalse(live_state.lora_released)

    def test_initialization_exceptions_preserve_map(self):
        targets = (
            "ReqState",
            "APIServerReqTimeStats",
            "APIServerReqTimeStats.init_trace_ctx",
            "APIServerReqTimeStats.set_created_time",
        )
        for target in targets:
            for failure_index in range(2):
                with self.subTest(target=target, failure_index=failure_index):
                    tm = _make_tokenizer_manager(self)
                    tm.enable_trace = True
                    tm.rid_to_state = {"live": _make_req_state("live")}
                    original_map = tm.rid_to_state
                    original_states = dict(original_map)
                    obj = GenerateReqInput(text=["a", "b"], rid=["fresh0", "fresh1"])
                    obj.normalize_batch_and_arguments()
                    first_result = {
                        "ReqState": _make_req_state("fresh0"),
                        "APIServerReqTimeStats": APIServerReqTimeStats(),
                    }.get(target)
                    effects = [first_result] * failure_index + [
                        RuntimeError("initialization failed")
                    ]
                    with patch(
                        f"sglang.srt.managers.tokenizer_manager.{target}",
                        side_effect=effects,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "initialization failed"
                        ):
                            tm._init_req_state(obj)
                    self.assert_map_unchanged(tm, original_map, original_states)

    def test_valid_single_and_batch_requests(self):
        for request_type in (GenerateReqInput, EmbeddingReqInput):
            for single in (True, False):
                with self.subTest(request_type=request_type.__name__, single=single):
                    tm = _make_tokenizer_manager(self)
                    live_state = _make_req_state("live")
                    tm.rid_to_state = {"live": live_state}
                    obj = request_type(
                        text="a" if single else ["a", "b"],
                        rid="fresh" if single else ["fresh0", "fresh1"],
                        received_time=123.0,
                    )
                    obj.normalize_batch_and_arguments()
                    tm._init_req_state(obj)
                    items = [obj] if single else [obj[i] for i in range(len(obj.rid))]
                    self.assertEqual(len(tm.rid_to_state), len(items) + 1)
                    self.assertIs(tm.rid_to_state["live"], live_state)
                    for item in items:
                        state = tm.rid_to_state[item.rid]
                        self.assertIs(state.obj, item)
                        self.assertEqual(state.time_stats.created_time, 123.0)
                        self.assertFalse(state.dispatched)
                        self.assertFalse(state.event.is_set())

    def test_within_batch_duplicates_preserve_map(self):
        for request_type in (GenerateReqInput, EmbeddingReqInput):
            with self.subTest(request_type=request_type.__name__):
                tm = _make_tm_for_generate(self)
                tm.rid_to_state = {"live": _make_req_state("live")}
                original_map = tm.rid_to_state
                original_states = dict(original_map)
                obj = request_type(text=["a", "b"], rid=["duplicate", "duplicate"])

                async def drive():
                    with self.assertRaisesRegex(ValueError, "Duplicate"):
                        await tm.generate_request(obj).__anext__()

                asyncio.run(drive())
                self.assert_map_unchanged(tm, original_map, original_states)

    def test_direct_initialization_rejects_duplicate_ids(self):
        tm = _make_tokenizer_manager(self)
        obj = GenerateReqInput(text=["a", "b"], rid=["fresh0", "fresh1"])
        obj.normalize_batch_and_arguments()
        obj.rid = ["duplicate", "duplicate"]
        with self.assertRaisesRegex(ValueError, "Duplicate request ID"):
            tm._init_req_state(obj)
        self.assertFalse(tm.rid_to_state)

    def test_trace_metadata_is_preserved(self):
        tm = _make_tokenizer_manager(self)
        tm.enable_trace = True
        obj = GenerateReqInput(
            text=["a", "b"],
            rid=["fresh0", "fresh1"],
            bootstrap_room=[10, 11],
            received_time=123.0,
        )
        obj.normalize_batch_and_arguments()
        request = Mock(headers={"traceparent": "parent", "unrelated": "ignored"})
        with patch.object(APIServerReqTimeStats, "init_trace_ctx") as trace:
            tm._init_req_state(obj, request)
        self.assertEqual(trace.call_count, 2)
        for i, trace_call in enumerate(trace.call_args_list):
            self.assertEqual(
                trace_call.args, (obj.rid[i], 10 + i, {"traceparent": "parent"})
            )
            self.assertEqual(tm.rid_to_state[obj.rid[i]].time_stats.created_time, 123.0)


class TestResubmitAfterCompletion(CustomTestCase):
    """End-to-end test: complete a request, then resubmit with the same rid."""

    def test_complete_then_resubmit_same_rid(self):
        """A request that completes normally should allow resubmission with the same rid."""
        tm = _make_tokenizer_manager(self)
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
        tm = _make_tokenizer_manager(self)
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


def _make_tm_for_generate(case, enable_lora=False) -> TokenizerManager:
    """Augment the mocked TokenizerManager with what generate_request needs."""
    tm = _make_tokenizer_manager(case)
    tm.server_args.language_only = False
    tm.server_args.tokenizer_worker_num = 1
    tm.server_args.enable_strict_thinking = False
    tm.auto_create_handle_loop = Mock()
    tm._set_default_priority = Mock()
    tm.request_logger = RequestLogger(
        log_requests=False,
        log_requests_level=2,
        log_requests_format="text",
        log_requests_target=None,
    )
    tm.tokenizer = None
    tm.is_pause = False
    tm.is_pause_cond = asyncio.Condition()
    tm._elastic_shrink_pause_event = asyncio.Event()
    tm._elastic_shrink_pause_event.set()
    tm.model_update_lock = Mock()
    tm.model_update_lock.reader_lock = _DummyAsyncCM()
    tm.request_metrics_exporter_manager = Mock()
    tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
    tm._lora_release_tasks = set()
    if enable_lora:
        tm.enable_lora = True
        tm.server_args.max_loaded_loras = None
        tm.lora_registry = LoRARegistry(
            [LoRARef(lora_id="adapter", lora_name="adapter")]
        )
    else:
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


class TestReleaseReqStatesOnFailure(CustomTestCase):
    """Direct tests for _release_req_states_on_failure."""

    def test_undelivered_single_is_dropped(self):
        tm = _make_tokenizer_manager(self)
        rid = "d_single"
        tm.rid_to_state[rid] = _make_req_state(rid)
        tm._release_req_states_on_failure({rid: tm.rid_to_state[rid]})
        self.assertNotIn(rid, tm.rid_to_state)

    def test_undelivered_batch_removes_all(self):
        tm = _make_tokenizer_manager(self)
        rids = ["d0", "d1", "d2"]
        for r in rids:
            tm.rid_to_state[r] = _make_req_state(r)
        tm._release_req_states_on_failure(dict(tm.rid_to_state))
        for r in rids:
            self.assertNotIn(r, tm.rid_to_state)

    def test_ignores_already_removed(self):
        """A rid that is no longer present must not raise."""
        tm = _make_tokenizer_manager(self)
        tm.rid_to_state["p1"] = _make_req_state("p1")
        tm._release_req_states_on_failure(
            {
                "p1": tm.rid_to_state["p1"],
                "already_gone": _make_req_state("already_gone"),
            }
        )
        self.assertNotIn("p1", tm.rid_to_state)

    def test_dispatched_single_is_aborted_and_state_kept(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock()
        tm.enable_metrics = True
        tm.metrics_collector = MagicMock()
        rid = "d_live"
        state = _make_req_state(rid)
        state.dispatched = True
        tm.rid_to_state[rid] = state
        tm._release_req_states_on_failure({rid: tm.rid_to_state[rid]})
        tm._release_req_states_on_failure({rid: tm.rid_to_state[rid]})

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertEqual(
            [type(m) for m in sent], [AbortReq], "expected exactly one AbortReq"
        )
        self.assertEqual(sent[0].rid, rid)
        self.assertIn(rid, tm.rid_to_state)
        self.assertTrue(state.abort_sent)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_dispatched_batch_aborts_delivered_and_drops_rest(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock()
        delivered, undelivered = "d_delivered", "d_undelivered"
        live = _make_req_state(delivered)
        live.dispatched = True
        tm.rid_to_state[delivered] = live
        tm.rid_to_state[undelivered] = _make_req_state(undelivered)
        tm._release_req_states_on_failure(dict(tm.rid_to_state))

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertEqual([type(m) for m in sent], [AbortReq])
        self.assertEqual(sent[0].rid, delivered)
        self.assertIn(delivered, tm.rid_to_state)
        self.assertNotIn(undelivered, tm.rid_to_state)

    def test_abort_failure_does_not_stop_cleanup(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock(side_effect=RuntimeError("send failed"))
        delivered, undelivered = "live", "pending"
        live = _make_req_state(delivered)
        live.dispatched = True
        tm.rid_to_state[delivered] = live
        tm.rid_to_state[undelivered] = _make_req_state(undelivered)

        with self.assertLogs(level="ERROR"):
            tm._release_req_states_on_failure(dict(tm.rid_to_state))

        self.assertIn(delivered, tm.rid_to_state)
        self.assertFalse(live.abort_sent)
        self.assertNotIn(undelivered, tm.rid_to_state)

    def test_completed_sample_reuse_survives_sibling_cleanup(self):
        """Batch cleanup must preserve a reused sample RID, adapter ref and encoder gate."""

        async def drive(dispatched):
            tm = _make_tm_for_generate(self, enable_lora=True)
            tm._dispatch_to_scheduler = Mock()
            samples = [
                GenerateReqInput(text="sample", lora_path="adapter") for _ in range(2)
            ]
            for sample in samples:
                sample.normalize_batch_and_arguments()
                tm._init_req_state(sample)
                await tm._validate_and_resolve_lora(sample)
            request_states = dict(tm.rid_to_state)
            rid = samples[0].rid
            await tm._handle_batch_output(_make_batch_str_output(rid))
            await asyncio.gather(*tm._lora_release_tasks)

            replacement = GenerateReqInput(text="new", rid=rid, lora_path="adapter")
            replacement.normalize_batch_and_arguments()
            tm._init_req_state(replacement)
            await tm._validate_and_resolve_lora(replacement)
            replacement_state = tm.rid_to_state[rid]
            replacement_state.dispatched = dispatched
            gate = asyncio.Event()
            tm.encoder_dispatch_ready[rid] = gate

            tm._release_req_states_on_failure(request_states)
            await asyncio.gather(*tm._lora_release_tasks)
            self.assertIs(tm.rid_to_state.get(rid), replacement_state)
            self.assertNotIn(samples[1].rid, tm.rid_to_state)
            self.assertIs(tm.encoder_dispatch_ready.get(rid), gate)
            self.assertFalse(gate.is_set())
            self.assertEqual(tm.lora_registry._counters["adapter"].value(), 1)
            tm._dispatch_to_scheduler.assert_not_called()

            await tm._handle_batch_output(_make_batch_str_output(rid))
            await asyncio.gather(*tm._lora_release_tasks)
            self.assertEqual(tm.lora_registry._counters["adapter"].value(), 0)

        for dispatched in (False, True):
            with self.subTest(dispatched=dispatched):
                asyncio.run(drive(dispatched))


class TestParallelStreamTaskCleanup(CustomTestCase):
    def test_failing_choice_cancels_and_closes_sibling_waiters(self):
        tm = _make_tokenizer_manager(self)

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
        tm = _make_tokenizer_manager(self)

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
        """Validation failure must return the adapter reference acquired earlier."""
        tm = _make_tm_for_generate(self, enable_lora=True)
        obj = GenerateReqInput(text="prompt", rid="single_overlen", lora_path="adapter")
        tm._tokenize_one_request = AsyncMock(side_effect=ValueError("input too long"))
        tm._send_one_request = Mock()

        async def drive():
            with self.assertRaisesRegex(ValueError, "input too long"):
                await tm.generate_request(obj).__anext__()
            await asyncio.gather(*tm._lora_release_tasks)
            self.assertEqual(tm.lora_registry._counters["adapter"].value(), 0)

        asyncio.run(drive())
        tm._tokenize_one_request.assert_awaited_once()
        tm._send_one_request.assert_not_called()
        self.assertNotIn(obj.rid, tm.rid_to_state)

    def test_batch_failure_before_dispatch_cleans_up_all(self):
        tm = _make_tm_for_generate(self)
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

    def test_parallel_sampling_failure_cleans_generated_rid(self):
        tm = _make_tm_for_generate(self)
        obj = GenerateReqInput(
            text=["hello"],
            rid=["base"],
            sampling_params={"n": 2},
        )
        tokenized = MagicMock()
        tokenized.mm_inputs = None
        tokenized.sampling_params = MagicMock()
        tm._tokenize_one_request = AsyncMock(return_value=tokenized)
        tm._send_one_request = Mock(side_effect=RuntimeError("dispatch failed"))

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
            asyncio.run(drive())

        self.assertFalse(tm.rid_to_state)

    def test_thinking_budget_rejects_runtime_without_strict_thinking(self):
        tm = _make_tm_for_generate(self)
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


class TestWaitOneResponseAfterStateFreed(CustomTestCase):
    """A waiter built before its request finishes must still deliver the output.

    Batch dispatch builds every waiter before advancing any, and the
    scheduler-response path drops rid_to_state as soon as a request finishes.
    """

    def test_generator_built_before_finish_still_delivers_output(self):
        tm = _make_tokenizer_manager(self)
        tm.request_logger = Mock()
        tm.request_metrics_exporter_manager = MagicMock()
        tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
        rid = "freed_state_rid"
        state = _make_req_state(rid)
        state.obj.background = True  # skip the fastapi disconnect probe
        tm.rid_to_state[rid] = state

        async def drive():
            waiter = tm._wait_one_response(state.obj, None)
            await tm._handle_batch_output(_make_batch_str_output(rid))
            self.assertNotIn(rid, tm.rid_to_state)
            return await waiter.__anext__()

        out = asyncio.run(drive())
        self.assertEqual(out["meta_info"]["id"], rid)
        self.assertEqual(out["text"], "hello")


class TestResponseBeforeSendResumes(CustomTestCase):
    def test_completed_response_survives_send_resumption(self):
        for request_type, paths in (
            (GenerateReqInput, ("single", "batch", "sequential", "parallel")),
            (EmbeddingReqInput, ("single", "batch", "sequential")),
        ):
            for path in paths:
                for stream in (
                    (False, True) if request_type is GenerateReqInput else (False,)
                ):
                    with self.subTest(
                        request_type=request_type.__name__, path=path, stream=stream
                    ):
                        asyncio.run(self._drive(request_type, path, stream))

    def test_send_completion_preserves_reused_rid_and_encoder_gate(self):
        override = get_context().override_server_args(batch_notify_size=100)
        with override:
            for path in ("single", "batch"):
                with self.subTest(path=path):
                    asyncio.run(self._drive(GenerateReqInput, path, False, reuse=True))

    def test_dispatch_metadata_precedes_early_terminal_response(self):
        for request_type in (GenerateReqInput, EmbeddingReqInput):
            for path in ("single", "batch"):
                for pickle_ipc in (False, True):
                    with (
                        self.subTest(
                            request_type=request_type.__name__,
                            path=path,
                            pickle_ipc=pickle_ipc,
                        ),
                        patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                        get_context().override_server_args(batch_notify_size=100),
                    ):
                        asyncio.run(
                            self._drive(request_type, path, False, check_timing=True)
                        )

    async def _drive(self, request_type, path, stream, reuse=False, check_timing=False):
        tm = _make_tm_for_generate(self)
        tm.enable_metrics = check_timing
        if check_timing:
            tm.disaggregation_mode = DisaggregationMode.NULL
            tm.metrics_collector = Mock(labels={})
            tm.enable_priority_scheduling = False
        tm.cuda_vmm_feature_transport = SimpleNamespace(
            prepare_for_dispatch_async=AsyncMock(return_value=[]),
            cancel_for_dispatch=Mock(),
        )
        tm.positional_embeds_transport = SimpleNamespace(
            prepare_for_dispatch=Mock(return_value=[]), cancel_for_dispatch=Mock()
        )
        tm.batch_tokenized_generate_req_type = BatchTokenizedGenerateReqInput
        kwargs = {}
        if request_type is GenerateReqInput:
            kwargs = {"stream": stream, "return_prompt_token_ids": True}
            if path == "parallel":
                kwargs["sampling_params"] = {"n": 2, "max_new_tokens": 1}
        obj = request_type(text="prompt" if path == "single" else ["a", "b"], **kwargs)

        def tokenize(req):
            if reuse:
                old_gates[req.rid] = asyncio.Event()
                tm.encoder_dispatch_ready[req.rid] = old_gates[req.rid]
            tokenized = dict(
                rid=req.rid,
                input_text="prompt",
                input_ids=array("q", [1]),
                mm_inputs=None,
                token_type_ids=None,
                sampling_params=SamplingParams(max_new_tokens=1),
                time_stats=tm.rid_to_state[req.rid].time_stats,
            )
            if request_type is GenerateReqInput:
                return TokenizedGenerateReqInput(
                    **tokenized,
                    input_embeds=None,
                    return_logprob=False,
                    logprob_start_len=-1,
                    top_logprobs_num=0,
                    token_ids_logprob=None,
                    stream=stream,
                )
            return TokenizedEmbeddingReqInput(**tokenized)

        tm._tokenize_one_request = AsyncMock(side_effect=tokenize)
        tm._should_use_batch_tokenization = Mock(return_value=path == "batch")
        tm._batch_tokenize_and_process = AsyncMock(
            side_effect=lambda size, req: [tokenize(req[i]) for i in range(size)]
        )
        completed = []
        responders = []
        replacements = []
        old_gates = {}
        completed_states = []
        dispatch_finish_before_enqueue = []

        def dispatch(tokenized):
            future = asyncio.get_running_loop().create_future()
            if check_timing:
                if io_struct._USE_PICKLE_IPC:
                    tokenized = pickle.loads(pickle.dumps(tokenized))
                else:
                    tokenized = io_struct.msgpack_decode(
                        io_struct.msgpack_encode(tokenized)
                    )
            requests = getattr(tokenized, "batch", [tokenized])

            async def respond():
                if check_timing:
                    self.assertFalse(future.done())
                    for req in requests:
                        req.unwrap_pickle_fields()
                        state = tm.rid_to_state[req.rid]
                        self.assertFalse(state.dispatched)
                        self.assertIsNot(req.time_stats, state.time_stats)
                        self.assertEqual(
                            req.time_stats.api_server_dispatch_finish_time, 0.0
                        )
                        dispatch_finish_before_enqueue.append(
                            state.time_stats.api_server_dispatch_finish_time
                        )
                # Completing the send schedules its waiter; process the response
                # before that waiter gets its next event-loop turn.
                future.set_result(None)
                for req in requests:
                    state = tm.rid_to_state[req.rid]
                    if request_type is GenerateReqInput:
                        output = _make_batch_str_output(req.rid)
                    else:
                        output = BatchEmbeddingOutput(
                            rids=[req.rid],
                            finished_reasons=[{"type": "length"}],
                            embeddings=[[1.0]],
                            request_stats=[_request_stats(prompt_tokens=1)],
                            placeholder_tokens_idx=None,
                            placeholder_tokens_val=None,
                        )
                    await tm._handle_batch_output(output)
                    self.assertNotIn(req.rid, tm.rid_to_state)
                    completed.append(req.rid)
                    completed_states.append(state)
                    if reuse:
                        replacement = GenerateReqInput(text="new", rid=req.rid)
                        replacement.normalize_batch_and_arguments()
                        tm._init_req_state(replacement)
                        gate = asyncio.Event()
                        tm.encoder_dispatch_ready[req.rid] = gate
                        replacements.append(
                            (tm.rid_to_state[req.rid], gate, old_gates[req.rid])
                        )

            responders.append(asyncio.create_task(respond()))
            return future

        tm._dispatch_to_scheduler = Mock(side_effect=dispatch)
        generator = tm.generate_request(obj)
        try:
            outputs = [out async for out in generator]
            await asyncio.gather(*responders)
            if not stream and path != "single":
                outputs = outputs[0]
            expected = 1 if path == "single" else 4 if path == "parallel" else 2
            self.assertEqual(len(outputs), expected)
            self.assertEqual(
                len(completed), expected + (2 if path == "parallel" else 0)
            )
            if check_timing:
                self.assertEqual(len(dispatch_finish_before_enqueue), expected)
                self.assertTrue(all(ts > 0 for ts in dispatch_finish_before_enqueue))
            for out in outputs:
                self.assertIn(out["meta_info"]["id"], completed)
                if check_timing:
                    self.assertGreater(
                        out["meta_info"]["api_server_dispatch_finish_ts"], 0.0
                    )
                    self.assertLessEqual(
                        out["meta_info"]["api_server_dispatch_finish_ts"],
                        out["meta_info"]["request_finished_ts"],
                    )
                if request_type is GenerateReqInput:
                    self.assertEqual(out["text"], "hello")
                    self.assertEqual(out["prompt_token_ids"], [1])
                else:
                    self.assertEqual(out["embedding"], [1.0])
            if reuse:
                self.assertEqual(len(tm.rid_to_state), expected)
                for state, gate, old_gate in replacements:
                    self.assertIs(tm.rid_to_state[state.obj.rid], state)
                    self.assertFalse(state.dispatched)
                    self.assertIs(tm.encoder_dispatch_ready[state.obj.rid], gate)
                    self.assertFalse(gate.is_set())
                    if path == "single":
                        self.assertTrue(old_gate.is_set())
            else:
                self.assertFalse(tm.rid_to_state)
            if check_timing:
                self.assertTrue(all(state.finished for state in completed_states))
            tm.cuda_vmm_feature_transport.cancel_for_dispatch.assert_not_called()
            tm.positional_embeds_transport.cancel_for_dispatch.assert_not_called()
        finally:
            await asyncio.gather(*responders, return_exceptions=True)
            await generator.aclose()


class TestDisconnectDuringPendingSend(CustomTestCase):
    @unittest.skipUnless(sys.version_info >= (3, 12), "Eager tasks require Python 3.12")
    def test_immediate_http_send_does_not_yield(self):
        for failure in (None, RuntimeError, asyncio.CancelledError):
            with self.subTest(failure=failure):

                async def drive():
                    tm = _make_tm_for_generate(self)
                    obj = GenerateReqInput(text="prompt")
                    request = Mock()
                    turns = []
                    asyncio.get_running_loop().call_soon(turns.append, "callback")

                    async def send():
                        if failure is not None:
                            raise failure("enqueue failed")

                    if failure is None:
                        await tm._send_with_disconnect_monitor(send(), obj, request)
                    else:
                        with self.assertRaises(failure):
                            await tm._send_with_disconnect_monitor(send(), obj, request)
                    self.assertEqual(turns, [])
                    request.is_disconnected.assert_not_called()
                    await asyncio.sleep(0)
                    self.assertEqual(turns, ["callback"])

                asyncio.run(drive())

    def test_completed_future_and_custom_awaitable_are_supported(self):
        for custom in (False, True):
            with self.subTest(custom=custom):

                async def drive():
                    tm = _make_tm_for_generate(self)
                    future = asyncio.get_running_loop().create_future()
                    future.set_result(None)

                    class Send:
                        def __await__(self):
                            return future.__await__()

                    request = Mock()
                    await tm._send_with_disconnect_monitor(
                        Send() if custom else future,
                        GenerateReqInput(text="prompt"),
                        request,
                    )
                    request.is_disconnected.assert_not_called()

                asyncio.run(drive())

    def test_http_send_preserves_loop_task_factory(self):
        async def drive():
            tm = _make_tm_for_generate(self)
            loop = asyncio.get_running_loop()
            created = []

            def factory(loop, coro, **kwargs):
                created.append(coro)
                return asyncio.Task(coro, loop=loop, **kwargs)

            async def send():
                pass

            previous = loop.get_task_factory()
            loop.set_task_factory(factory)
            try:
                sending = send()
                await tm._send_with_disconnect_monitor(
                    sending, GenerateReqInput(text="prompt"), Mock()
                )
                self.assertEqual(created, [sending])
            finally:
                loop.set_task_factory(previous)

        asyncio.run(drive())

    def test_http_send_uses_scheduled_fallback_before_python_312(self):
        async def drive():
            tm = _make_tm_for_generate(self)
            turns = []
            asyncio.get_running_loop().call_soon(turns.append, "callback")

            async def send():
                self.assertEqual(turns, ["callback"])

            with patch(
                "sglang.srt.managers.tokenizer_manager.sys.version_info", (3, 11)
            ):
                await tm._send_with_disconnect_monitor(
                    send(), GenerateReqInput(text="prompt"), Mock()
                )

        asyncio.run(drive())

    def test_completed_sender_failure_during_caller_cancellation_is_retrieved(self):
        async def drive():
            tm = _make_tm_for_generate(self)
            obj = GenerateReqInput(text="prompt")
            loop = asyncio.get_running_loop()
            owner = asyncio.current_task()
            errors = []
            previous = loop.get_exception_handler()
            loop.set_exception_handler(lambda _, context: errors.append(context))
            try:

                async def fail():
                    raise RuntimeError("enqueue failed")

                send_task = asyncio.create_task(fail())
                send_task.add_done_callback(lambda _: owner.cancel())
                with self.assertRaises(asyncio.CancelledError):
                    await tm._send_with_disconnect_monitor(send_task, obj, Mock())
                del send_task
                await asyncio.sleep(0)
                self.assertEqual(errors, [])
            finally:
                loop.set_exception_handler(previous)

        asyncio.run(drive())

    def test_completed_cancelled_sender_preserves_disconnect_error(self):
        async def drive():
            tm = _make_tm_for_generate(self)
            obj = GenerateReqInput(text="prompt")
            send_task = asyncio.create_task(asyncio.Event().wait())

            async def disconnect():
                send_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await send_task
                return True

            request = Mock()
            request.is_disconnected = AsyncMock(side_effect=disconnect)
            with patch(
                "sglang.srt.managers.tokenizer_manager._REQUEST_STATE_WAIT_TIMEOUT",
                0.001,
            ):
                with self.assertRaisesRegex(ValueError, "disconnected"):
                    await tm._send_with_disconnect_monitor(send_task, obj, request)
            self.assertTrue(send_task.cancelled())
            request.is_disconnected.assert_awaited_once()

        asyncio.run(drive())

    def test_disconnect_releases_pending_send_and_reader_lock(self):
        for batched in (False, True):
            for pickle_ipc in (False, True):
                with self.subTest(batched=batched, pickle_ipc=pickle_ipc):
                    asyncio.run(self._drive(batched, pickle_ipc))

    def test_cancellation_after_enqueue_preserves_queued_resources(self):
        for pickle_ipc in (False, True):
            with self.subTest(pickle_ipc=pickle_ipc):
                asyncio.run(self._drive(False, pickle_ipc, queued=True))

    def test_cancellation_during_disconnect_probe_preserves_queued_resources(self):
        for batched in (False, True):
            for pickle_ipc in (False, True):
                with self.subTest(batched=batched, pickle_ipc=pickle_ipc):
                    asyncio.run(
                        self._drive(
                            batched, pickle_ipc, queued=True, cancel_during_probe=True
                        )
                    )

    async def _drive(
        self, batched, pickle_ipc, queued=False, cancel_during_probe=False
    ):
        tm = _make_tm_for_generate(self, enable_lora=True)
        tm.model_update_lock = RWLock()
        tm.tokenizer_ipc_name = None
        tm.disaggregation_mode = DisaggregationMode.NULL
        tm.cuda_vmm_feature_transport = SimpleNamespace(
            prepare_for_dispatch_async=AsyncMock(return_value=[]),
            cancel_for_dispatch=Mock(),
        )
        tm.positional_embeds_transport = SimpleNamespace(
            prepare_for_dispatch=Mock(return_value=[]), cancel_for_dispatch=Mock()
        )
        tm.batch_tokenized_generate_req_type = BatchTokenizedGenerateReqInput
        source = torch.arange(16, dtype=torch.float32)
        tokenized = []

        def tokenize(req):
            result = TokenizedGenerateReqInput(
                rid=req.rid,
                input_text="prompt",
                input_ids=array("q", [1]),
                input_embeds=None,
                mm_inputs=(
                    None
                    if batched
                    else MultimodalProcessorOutput(
                        mm_items=[
                            MultimodalDataItem(
                                modality=Modality.IMAGE, feature=source, hash=123
                            )
                        ]
                    )
                ),
                token_type_ids=None,
                sampling_params=SamplingParams(max_new_tokens=1),
                return_logprob=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
                lora_id=req.lora_id,
                time_stats=tm.rid_to_state[req.rid].time_stats,
            )
            tokenized.append(result)
            return result

        tm._tokenize_one_request = AsyncMock(side_effect=tokenize)
        tm._should_use_batch_tokenization = Mock(return_value=batched)
        tm._batch_tokenize_and_process = AsyncMock(
            side_effect=lambda size, req: [tokenize(req[i]) for i in range(size)]
        )
        obj = GenerateReqInput(
            text=["a", "b"] if batched else "prompt", lora_path="adapter"
        )
        messages = asyncio.Queue()

        async def receive():
            if cancel_during_probe and not futures[0].done():
                # Starlette's self-cancelling scope encloses this receive.
                # Complete enqueue and cancel the HTTP owner in that scope.
                futures[0].add_done_callback(lambda _: task.cancel())
                futures[0].set_result(None)
            return await messages.get()

        request = Request(
            {"type": "http", "method": "POST", "path": "/generate", "headers": []},
            receive=receive,
        )
        context = zmq.asyncio.Context()
        sender = context.socket(zmq.PUSH)
        peer = context.socket(zmq.PULL)
        monitor = sender.get_monitor_socket(events=zmq.EVENT_DISCONNECTED)
        endpoint = f"tcp://127.0.0.1:{sender.bind_to_random_port('tcp://127.0.0.1')}"
        tm.send_to_scheduler = sender
        pending = asyncio.Event()
        futures = []
        original_dispatch = tm._dispatch_to_scheduler

        def dispatch(packet):
            future = original_dispatch(packet)
            if not isinstance(packet, AbortReq):
                if cancel_during_probe:
                    # The real packet is queued; expose completion to the sender
                    # only when the HTTP disconnect probe starts.
                    self.assertTrue(future.done())
                    self.assertIsNone(future.exception())
                    future = asyncio.get_running_loop().create_future()
                self.assertFalse(future.done())
                futures.append(future)
                pending.set()
            return future

        tm._dispatch_to_scheduler = dispatch
        generator = tm.generate_request(obj, request)
        task = None
        try:
            peer.connect(endpoint)
            await asyncio.wait_for(sender.send(b"connected"), 3)
            self.assertEqual(await asyncio.wait_for(peer.recv(), 3), b"connected")
            if not cancel_during_probe:
                peer.close(linger=0)
                event = await asyncio.wait_for(recv_monitor_message(monitor), 3)
                self.assertEqual(event["event"], zmq.EVENT_DISCONNECTED)

                async def wait_until_unwritable():
                    while sender.getsockopt(zmq.EVENTS) & zmq.POLLOUT:
                        await asyncio.sleep(0.001)

                await asyncio.wait_for(wait_until_unwritable(), 3)
            self.assertEqual(sender.getsockopt(zmq.SNDTIMEO), -1)
            with (
                patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                patch.object(mm_utils, "_get_is_default_transport", return_value=False),
                patch.object(
                    mm_utils,
                    "get_serving",
                    return_value=SimpleNamespace(skip_tokenizer_init=False),
                ),
                patch(
                    "sglang.srt.managers.tokenizer_manager._REQUEST_STATE_WAIT_TIMEOUT",
                    0.01,
                ),
            ):
                task = asyncio.create_task(anext(generator))
                await asyncio.wait_for(pending.wait(), 3)
                self.assertEqual(tm.model_update_lock._readers, 1)
                self.assertFalse(futures[0].done())
                if queued:
                    if not cancel_during_probe:
                        futures[0].add_done_callback(lambda _: task.cancel())
                        peer = context.socket(zmq.PULL)
                        peer.connect(endpoint)
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 3)
                    self.assertGreater(task.cancelling(), 0)
                    packet = await asyncio.wait_for(io_struct.async_sock_recv(peer), 3)
                    for received in packet.batch if batched else [packet]:
                        if received.mm_inputs is not None:
                            mm_utils.unwrap_shm_features(received)
                            self.assertTrue(
                                torch.equal(
                                    received.mm_inputs.mm_items[0].feature, source
                                )
                            )
                            self.assertEqual(received.mm_inputs.mm_items[0].hash, 123)
                        state = tm.rid_to_state[received.rid]
                        self.assertTrue(state.dispatched)
                        self.assertTrue(state.abort_sent)
                        await tm._handle_batch_output(
                            _make_batch_str_output(received.rid)
                        )
                    tm.cuda_vmm_feature_transport.cancel_for_dispatch.assert_not_called()
                    tm.positional_embeds_transport.cancel_for_dispatch.assert_not_called()
                    if cancel_during_probe:
                        expected_rids = (
                            {received.rid for received in packet.batch}
                            if batched
                            else {packet.rid}
                        )
                        aborted_rids = set()
                        for _ in expected_rids:
                            abort = await asyncio.wait_for(
                                io_struct.async_sock_recv(peer), 3
                            )
                            self.assertIsInstance(abort, AbortReq)
                            aborted_rids.add(abort.rid)
                        self.assertEqual(aborted_rids, expected_rids)
                else:
                    messages.put_nowait({"type": "http.disconnect"})
                    with self.assertRaisesRegex(ValueError, "disconnected"):
                        await asyncio.wait_for(task, 0.5)
                    self.assertTrue(futures[0].cancelled())
                    self.assertFalse(tm.rid_to_state)
                    tm.cuda_vmm_feature_transport.cancel_for_dispatch.assert_called_once()
                    tm.positional_embeds_transport.cancel_for_dispatch.assert_called_once()
                    peer = context.socket(zmq.PULL)
                    peer.connect(endpoint)
                    await asyncio.wait_for(sender.send(b"marker"), 3)
                    self.assertEqual(await asyncio.wait_for(peer.recv(), 3), b"marker")
                self.assertEqual(tm.model_update_lock._readers, 0)
                async with tm.model_update_lock.writer_lock:
                    pass
                await asyncio.gather(*tm._lora_release_tasks)
                self.assertEqual(tm.lora_registry._counters["adapter"].value(), 0)
        finally:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await generator.aclose()
            for item in tokenized:
                mm_utils.discard_shm_features(item)
            sender.close(linger=0)
            peer.close(linger=0)
            monitor.close(linger=0)
            context.destroy(linger=0)


class TestDisconnectAfterDispatchAbortsRequest(CustomTestCase):
    """Cancellation after dispatch must stop the scheduler request."""

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj, **kwargs: obj,
    )
    def test_cancel_after_dispatch_sends_abort_and_keeps_state(self, _wrap_shm):
        tm = _make_tm_for_generate(self)
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = AsyncMock(
            return_value=[]
        )
        tm.positional_embeds_transport = Mock()
        tm._dispatch_to_scheduler = Mock()
        rid = "disconnect_zombie"
        obj = _make_generate_obj(rid, is_single=True)
        obj.return_prompt_token_ids = False
        tokenized = MagicMock()
        tokenized.rid = rid
        tokenized.mm_inputs = None
        tm._tokenize_one_request = AsyncMock(return_value=tokenized)

        async def drive():
            task = asyncio.create_task(tm.generate_request(obj).__anext__())
            for _ in range(100):
                await asyncio.sleep(0)
                if tm._dispatch_to_scheduler.called:
                    break
            self.assertTrue(
                tm._dispatch_to_scheduler.called, "request never dispatched"
            )
            state = tm.rid_to_state.get(rid)
            self.assertIsNotNone(state)
            self.assertTrue(state.dispatched)

            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(drive())

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        aborts = [m for m in sent if isinstance(m, AbortReq) and m.rid == rid]
        self.assertTrue(aborts, "disconnect must send an AbortReq to the scheduler")
        self.assertIn(rid, tm.rid_to_state)


class TestAbortBeforeDispatch(CustomTestCase):
    """An abort that lands before dispatch must finish the request as aborted.

    The scheduler drops an AbortReq for a rid it has not received, and
    abort_request sends only one, so a request dispatched after its abort
    would run unaborted while its caller waits.
    """

    @staticmethod
    def _make_tm(case) -> TokenizerManager:
        tm = _make_tm_for_generate(case)
        tm._dispatch_to_scheduler = Mock()
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = AsyncMock(
            return_value=[]
        )
        tm.positional_embeds_transport = Mock()
        # --mm-feature-transport=cuda_ipc: preprocessing published the features.
        tm.mm_processor = SimpleNamespace(cudaipc_mmfeature_pool=Mock())
        return tm

    @staticmethod
    def _mm_inputs():
        proxies = [
            CudaIpcTensorTransportProxy(
                data=torch.zeros(64, dtype=torch.uint8),
                info_data=torch.zeros(16),
                pool_ipc_handle=("test-pool",),
                pool_byte_offset=256 * (index + 1),
                ready_byte_offset=12 * index,
                ack_byte_offset=12 * index + 4,
                generation=1,
                total_consumer_count=2,
                use_pool_handle_cache=True,
            )
            for index in range(2)
        ]
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=proxies[0],
            precomputed_embeddings=proxies[1],
        )
        return MultimodalProcessorOutput(mm_items=[item])

    def _assert_cuda_ipc_cancelled(self, tm, mm_inputs):
        item = mm_inputs.mm_items[0]
        cancelled = tm.mm_processor.cudaipc_mmfeature_pool.cancel_proxy.call_args_list
        self.assertEqual(
            [c.args[0] for c in cancelled], [item.feature, item.precomputed_embeddings]
        )

    def _assert_finished_as_aborted(self, tm, rid, out):
        self.assertEqual(out["meta_info"]["id"], rid)
        self.assertEqual(out["meta_info"]["finish_reason"]["type"], "abort")
        self.assertNotIn(rid, tm.rid_to_state)

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj, **kwargs: obj,
    )
    def test_abort_during_preprocessing_is_not_dispatched(self, _wrap_shm):
        # Feature publication is the last await before dispatch.
        for stage in ("tokenize", "publish"):
            with self.subTest(stage=stage):
                tm = self._make_tm(self)
                rid = f"abort_during_{stage}"
                obj = GenerateReqInput(text="prompt", rid=rid, sampling_params={})
                tokenized = MagicMock()
                tokenized.rid = rid
                tokenized.mm_inputs = self._mm_inputs()
                prepared = [Mock()]

                async def drive():
                    entered, release = asyncio.Event(), asyncio.Event()

                    async def run_stage(name, result):
                        if name == stage:
                            entered.set()
                            await release.wait()
                        return result

                    tm._tokenize_one_request = lambda _obj: run_stage(
                        "tokenize", tokenized
                    )
                    tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = (
                        lambda _mm_inputs: run_stage("publish", prepared)
                    )
                    task = asyncio.create_task(tm.generate_request(obj).__anext__())
                    await asyncio.wait_for(entered.wait(), timeout=2)
                    tm.abort_request(rid)
                    release.set()
                    return await asyncio.wait_for(task, timeout=2)

                out = asyncio.run(drive())

                self._assert_finished_as_aborted(tm, rid, out)
                tm.cuda_vmm_feature_transport.cancel_for_dispatch.assert_called_once_with(
                    prepared
                )
                self._assert_cuda_ipc_cancelled(tm, tokenized.mm_inputs)
                sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
                self.assertEqual([type(m) for m in sent], [AbortReq])
                self.assertEqual(sent[0].rid, rid)

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj, **kwargs: obj,
    )
    def test_abort_all_skips_preprocessing_requests_but_not_paused_ones(
        self, _wrap_shm
    ):
        """pause_generation(mode="abort") aborts the requests in flight; those
        that arrive during the pause run after it."""
        tm = self._make_tm(self)
        objs = {
            rid: GenerateReqInput(text="prompt", rid=rid, sampling_params={})
            for rid in ("in_flight", "held")
        }
        tokenized = {}
        for rid in objs:
            tokenized[rid] = MagicMock()
            tokenized[rid].rid = rid
            tokenized[rid].mm_inputs = None

        async def drive():
            entered, release = asyncio.Event(), asyncio.Event()

            async def tokenize(obj):
                if obj.rid == "in_flight":
                    entered.set()
                    await release.wait()
                return tokenized[obj.rid]

            async def until_dispatched(rid):
                while not tm.rid_to_state[rid].dispatched:
                    await asyncio.sleep(0)

            tm._tokenize_one_request = tokenize
            in_flight = asyncio.create_task(
                tm.generate_request(objs["in_flight"]).__anext__()
            )
            await asyncio.wait_for(entered.wait(), timeout=2)
            async with tm.is_pause_cond:
                tm.is_pause = True
            held = asyncio.create_task(tm.generate_request(objs["held"]).__anext__())
            while "held" not in tm.rid_to_state:
                await asyncio.sleep(0)
            tm.abort_request(abort_all=True)
            release.set()
            out = await asyncio.wait_for(in_flight, timeout=2)
            sent_during_pause = [
                c.args[0] for c in tm._dispatch_to_scheduler.call_args_list
            ]

            async with tm.is_pause_cond:
                tm.is_pause = False
                tm.is_pause_cond.notify_all()
            await asyncio.wait_for(until_dispatched("held"), timeout=2)
            held.cancel()
            return out, sent_during_pause

        out, sent = asyncio.run(drive())

        self._assert_finished_as_aborted(tm, "in_flight", out)
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].abort_all)
        self.assertIs(
            tm._dispatch_to_scheduler.call_args_list[1].args[0], tokenized["held"]
        )

    def test_batch_dispatches_only_requests_not_aborted(self):
        tm = self._make_tm(self)
        kept, aborted = "batch_kept", "batch_aborted"
        for rid in (kept, aborted):
            tm.rid_to_state[rid] = _make_req_state(rid)
        tokenized_objs = []
        for rid in (kept, aborted):
            tokenized = MagicMock()
            tokenized.rid = rid
            tokenized.mm_inputs = self._mm_inputs()
            tokenized_objs.append(tokenized)

        async def publish(_mm_inputs):
            # The abort lands while the batch's features are being published.
            tm.abort_request(aborted)
            return []

        tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = publish

        async def drive():
            await tm._send_batch_request(tokenized_objs)
            waiter = tm._wait_one_response(tm.rid_to_state[aborted].obj, None)
            return await asyncio.wait_for(waiter.__anext__(), timeout=2)

        out = asyncio.run(drive())

        self._assert_finished_as_aborted(tm, aborted, out)
        tm.cuda_vmm_feature_transport.cancel_for_dispatch.assert_called_once_with(
            tokenized_objs[1].mm_inputs.mm_items
        )
        # Only the skipped request's: the scheduler acknowledges the other's.
        self._assert_cuda_ipc_cancelled(tm, tokenized_objs[1].mm_inputs)
        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertIsInstance(sent[0], AbortReq)
        self.assertEqual(sent[0].rid, aborted)
        self.assertEqual(len(sent), 2)
        self.assertEqual(list(sent[1].batch), [tokenized_objs[0]])
        self.assertTrue(tm.rid_to_state[kept].dispatched)

    def test_public_batch_finishes_request_aborted_during_publication(self):
        async def drive():
            tm = self._make_tm(self)
            tm.disaggregation_mode = DisaggregationMode.NULL
            tm.batch_tokenized_generate_req_type = BatchTokenizedGenerateReqInput
            obj = GenerateReqInput(text=["kept", "aborted"], stream=True)

            def tokenize(req):
                return TokenizedGenerateReqInput(
                    rid=req.rid,
                    input_text=req.text,
                    input_ids=array("q", [1]),
                    input_embeds=None,
                    mm_inputs=None,
                    token_type_ids=None,
                    sampling_params=SamplingParams(max_new_tokens=1),
                    return_logprob=False,
                    logprob_start_len=-1,
                    top_logprobs_num=0,
                    token_ids_logprob=None,
                    stream=True,
                    time_stats=tm.rid_to_state[req.rid].time_stats,
                )

            tm._should_use_batch_tokenization = Mock(return_value=True)
            tm._batch_tokenize_and_process = AsyncMock(
                side_effect=lambda size, req: [tokenize(req[i]) for i in range(size)]
            )

            async def publish(_mm_inputs):
                tm.abort_request(obj.rid[1])
                return []

            tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = publish
            responders = []

            def dispatch(packet):
                if isinstance(packet, BatchTokenizedGenerateReqInput):
                    self.assertEqual([r.rid for r in packet.batch], [obj.rid[0]])
                    responders.append(
                        asyncio.create_task(
                            tm._handle_batch_output(_make_batch_str_output(obj.rid[0]))
                        )
                    )

            tm._dispatch_to_scheduler = Mock(side_effect=dispatch)
            generator = tm.generate_request(obj)
            try:

                async def consume():
                    return [out async for out in generator]

                outputs = await asyncio.wait_for(consume(), 2)
                await asyncio.gather(*responders)
                self.assertEqual(
                    {out["meta_info"]["id"] for out in outputs}, set(obj.rid)
                )
                aborted = next(
                    out for out in outputs if out["meta_info"]["id"] == obj.rid[1]
                )
                self.assertEqual(aborted["meta_info"]["finish_reason"]["type"], "abort")
                self.assertFalse(tm.rid_to_state)
            finally:
                await generator.aclose()
                await asyncio.gather(*responders, return_exceptions=True)

        asyncio.run(drive())


class TestLoRARequestLifecycle(CustomTestCase):
    def test_abort_waiter_preserves_reused_rid(self):
        """An old abort must neither erase a new request nor release its reference."""

        async def drive(status_code, expected_exception):
            tm = _make_tm_for_generate(self, enable_lora=True)
            obj = GenerateReqInput(text="old", rid="reused", lora_path="adapter")
            tm._tokenize_one_request = AsyncMock(
                side_effect=lambda req: SimpleNamespace(rid=req.rid)
            )
            dispatched = asyncio.Event()

            async def send(tokenized):
                tm._mark_state_dispatched(tokenized.rid)
                dispatched.set()

            tm._send_one_request = send
            generator = tm.generate_request(obj)
            waiter = asyncio.create_task(generator.__anext__())
            await dispatched.wait()

            abort = _make_abort_req(obj.rid)
            abort.finished_reason["status_code"] = status_code
            tm._handle_abort_req(abort)
            # Reuse the ID before the old waiter resumes on its abort notification.
            replacement = GenerateReqInput(text="new", rid=obj.rid, lora_path="adapter")
            replacement.normalize_batch_and_arguments()
            tm._init_req_state(replacement)
            await tm._validate_and_resolve_lora(replacement)
            replacement_state = tm.rid_to_state[obj.rid]

            with self.assertRaises(expected_exception) as ctx:
                await waiter
            if expected_exception is HTTPException:
                self.assertEqual(ctx.exception.status_code, status_code)
            await asyncio.gather(*tm._lora_release_tasks)
            self.assertIs(tm.rid_to_state.get(obj.rid), replacement_state)
            self.assertEqual(tm.lora_registry._counters["adapter"].value(), 1)

            await tm._handle_batch_output(_make_batch_str_output(obj.rid))
            await asyncio.gather(*tm._lora_release_tasks)
            self.assertEqual(tm.lora_registry._counters["adapter"].value(), 0)
            await generator.aclose()

        for status_code, expected_exception in (
            (400, ValueError),
            (503, HTTPException),
        ):
            with self.subTest(status_code=status_code):
                asyncio.run(drive(status_code, expected_exception))

    async def _drive_parallel_request(self, fail_second_warmup=False):
        tm = _make_tm_for_generate(self, enable_lora=True)
        obj = GenerateReqInput(
            text=["first", "second"],
            lora_path="adapter",
            sampling_params={"n": 2, "max_new_tokens": 4},
        )
        tm._tokenize_one_request = AsyncMock(
            side_effect=lambda req: SimpleNamespace(
                rid=req.rid,
                lora_id=req.lora_id,
                input_ids=[1],
                mm_inputs=None,
                time_stats=tm.rid_to_state[req.rid].time_stats,
                sampling_params=SamplingParams(max_new_tokens=4),
            )
        )
        sent = asyncio.Queue()

        async def send(tokenized):
            tm._mark_state_dispatched(tokenized.rid)
            sent.put_nowait(tokenized)

        tm._send_one_request = send
        generator = tm.generate_request(obj)
        response = asyncio.create_task(generator.__anext__())
        try:
            for i in range(2):
                warmup = await asyncio.wait_for(sent.get(), timeout=1)
                self.assertEqual(warmup.lora_id, "adapter")
                self.assertEqual(warmup.sampling_params.max_new_tokens, 0)
                reason = None
                if fail_second_warmup and i == 1:
                    reason = {
                        "type": "abort",
                        "status_code": 400,
                        "message": "invalid grammar",
                    }
                await tm._handle_batch_output(
                    _make_batch_str_output(warmup.rid, reason)
                )

            if fail_second_warmup:
                with self.assertRaisesRegex(ValueError, "invalid grammar"):
                    await response
                self.assertTrue(sent.empty())  # No samples were dispatched.
            else:
                await asyncio.gather(*tm._lora_release_tasks)
                self.assertEqual(tm.lora_registry._counters["adapter"].value(), 4)
                for remaining in (3, 2, 1, 0):
                    sample = await asyncio.wait_for(sent.get(), timeout=1)
                    await tm._handle_batch_output(_make_batch_str_output(sample.rid))
                    await asyncio.gather(*tm._lora_release_tasks)
                    self.assertEqual(
                        tm.lora_registry._counters["adapter"].value(), remaining
                    )
                self.assertEqual(len(await response), 4)

            await asyncio.gather(*tm._lora_release_tasks)
            self.assertEqual(tm.lora_registry._counters["adapter"].value(), 0)
            self.assertFalse(tm.rid_to_state)
        finally:
            response.cancel()
            await asyncio.gather(response, return_exceptions=True)
            await generator.aclose()

    def test_parallel_warmups_do_not_release_sample_references(self):
        """Prefix warmups must not let an adapter unload while samples use it."""
        asyncio.run(self._drive_parallel_request())

    def test_second_parallel_warmup_failure_releases_all_references(self):
        """A failed warmup must return references even after earlier warmups finish."""
        asyncio.run(self._drive_parallel_request(fail_second_warmup=True))

    async def _drive_parallel_lora_failure(self, mode, pickle_ipc, fail_at=1):
        tm = _make_tm_for_generate(self, enable_lora=True)
        tm.disaggregation_mode = DisaggregationMode.NULL
        tm.model_update_lock = RWLock()
        tm.tokenizer_ipc_name = None
        tm.canary_info_hook = None
        tm.mm_processor = None
        tm.cuda_vmm_feature_transport = SimpleNamespace(
            prepare_for_dispatch_async=AsyncMock(return_value=[]),
            cancel_for_dispatch=Mock(),
        )
        tm.positional_embeds_transport = SimpleNamespace(
            prepare_for_dispatch=Mock(return_value=[]),
            cancel_for_dispatch=Mock(),
        )
        obj = GenerateReqInput(
            text=["prompt"],
            lora_path="adapter",
            sampling_params={"n": 3, "max_new_tokens": 1},
        )
        initial_rids = set()
        samples = []
        futures = []
        responders = []
        pending = asyncio.Event()
        messages = asyncio.Queue()

        def references():
            return tm.lora_registry._counters["adapter"].value()

        def tokenize(req):
            initial_rids.update(tm.rid_to_state)
            self.assertEqual(len(initial_rids), 3)
            self.assertEqual(references(), 3)
            return TokenizedGenerateReqInput(
                rid=req.rid,
                input_text=req.text,
                input_ids=array("q", [1]),
                input_embeds=None,
                mm_inputs=None,
                token_type_ids=None,
                sampling_params=SamplingParams(max_new_tokens=1),
                return_logprob=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
                lora_id=req.lora_id,
                time_stats=tm.rid_to_state[req.rid].time_stats,
            )

        tm._tokenize_one_request = AsyncMock(side_effect=tokenize)
        create_state = tm._init_req_state
        child_count = 0

        def create(req, request=None):
            nonlocal child_count
            if isinstance(req.rid, str) and req.lora_path:
                child_count += 1
                if mode == "create_failure" and child_count == fail_at:
                    self.assertTrue(initial_rids.issubset(tm.rid_to_state))
                    children = set(tm.rid_to_state) - initial_rids
                    self.assertEqual(len(children), fail_at - 1)
                    self.assertTrue(
                        all(tm.rid_to_state[rid].lora_released for rid in children)
                    )
                    self.assertEqual(references(), 3)
                    self.assertFalse(samples)
                    raise RuntimeError("sample state creation failed")
            return create_state(req, request)

        tm._init_req_state = create

        def dispatch(packet):
            future = asyncio.get_running_loop().create_future()
            futures.append(future)
            if isinstance(packet, AbortReq):
                tm._handle_abort_req(packet)
            else:
                packet.unwrap_pickle_fields()
                self.assertEqual(packet.lora_id, "adapter")
                if packet.sampling_params.max_new_tokens == 0:
                    responders.append(
                        asyncio.create_task(
                            tm._handle_batch_output(_make_batch_str_output(packet.rid))
                        )
                    )
                else:
                    samples.append(packet)
                    if mode == "send_failure" and len(samples) == fail_at:
                        future.cancel()
                        raise RuntimeError("sample enqueue failed")
                    if mode in ("cancel", "cancel_immediate") and len(samples) == 1:
                        asyncio.get_running_loop().call_soon(task.cancel)
                        if mode == "cancel":
                            return future
                    if mode == "disconnect" and len(samples) == 1:
                        messages.put_nowait({"type": "http.disconnect"})
                        pending.set()
                        return future
                    if mode == "finished_first":
                        if len(samples) == 1:
                            responders.append(
                                asyncio.create_task(
                                    tm._handle_batch_output(
                                        _make_batch_str_output(packet.rid)
                                    )
                                )
                            )
                        elif len(samples) == 2:
                            pending.set()
                            return future
            future.set_result(None)
            return future

        tm.send_to_scheduler = SimpleNamespace(
            send_pyobj=lambda packet, flags, protocol: dispatch(
                pickle.loads(pickle.dumps(packet, protocol=protocol))
            ),
            send=lambda data, flags: dispatch(io_struct.msgpack_decode(data)),
        )
        request = Request(
            {"type": "http", "method": "POST", "path": "/generate", "headers": []},
            receive=messages.get,
        )
        generator = tm.generate_request(obj, request)
        task = None
        try:
            with (
                patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                patch(
                    "sglang.srt.managers.tokenizer_manager._REQUEST_STATE_WAIT_TIMEOUT",
                    0.01,
                ),
            ):
                task = asyncio.create_task(anext(generator))
                if mode == "finished_first":
                    await asyncio.wait_for(pending.wait(), 3)
                    await asyncio.gather(*responders)
                    await asyncio.gather(*tm._lora_release_tasks)
                    self.assertNotIn(samples[0].rid, tm.rid_to_state)
                    self.assertEqual(references(), 2)
                    self.assertEqual(len(tm.rid_to_state), 2)
                    self.assertTrue(
                        all(not state.dispatched for state in tm.rid_to_state.values())
                    )
                    task.cancel()
                exception = {
                    "cancel": asyncio.CancelledError,
                    "cancel_immediate": asyncio.CancelledError,
                    "finished_first": asyncio.CancelledError,
                    "disconnect": ValueError,
                    "send_failure": RuntimeError,
                    "create_failure": RuntimeError,
                }[mode]
                with self.assertRaises(exception):
                    await asyncio.wait_for(task, 3)
                await asyncio.gather(*responders)
                await asyncio.gather(*tm._lora_release_tasks)
                self.assertEqual(references(), 0)
                self.assertFalse(tm.rid_to_state)
                self.assertEqual(tm.model_update_lock._readers, 0)
                expected_samples = {
                    "cancel": 1,
                    "cancel_immediate": None,
                    "disconnect": 1,
                    "send_failure": fail_at,
                    "create_failure": 0,
                    "finished_first": 2,
                }[mode]
                if expected_samples is None:
                    self.assertIn(len(samples), (1, 2, 3))
                else:
                    self.assertEqual(len(samples), expected_samples)
                self.assertTrue(all(future.done() for future in futures))
                if mode == "cancel":
                    self.assertTrue(futures[1].cancelled())
                elif mode == "cancel_immediate":
                    self.assertFalse(futures[1].cancelled())
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await generator.aclose()
            await asyncio.gather(*responders, return_exceptions=True)
            await asyncio.gather(*tm._lora_release_tasks, return_exceptions=True)
            for future in futures:
                future.cancel()

    def test_parallel_http_cancel_releases_unsent_sample_references(self):
        for pickle_ipc in (False, True):
            with self.subTest(pickle_ipc=pickle_ipc):
                asyncio.run(self._drive_parallel_lora_failure("cancel", pickle_ipc))

    def test_parallel_http_disconnect_releases_unsent_sample_references(self):
        for pickle_ipc in (False, True):
            with self.subTest(pickle_ipc=pickle_ipc):
                asyncio.run(self._drive_parallel_lora_failure("disconnect", pickle_ipc))

    def test_parallel_http_cancel_after_immediate_enqueue_releases_references(self):
        for pickle_ipc in (False, True):
            with self.subTest(pickle_ipc=pickle_ipc):
                asyncio.run(
                    self._drive_parallel_lora_failure("cancel_immediate", pickle_ipc)
                )

    def test_parallel_enqueue_failure_releases_all_sample_references(self):
        for pickle_ipc in (False, True):
            for fail_at in (1, 2):
                with self.subTest(pickle_ipc=pickle_ipc, fail_at=fail_at):
                    asyncio.run(
                        self._drive_parallel_lora_failure(
                            "send_failure", pickle_ipc, fail_at
                        )
                    )

    def test_parallel_state_creation_failure_keeps_initial_owners(self):
        for pickle_ipc in (False, True):
            for fail_at in (2, 3):
                with self.subTest(pickle_ipc=pickle_ipc, fail_at=fail_at):
                    asyncio.run(
                        self._drive_parallel_lora_failure(
                            "create_failure", pickle_ipc, fail_at
                        )
                    )

    def test_parallel_terminal_sample_preserves_unsent_sample_references(self):
        for pickle_ipc in (False, True):
            with self.subTest(pickle_ipc=pickle_ipc):
                asyncio.run(
                    self._drive_parallel_lora_failure("finished_first", pickle_ipc)
                )


class TestPriorityRange(CustomTestCase):
    """Request priorities reach the radix cache as signed 64-bit integers, so an
    out-of-range value is rejected before dispatch instead of failing inside
    the scheduler, and the boundary values themselves pass through unchanged."""

    INT64_MIN = -(1 << 63)
    INT64_MAX = (1 << 63) - 1

    def test_int64_boundaries_are_preserved(self):
        tm = _make_tokenizer_manager(self)
        tm.enable_priority_scheduling = True
        tm.default_priority_value = 0
        for priority in (self.INT64_MIN, self.INT64_MAX):
            with self.subTest(priority=priority):
                obj = GenerateReqInput(input_ids=[1], priority=priority)
                tm._set_default_priority(obj)
                self.assertEqual(obj.priority, priority)

    def test_out_of_range_priority_is_rejected_before_dispatch(self):
        for enabled in (False, True):
            for priority in (self.INT64_MIN - 1, self.INT64_MAX + 1):
                for source in ("body", "default"):
                    if source == "default" and not enabled:
                        continue
                    with self.subTest(
                        enabled=enabled, priority=priority, source=source
                    ):
                        tm = _make_tm_for_generate(self)
                        del tm._set_default_priority  # the real one is under test
                        tm.enable_priority_scheduling = enabled
                        tm.default_priority_value = (
                            priority if source == "default" else 0
                        )
                        tm._init_req_state = Mock()
                        obj = GenerateReqInput(
                            input_ids=[1],
                            priority=priority if source == "body" else None,
                        )
                        with self.assertRaisesRegex(ValueError, "signed 64-bit"):
                            asyncio.run(tm.generate_request(obj).__anext__())
                        tm._init_req_state.assert_not_called()


class TestChatDisconnectAbort(CustomTestCase):
    """Regression tests for Issue #46517: Disconnected chat requests must abort on scheduler."""

    def test_completed_stream_cleanup_preserves_prefix_related_request(self):
        """Finishing job must not abort job-1 when delayed response cleanup runs."""

        async def drive():
            tm = _make_tokenizer_manager(self)
            tm.server_args.tokenizer_worker_num = 1
            tm._dispatch_to_scheduler = Mock()
            short = _make_req_state("job")
            long = _make_req_state("job-1")
            for state in (short, long):
                state.obj.stream = True
                state.obj.is_single = True
                state.dispatched = True
                tm.rid_to_state[state.obj.rid] = state
            cleanup = tm.create_abort_task(short.obj)

            await tm._handle_batch_output(_make_batch_str_output("job"))
            self.assertNotIn("job", tm.rid_to_state)
            self.assertIs(tm.rid_to_state["job-1"], long)

            # Run the response's delayed background task after normal completion.
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await cleanup()
            self.assertEqual(tm._dispatch_to_scheduler.call_args_list, [])

            await tm._handle_batch_output(
                _make_batch_str_output("job-1", finished_reason=_NOT_FINISHED)
            )
            self.assertFalse(long.finished)
            self.assertFalse(long.abort_sent)
            self.assertIsNone(long.out_list[-1]["meta_info"]["finish_reason"])

            # Cleanup must still abort a request that has not finished.
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await tm.create_abort_task(long.obj)()
            sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
            self.assertEqual(len(sent), 1)
            self.assertIsInstance(sent[0], AbortReq)
            self.assertEqual(sent[0].rid, "job-1")
            self.assertTrue(long.abort_sent)

        asyncio.run(drive())

    def test_orphan_unfinished_output_sends_no_abort(self):
        """An output for an rid with no state must not abort by rid prefix.

        The scheduler matches AbortReq rids with startswith, so aborting "job"
        here would also stop a live "job-1".
        """
        tm = _make_tokenizer_manager(self)
        tm._dispatch_to_scheduler = Mock()
        live = _make_req_state("job-1")
        tm.rid_to_state["job-1"] = live

        asyncio.run(
            tm._handle_batch_output(
                _make_batch_str_output("job", finished_reason=_NOT_FINISHED)
            )
        )

        tm._dispatch_to_scheduler.assert_not_called()
        self.assertIs(tm.rid_to_state["job-1"], live)
        self.assertFalse(live.abort_sent)

    def test_streaming_disconnect_aborts_before_next_chunk(self):
        """A disconnected stream aborts on the next output instead of sending it."""
        tm = _make_tokenizer_manager(self)
        tm._dispatch_to_scheduler = Mock()
        tm.server_args.tokenizer_worker_num = 1
        state = _make_req_state("stream_disconnect_rid")
        state.obj.stream = True
        state.obj.background = False
        state.dispatched = True
        state.out_list = [{"text": "token1", "meta_info": {}}]
        state.event.set()
        tm.rid_to_state[state.obj.rid] = state

        request = Mock()
        request.is_disconnected = AsyncMock(return_value=True)
        sent_chunks = []

        async def drive():
            async for chunk in tm._stream_one_response(
                obj=state.obj, state=state, request=request
            ):
                sent_chunks.append(chunk)

        with self.assertRaisesRegex(ValueError, "type 3"):
            asyncio.run(drive())

        self.assertEqual(sent_chunks, [])
        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertEqual([m.rid for m in sent], [state.obj.rid])
        self.assertTrue(state.abort_sent)

    def _make_streaming_state(self, tm, rid):
        state = _make_req_state(rid)
        state.obj.stream = True
        state.obj.background = False
        state.dispatched = True
        tm.rid_to_state[rid] = state
        return state

    @staticmethod
    async def _next_chunk(stream, state, i):
        state.out_list.append({"text": f"token{i}", "meta_info": {}})
        state.event.set()
        return await anext(stream)

    def test_stream_polls_disconnect_once_per_wait_timeout(self):
        """A stream must not poll is_disconnected() for every output.

        Each poll costs an event-loop pass; polling per output cut 8k/1k
        concurrency-256 streaming throughput by 10-25%.
        """
        tm = _make_tokenizer_manager(self)
        state = self._make_streaming_state(tm, "stream_poll_rid")
        request = Mock()
        request.is_disconnected = AsyncMock(return_value=False)

        async def drive():
            stream = tm._stream_one_response(
                obj=state.obj, state=state, request=request
            )
            for i in range(8):
                await self._next_chunk(stream, state, i)
            await stream.aclose()

        with patch(
            "sglang.srt.managers.tokenizer_manager._REQUEST_STATE_WAIT_TIMEOUT", 3600
        ):
            asyncio.run(drive())
        request.is_disconnected.assert_awaited_once()

    def test_stream_polls_again_once_wait_timeout_elapses(self):
        """A stream that keeps producing outputs still aborts when a later
        poll finds its client gone."""
        tm = _make_tokenizer_manager(self)
        tm._dispatch_to_scheduler = Mock()
        tm.server_args.tokenizer_worker_num = 1
        state = self._make_streaming_state(tm, "stream_repoll_rid")
        request = Mock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])

        async def drive():
            stream = tm._stream_one_response(
                obj=state.obj, state=state, request=request
            )
            await self._next_chunk(stream, state, 0)
            await asyncio.sleep(0.01)
            await self._next_chunk(stream, state, 1)

        with patch(
            "sglang.srt.managers.tokenizer_manager._REQUEST_STATE_WAIT_TIMEOUT", 0.001
        ):
            with self.assertRaisesRegex(ValueError, "type 3"):
                asyncio.run(drive())
        self.assertEqual(request.is_disconnected.await_count, 2)
        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertEqual([m.rid for m in sent], [state.obj.rid])
        self.assertTrue(state.abort_sent)


class TestChatStreamDisconnect(CustomTestCase):
    """A /v1/chat/completions stream that loses its client must stop its own
    scheduler request, and only that one."""

    @staticmethod
    def _make_handler(tm):
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        tm.config_value = Mock(return_value=None)
        tm.model_config = Mock(is_multimodal=False)
        tm.model_config.get_default_sampling_params.return_value = {}
        tm.model_config.hf_config = Mock(architectures=["LlamaForCausalLM"])
        tm.model_config.hf_config.to_dict.return_value = {}
        tm.tokenizer = Mock(chat_template=None)
        tm.served_model_name = "m"
        return OpenAIServingChat(tm, Mock())

    def _make_tm(self):
        tm = _make_tm_for_generate(self)
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = AsyncMock(
            return_value=[]
        )
        tm.positional_embeds_transport = Mock()
        tm._dispatch_to_scheduler = Mock()

        async def tokenize(obj):
            tokenized = MagicMock()
            tokenized.rid = obj.rid
            tokenized.mm_inputs = None
            return tokenized

        tm._tokenize_one_request = tokenize
        return tm

    @staticmethod
    def _make_obj(rid):
        obj = _make_generate_obj(rid, is_single=True)
        obj.return_prompt_token_ids = False
        obj.stream = True
        obj.background = False
        obj.return_logprob = False
        obj.lora_path = None
        obj.log_metrics = False
        return obj

    @staticmethod
    def _make_raw_request():
        raw = Mock()
        raw.headers = {}
        raw.is_disconnected = AsyncMock(return_value=False)
        return raw

    async def _start_stream(self, tm, rid):
        """Return the StreamingResponse once the first chunk is ready."""
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        handler = self._make_handler(tm)
        request = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        task = asyncio.create_task(
            handler._handle_streaming_request(
                self._make_obj(rid), request, self._make_raw_request()
            )
        )
        for _ in range(100):
            await asyncio.sleep(0)
            if rid in tm.rid_to_state and tm.rid_to_state[rid].dispatched:
                break
        self.assertTrue(tm.rid_to_state[rid].dispatched)
        await tm._handle_batch_output(
            _make_batch_str_output(rid, finished_reason=_NOT_FINISHED)
        )
        return await task

    @staticmethod
    def _aborted_rids(tm):
        return [
            c.args[0].rid
            for c in tm._dispatch_to_scheduler.call_args_list
            if isinstance(c.args[0], AbortReq)
        ]

    def test_send_failure_aborts_request_and_releases_state(self):
        """A failed write must abort the request even though Starlette skips
        the background task on ASGI spec >= 2.4."""
        from starlette.requests import ClientDisconnect

        async def drive():
            tm = self._make_tm()
            response = await self._start_stream(tm, "chat")

            async def send(message):
                if message["type"] == "http.response.body":
                    raise OSError("connection reset")

            scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
            with self.assertRaises(ClientDisconnect):
                await response(scope, AsyncMock(), send)

            self.assertEqual(self._aborted_rids(tm), ["chat"])
            # State is kept until the scheduler confirms the abort.
            self.assertTrue(tm.rid_to_state["chat"].abort_sent)
            tm._handle_abort_req(_make_abort_req("chat"))
            self.assertNotIn("chat", tm.rid_to_state)

        asyncio.run(drive())

    def test_client_disconnect_aborts_running_request(self):
        """An http.disconnect while waiting for tokens aborts the request."""

        async def drive():
            tm = self._make_tm()
            response = await self._start_stream(tm, "chat")
            first_chunk_sent = asyncio.Event()

            async def send(message):
                if message["type"] == "http.response.body":
                    first_chunk_sent.set()

            async def receive():
                await first_chunk_sent.wait()
                return {"type": "http.disconnect"}

            scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await response(scope, receive, send)

            self.assertEqual(self._aborted_rids(tm), ["chat"])
            self.assertTrue(tm.rid_to_state["chat"].abort_sent)

        asyncio.run(drive())

    def test_closing_finished_stream_spares_prefix_related_request(self):
        """Closing a stream whose request already finished must not abort
        "job-1", which the scheduler would match against rid "job"."""
        from starlette.requests import ClientDisconnect

        async def drive():
            tm = self._make_tm()
            response = await self._start_stream(tm, "job")
            live = _make_req_state("job-1")
            live.dispatched = True
            tm.rid_to_state["job-1"] = live
            await tm._handle_batch_output(_make_batch_str_output("job"))
            self.assertNotIn("job", tm.rid_to_state)

            body_chunks = 0

            async def send(message):
                nonlocal body_chunks
                if message["type"] == "http.response.body":
                    body_chunks += 1
                    if body_chunks == 2:
                        raise OSError("connection reset")

            scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
            with self.assertRaises(ClientDisconnect):
                await response(scope, AsyncMock(), send)

            self.assertEqual(self._aborted_rids(tm), [])
            self.assertIs(tm.rid_to_state["job-1"], live)
            self.assertFalse(live.abort_sent)

        asyncio.run(drive())


class TestReadsInputIdsLate(CustomTestCase):
    def _manager(self, *, log=False, level=2, exporter=False, dump="", crash=None):
        return SimpleNamespace(
            request_logger=RequestLogger(
                log_requests=log,
                log_requests_level=level,
                log_requests_format="text",
                log_requests_target=None,
            ),
            request_metrics_exporter_manager=Mock(
                exporter_enabled=Mock(return_value=exporter)
            ),
            dump_requests_folder=dump,
            crash_dump_folder=crash,
        )

    def test_prompt_is_kept_only_while_something_reads_it(self):
        cases = [
            ({}, False),
            ({"log": True, "level": 0}, False),
            ({"log": True, "level": 1}, False),
            ({"log": True, "level": 2}, True),
            ({"log": True, "level": 3}, True),
            ({"exporter": True}, True),
            ({"dump": "/tmp/dump"}, True),
            ({"crash": "/tmp/crash"}, True),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                manager = self._manager(**kwargs)
                self.assertEqual(
                    TokenizerManager._reads_input_ids_late(manager), expected
                )

    def test_generate_request_releases_prompt_after_dispatch(self):
        for dump, released in (("", True), ("/tmp/dump", False)):
            with self.subTest(dump=dump):
                tm = _make_tm_for_generate(self)
                tm.dump_requests_folder = dump
                obj = GenerateReqInput(input_ids=[1, 2, 3], rid="r", sampling_params={})
                tokenized = MagicMock(input_ids=[1, 2, 3], mm_inputs=None)
                tm._tokenize_one_request = AsyncMock(return_value=tokenized)
                tm._send_one_request = AsyncMock()

                async def wait_one_response(obj, request, *, state):
                    self.assertIs(state, tm.rid_to_state[obj.rid])
                    yield {
                        "input_ids": obj.input_ids,
                        "released": state.prompt_released,
                    }

                tm._wait_one_response = wait_one_response

                async def drive():
                    return await tm.generate_request(obj).__anext__()

                out = asyncio.run(drive())
                tm._send_one_request.assert_awaited_once()
                self.assertEqual(out["input_ids"] is None, released)
                self.assertEqual(out["released"], released)

    def test_dumps_turned_on_later_skip_released_prompts(self):
        """Dumps enabled at runtime must not record a request whose prompt is gone."""
        for released in (False, True):
            with self.subTest(released=released):
                tm = _make_tokenizer_manager(self)
                tm.dump_requests_folder = "/tmp/dump"
                tm.crash_dump_folder = "/tmp/crash"
                tm.dump_requests = Mock()
                tm.record_request_for_crash_dump = Mock()
                state = _make_req_state("released_rid")
                state.obj.log_metrics = True
                state.prompt_released = released
                tm.rid_to_state["released_rid"] = state
                asyncio.run(
                    tm._handle_batch_output(_make_batch_str_output("released_rid"))
                )
                self.assertEqual(tm.dump_requests.called, not released)
                self.assertEqual(tm.record_request_for_crash_dump.called, not released)


if __name__ == "__main__":
    unittest.main(verbosity=2)
