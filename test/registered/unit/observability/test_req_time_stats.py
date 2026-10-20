"""Unit tests for ReqTimeStats.

ReqTimeStatsBase.__setstate__ rebases perf_counter fields onto the receiving
process's clock anchor. Rebasing a field that was never stamped (0.0) turns
the sentinel into a tiny epsilon (sender_diff - receiver_diff), which defeats
== 0.0 / > 0.0 "was this stamped?" checks downstream. Concretely, a PD decode
server never stamps prefill_finished_time locally; if the sentinel arrives at
the tokenizer as an epsilon, first-token bookkeeping mistakes it for a real
stamp and the TTFT / inter-token-latency histograms record ~node-uptime-sized
garbage samples.

The same "was this stamped?" rule governs the gen_ai.latency.* span attrs: the
ones derived from finished_time are only emitted once it is stamped, and
set_finished_time() closes the trace root span, so they have to be derived
inside that call rather than by the caller.
"""

import contextlib
import pickle
import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.observability.req_time_stats as rts
import sglang.srt.observability.trace as trace_mod
import sglang.srt.observability.trace_async as trace_async
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.observability.trace import SpanAttributes
from sglang.srt.runtime_context import get_parallel, get_resources
from sglang.srt.sampling.sampling_params import SamplingParams

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import StatusCode

    _has_otel = True
except ImportError:
    _has_otel = False

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSetstatePreservesUnsetTimeSentinels(CustomTestCase):
    def test_two_hop_round_trip(self):
        src = rts.SchedulerReqTimeStats()
        src.enable_metrics = True
        src.wait_queue_entry_time = 123.456
        src.prefill_finished_time = 0.0

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(src)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            hop1 = pickle.loads(blob)
            blob2 = pickle.dumps(hop1)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            hop2 = pickle.loads(blob2)

        self.assertEqual(hop2.prefill_finished_time, 0.0)
        self.assertAlmostEqual(hop2.wait_queue_entry_time, 123.456 - 9.0)


class TestOutputMetaInfo(CustomTestCase):
    def test_first_token_latency_requires_valid_timestamps(self):
        for created, first, expected in (
            (1.0, 1.5, 0.5),
            (0.0, 1.5, None),
            (1.0, 0.0, None),
            (1.0, 1.0, None),
            (1.0, 0.5, None),
        ):
            with self.subTest(created=created, first=first):
                stats = rts.APIServerReqTimeStats()
                stats.created_time = created
                stats.first_token_time = first
                meta_info = stats.convert_to_output_meta_info()
                if expected is None:
                    self.assertNotIn("first_token_latency", meta_info)
                else:
                    self.assertAlmostEqual(meta_info["first_token_latency"], expected)

    def test_tpot_excludes_first_token_from_denominator(self):
        stats = rts.APIServerReqTimeStats()
        stats.created_time = 1.0
        stats.first_token_time = 1.5
        stats.finished_time = 3.5
        # 101 completion tokens, 100 of them in the 2.0s decode interval.
        self.assertAlmostEqual(stats.get_time_per_output_token(101), 0.02)
        self.assertAlmostEqual(stats.get_decode_throughput(101), 50.0)
        meta_info = stats.convert_to_output_meta_info(completion_tokens=101)
        self.assertAlmostEqual(meta_info["decode_throughput"], 50.0)

    def test_tpot_undefined_cases(self):
        for first, finished, completion_tokens in (
            # Single-batch finish: first token stamped in the same batch.
            (3.5, 3.5, 40),
            # First token never stamped.
            (0.0, 3.5, 40),
            # One or zero output tokens: no decode interval to measure.
            (1.5, 3.5, 1),
            (1.5, 3.5, 0),
        ):
            with self.subTest(
                first=first, finished=finished, completion_tokens=completion_tokens
            ):
                stats = rts.APIServerReqTimeStats()
                stats.created_time = 1.0
                stats.first_token_time = first
                stats.finished_time = finished
                self.assertIsNone(stats.get_time_per_output_token(completion_tokens))
                self.assertIsNone(stats.get_decode_throughput(completion_tokens))
                meta_info = stats.convert_to_output_meta_info(
                    completion_tokens=completion_tokens
                )
                self.assertNotIn("decode_throughput", meta_info)


class TestConvertToGenAiSpanAttrs(CustomTestCase):
    def _stats_after_first_token(self) -> rts.APIServerReqTimeStats:
        stats = rts.APIServerReqTimeStats()
        stats.created_time = 1.0
        stats.api_server_dispatch_finish_time = 1.1
        stats.first_token_time = 1.5
        return stats

    def test_prefill_without_finished_time_omits_e2e_and_decode(self):
        attrs = self._stats_after_first_token().convert_to_gen_ai_span_attrs()
        self.assertIn(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL, attrs)
        self.assertIn(SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN, attrs)
        self.assertNotIn(SpanAttributes.GEN_AI_LATENCY_E2E, attrs)
        self.assertNotIn(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE, attrs)
        self.assertNotIn(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE, attrs)

    def test_finished_time_populates_e2e_and_decode(self):
        stats = self._stats_after_first_token()
        stats.finished_time = 2.0
        attrs = stats.convert_to_gen_ai_span_attrs()
        self.assertAlmostEqual(attrs[SpanAttributes.GEN_AI_LATENCY_E2E], 1.0)
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE], 0.5
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL], 0.4
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE], 0.9
        )


class TestSetFinishedTimeSpanAttrs(CustomTestCase):
    def _tracing_stats(self) -> rts.APIServerReqTimeStats:
        stats = rts.APIServerReqTimeStats()
        stats.trace_ctx = mock.MagicMock()
        stats.trace_ctx.tracing_enable = True
        return stats

    def test_passes_caller_attrs_into_trace_req_finish(self):
        stats = self._tracing_stats()

        stats.set_finished_time(ts=1.25, span_attrs={"gen_ai.request.id": "rid-1"})

        self.assertEqual(stats.finished_time, 1.25)
        stats.trace_ctx.trace_req_finish.assert_called_once_with(
            mock.ANY, attrs={"gen_ai.request.id": "rid-1"}
        )

    def test_merges_latency_attrs_derived_from_finished_time(self):
        stats = self._tracing_stats()
        stats.created_time = 1.0
        stats.api_server_dispatch_finish_time = 1.1
        stats.first_token_time = 1.5

        stats.set_finished_time(ts=2.0, span_attrs={"gen_ai.request.id": "rid-1"})

        attrs = stats.trace_ctx.trace_req_finish.call_args.kwargs["attrs"]
        self.assertEqual(attrs["gen_ai.request.id"], "rid-1")
        self.assertAlmostEqual(attrs[SpanAttributes.GEN_AI_LATENCY_E2E], 1.0)
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE], 0.5
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE], 0.9
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL], 0.4
        )

    def test_caller_attrs_not_mutated(self):
        stats = self._tracing_stats()
        stats.created_time = 1.0
        caller_attrs = {"gen_ai.request.id": "rid-1"}

        stats.set_finished_time(ts=2.0, span_attrs=caller_attrs)

        self.assertEqual(caller_attrs, {"gen_ai.request.id": "rid-1"})


class TestQueueTimeAcrossRetraction(CustomTestCase):
    """A retracted request re-enters the waiting queue after its first forward.

    queue_time must be the total time spent waiting across every entry. It used
    to be first-forward minus the latest queue entry, which went negative.
    """

    T0 = 1000.0

    def _stats(self, disagg_mode=rts.DisaggregationMode.NULL):
        stats = rts.SchedulerReqTimeStats(disagg_mode=disagg_mode)
        stats.enable_metrics = True
        stats.metrics_collector = mock.MagicMock()
        return stats

    def _retract_once(self, stats, forward_calls=(0.10,), readmit_calls=(1.50,)):
        # waits: [0.00, 0.10) before the first forward, [1.00, 1.50) after retraction
        stats.set_wait_queue_entry_time(self.T0 + 0.00)
        for dt in forward_calls:
            stats.set_forward_entry_time(self.T0 + dt)
        stats.set_wait_queue_entry_time(self.T0 + 1.00)
        for dt in readmit_calls:
            stats.set_forward_entry_time(self.T0 + dt)
        stats.set_completion_time(self.T0 + 2.00)

    def test_retracted_request_reports_total_queue_time(self):
        stats = self._stats()
        self._retract_once(stats)

        self.assertAlmostEqual(stats.convert_to_output_meta_info()["queue_time"], 0.6)
        with mock.patch.object(rts, "SGLANG_TEST_REQUEST_TIME_STATS", True):
            self.assertIn("queue_duration=600.00ms", stats.convert_to_duration())

    def test_queue_time_survives_clock_rebase_across_hops(self):
        stats = self._stats()
        self._retract_once(stats)

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(stats)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            blob = pickle.dumps(pickle.loads(blob))
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            tokenizer_side = pickle.loads(blob)

        self.assertAlmostEqual(tokenizer_side.get_queueing_time(), 0.6)

    def test_prefill_chunks_do_not_extend_queue_time(self):
        stats = self._stats()
        self._retract_once(
            stats, forward_calls=(0.10, 0.30, 0.50), readmit_calls=(1.50, 1.70)
        )

        self.assertAlmostEqual(stats.get_queueing_time(), 0.6)

    def test_queue_time_histogram_keeps_one_first_wait_sample(self):
        stats = self._stats()
        self._retract_once(stats)

        stats.metrics_collector.observe_queue_time.assert_called_once()
        (sample,) = stats.metrics_collector.observe_queue_time.call_args.args
        self.assertAlmostEqual(sample, 0.1)

    def test_decode_rebootstrap_counts_both_waits(self):
        stats = self._stats(disagg_mode=rts.DisaggregationMode.DECODE)
        stats.set_wait_queue_entry_time(self.T0 + 0.00)
        stats.set_forward_entry_time(self.T0 + 0.10)
        # held for re-bootstrap: not in the waiting queue until [1.00, 1.25)
        stats.set_retract_time(self.T0 + 0.50)
        stats.set_wait_queue_entry_time(self.T0 + 1.00)
        stats.set_forward_entry_time(self.T0 + 1.25)
        stats.set_quick_finish_time(self.T0 + 1.30)

        self.assertAlmostEqual(stats.get_queueing_time(), 0.35)

    def test_prefill_retry_reset_discards_earlier_attempt_waits(self):
        stats = self._stats(disagg_mode=rts.DisaggregationMode.PREFILL)
        stats.set_wait_queue_entry_time(self.T0 + 0.00)
        stats.set_forward_entry_time(self.T0 + 0.30)
        stats.reset_prefill_retry_time()
        stats.set_wait_queue_entry_time(self.T0 + 1.00)
        stats.set_forward_entry_time(self.T0 + 1.20)

        self.assertAlmostEqual(stats.get_queueing_time(), 0.2)


class TestStorageAdmissionWait(CustomTestCase):
    def _stats(self):
        stats = rts.SchedulerReqTimeStats(
            enable_metrics=True, metrics_collector=mock.MagicMock()
        )
        stats.set_wait_queue_entry_time(100.0)
        return stats

    def test_blocked_polls_accumulate_wait_episodes(self):
        stats = self._stats()
        for observe, output_key in (
            (stats.observe_storage_prefetch_ready, "storage_prefetch_wait"),
            (stats.observe_storage_staging_ready, "storage_staging_wait"),
        ):
            with (
                self.subTest(gate=output_key),
                mock.patch.object(rts.time, "perf_counter") as clock,
            ):
                for ts, ready in (
                    (100.0, True),
                    (100.1, False),
                    (100.15, False),
                    (100.2, False),
                    (100.3, True),
                    (100.4, True),
                    (100.5, False),
                    (100.6, True),
                ):
                    clock.return_value = ts
                    observe(ready)
                    if not ready:
                        self.assertNotIn(
                            output_key, stats.convert_to_output_meta_info()
                        )
                self.assertAlmostEqual(
                    stats.convert_to_output_meta_info()[output_key], 0.3
                )

    def test_zero_missing_and_open_intervals_survive_serialization(self):
        stats = self._stats()
        stats.observe_storage_prefetch_ready(True)
        stats.observe_storage_staging_ready(True)
        zero = pickle.loads(pickle.dumps(stats)).convert_to_output_meta_info()
        self.assertEqual(zero["storage_prefetch_wait"], 0.0)
        self.assertEqual(zero["storage_staging_wait"], 0.0)

        with mock.patch.object(rts.time, "perf_counter", return_value=100.1):
            stats.observe_storage_prefetch_ready(False)
            stats.observe_storage_staging_ready(False)
        incomplete = pickle.loads(pickle.dumps(stats)).convert_to_output_meta_info()
        self.assertNotIn("storage_prefetch_wait", incomplete)
        self.assertNotIn("storage_staging_wait", incomplete)

        legacy = rts.SchedulerReqTimeStats()
        legacy.__setstate__({"has_timing_data": True, "queue_duration_s": 1.0})
        missing = pickle.loads(pickle.dumps(legacy)).convert_to_output_meta_info()
        self.assertNotIn("storage_prefetch_wait", missing)
        self.assertNotIn("storage_staging_wait", missing)

    def test_overlap_retraction_clock_rebase_and_retry(self):
        stats = self._stats()
        with mock.patch.object(
            rts.time, "perf_counter", side_effect=[100.1, 100.2, 100.4, 100.5]
        ):
            stats.observe_storage_staging_ready(False)
            stats.observe_storage_prefetch_ready(False)
            stats.observe_storage_prefetch_ready(True)
            stats.observe_storage_staging_ready(True)
        stats.set_forward_entry_time(100.6)
        stats.set_wait_queue_entry_time(101.0)
        with mock.patch.object(rts.time, "perf_counter", side_effect=[101.1, 101.4]):
            stats.observe_storage_prefetch_ready(False)
            stats.observe_storage_prefetch_ready(True)
        stats.set_forward_entry_time(101.5)

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(stats)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            blob = pickle.dumps(pickle.loads(blob))
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            meta = pickle.loads(blob).convert_to_output_meta_info()
        self.assertAlmostEqual(meta["storage_prefetch_wait"], 0.5)
        self.assertAlmostEqual(meta["storage_staging_wait"], 0.4)
        self.assertAlmostEqual(meta["queue_time"], 1.1)

        with mock.patch.object(rts.time, "perf_counter", return_value=102.0):
            stats.observe_storage_staging_ready(False)
        stats.reset_prefill_retry_time()
        stats.set_wait_queue_entry_time(103.0)
        meta = stats.convert_to_output_meta_info()
        self.assertEqual(meta["storage_prefetch_wait"], 0.0)
        self.assertEqual(meta["storage_staging_wait"], 0.0)

    def test_metrics_disabled_and_decode_do_not_record_local_prefetch(self):
        for stats in (
            rts.SchedulerReqTimeStats(),
            rts.SchedulerReqTimeStats(
                enable_metrics=True,
                metrics_collector=mock.MagicMock(),
                disagg_mode=rts.DisaggregationMode.DECODE,
            ),
        ):
            stats.set_wait_queue_entry_time(100.0)
            stats.observe_storage_prefetch_ready(False)
            stats.observe_storage_prefetch_ready(True)
            stats.observe_storage_staging_ready(False)
            stats.observe_storage_staging_ready(True)
            meta = stats.convert_to_output_meta_info()
            self.assertNotIn("storage_prefetch_wait", meta)
            self.assertNotIn("storage_staging_wait", meta)


class _RecordingSocket:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, msg, flags=0):
        self.sent.append(msg)


class _RecordingScheduler(Scheduler):
    def _create_generation_req(self, recv_req, *args, **kwargs):
        self.req = super()._create_generation_req(recv_req, *args, **kwargs)
        return self.req


@unittest.skipUnless(_has_otel, "opentelemetry not installed")
class TestExternalTraceHeaderParentsSchedulerSpans(CustomTestCase):
    """A frontend that sends a W3C trace context but no time_stats (the Rust
    server) gets the scheduler's spans in the caller's trace, the thread span a
    direct child of the caller's span, and no SGLang root span; the async
    exporter, which learns the root only from the ops it is sent, agrees.
    Requests enter through Scheduler.handle_generate_request, which picks the
    seed."""

    TRACE_ID = 0x0AF7651916CD43DD8448EB211C80319C
    SPAN_ID = 0xB7AD6B7169203331
    BOOTSTRAP_ROOM = 0x2A
    HEADER = {
        "traceparent": f"00-{TRACE_ID:032x}-{SPAN_ID:016x}-01",
        "tracestate": "v=1",
    }

    def setUp(self):
        self.orig_initialized = trace_mod.opentelemetry_initialized
        self.orig_tracer = trace_mod.tracer
        self.orig_threads = trace_mod.threads_info.copy()
        self.orig_level = get_resources().trace_level

        otel_trace._TRACER_PROVIDER_SET_ONCE._done = False
        otel_trace._TRACER_PROVIDER = None
        # The async exporter presets span ids, which only this generator consumes.
        provider = TracerProvider(id_generator=trace_mod.TraceCustomIdGenerator())
        self.exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        otel_trace.set_tracer_provider(provider)
        trace_mod.opentelemetry_initialized = True
        trace_mod.tracer = otel_trace.get_tracer("test")
        trace_mod.set_global_trace_level(3)

    def tearDown(self):
        trace_mod.opentelemetry_initialized = self.orig_initialized
        trace_mod.tracer = self.orig_tracer
        trace_mod.threads_info.clear()
        trace_mod.threads_info.update(self.orig_threads)
        get_resources().trace_level = self.orig_level

    def _seed(self, external_trace_header):
        return rts.time_stats_from_external_trace_header(
            rid="req-1",
            bootstrap_room=None,
            disagg_mode=rts.DisaggregationMode.NULL,
            external_trace_header=external_trace_header,
        )

    def _admit(
        self,
        *,
        time_stats=None,
        disagg_mode=rts.DisaggregationMode.NULL,
        bootstrap_room=None,
    ):
        """Build the request's Req through Scheduler.handle_generate_request,
        from a header the Rust server sends with time_stats=None by default."""
        scheduler = Scheduler.__new__(_RecordingScheduler)
        scheduler.enable_session_radix_cache = False
        scheduler.disaggregation_mode = disagg_mode
        scheduler.transfer_backend = TransferBackend.MOONCAKE
        scheduler.model_config = SimpleNamespace(hf_eos_token_id=None, vocab_size=128)
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.kv_events_publisher = SimpleNamespace(enable_kv_cache_events=False)
        scheduler.tokenizer = None
        scheduler.dllm_config = None
        scheduler._maybe_namespace_elastic_radix_cache = lambda req: None
        scheduler.init_req_max_new_tokens = lambda req: None
        scheduler._add_request_to_queue = lambda req: None
        scheduler.output_streamer = mock.MagicMock()
        recv = TokenizedGenerateReqInput(
            rid="req-1",
            input_text=None,
            input_ids=array("q", [1, 2, 3]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=4),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            bootstrap_port=1,
            bootstrap_room=bootstrap_room,
            time_stats=time_stats,
            external_trace_header=self.HEADER,
        )
        # An mm_input_error ends admission right after the Req is queued.
        with get_parallel().override(tp_rank=0):
            scheduler.handle_generate_request(recv, mm_input_error="stop")
        return scheduler

    def _run_scheduler_side(self):
        stats = self._admit(bootstrap_room=self.BOOTSTRAP_ROOM).req.time_stats
        stats.trace_ctx.trace_slice_start("prefill_forward", level=1)
        stats.trace_ctx.trace_slice_end("prefill_forward", level=1)
        stats.set_completion_time()

    def _assert_parented_on_header(self):
        spans = {span.name: span for span in self.exporter.get_finished_spans()}
        self.assertEqual(len(spans), 2, list(spans))
        slice_span = spans.pop("prefill_forward")
        (thread_span,) = spans.values()
        self.assertEqual(thread_span.context.trace_id, self.TRACE_ID)
        self.assertEqual(slice_span.context.trace_id, self.TRACE_ID)
        self.assertEqual(thread_span.parent.span_id, self.SPAN_ID)
        self.assertTrue(thread_span.parent.is_remote)
        self.assertEqual(slice_span.parent.span_id, thread_span.context.span_id)
        # No SGLang span above it carries the ids a trace backend filters by.
        self.assertEqual(thread_span.attributes["rid"], "req-1")
        self.assertEqual(thread_span.attributes["bootstrap_room"], "0x2a")

    def test_sync_tracing(self):
        self._run_scheduler_side()
        self._assert_parented_on_header()

    @contextlib.contextmanager
    def _async_tracing(self):
        socket = _RecordingSocket()
        with (
            mock.patch.object(rts, "is_async_tracing_available", return_value=True),
            mock.patch.object(
                trace_async, "is_async_tracing_available", return_value=True
            ),
            mock.patch.object(trace_async, "_get_zmq_socket", return_value=socket),
        ):
            yield socket

    def test_async_tracing_replays_to_the_same_tree(self):
        with self._async_tracing() as socket:
            self._run_scheduler_side()
        self.assertEqual(len(self.exporter.get_finished_spans()), 0)

        exporter = trace_async._TraceExporterProcess.__new__(
            trace_async._TraceExporterProcess
        )
        contexts = {}
        for msg in socket.sent:
            exporter._replay_batch(
                msg,
                contexts,
                trace_mod.threads_info,
                trace_mod.TraceCustomIdGenerator,
                trace_mod.TraceReqContext,
                trace_mod.TraceSliceContext,
                trace_mod.TraceEvent,
            )
        self._assert_parented_on_header()

    def _python_frontend_time_stats(self):
        # What the tokenizer pickles to the scheduler for the same request.
        stats = rts.APIServerReqTimeStats()
        stats.init_trace_ctx("req-1", None, self.HEADER)
        stats.trace_ctx.trace_req_start()
        return pickle.loads(pickle.dumps(stats))

    def test_frontend_time_stats_take_precedence_over_the_header(self):
        """The Python frontend's time_stats already parent on the caller under
        its root span; the scheduler must not re-seed them from the header."""
        sent = self._python_frontend_time_stats()
        req = self._admit(time_stats=sent).req
        self.assertIs(req.time_stats.trace_ctx, sent.trace_ctx)
        # The frontend's root span carries the ids, so its thread spans do not.
        req.time_stats.trace_ctx.abort()
        thread_spans = self.exporter.get_finished_spans()
        self.assertTrue(thread_spans)
        for span in thread_spans:
            self.assertNotIn("rid", span.attributes)

    def test_pd_reject_without_bootstrap_room_closes_the_seeded_trace(self):
        """A PD request without a bootstrap room gets a 400 abort. With no
        frontend time_stats the refusal must end the seeded thread span with
        the reason rather than fail on a missing trace context."""
        scheduler = self._admit(disagg_mode=rts.DisaggregationMode.PREFILL)
        req = scheduler.req
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
        self.assertEqual(req.finished_reason.status_code, HTTPStatus.BAD_REQUEST)
        (thread_span,) = self.exporter.get_finished_spans()
        self.assertEqual(thread_span.parent.span_id, self.SPAN_ID)
        self.assertEqual(thread_span.status.status_code, StatusCode.ERROR)
        self.assertIn("without bootstrap room id", thread_span.attributes["reason"])
        self.assertEqual(thread_span.attributes["rid"], "req-1")
        self.assertNotIn("bootstrap_room", thread_span.attributes)

    def test_no_seed_without_a_valid_context_or_tracing(self):
        # A context the propagator rejects would give each rank its own root.
        self.assertIsNone(self._seed({"traceparent": "garbage"}))
        self.assertIsNone(self._seed({"traceparent": f"00-{0:032x}-{0:016x}-01"}))
        trace_mod.opentelemetry_initialized = False
        self.assertIsNone(self._seed(self.HEADER))


if __name__ == "__main__":
    unittest.main()
