"""Unit tests for DeferredOutputSource: holding finished requests' responses."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, Mock, call, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class Source:
    """Holds requests until the test marks them ready."""

    def __init__(self):
        self.held = []
        self.ready = []

    def hold(self, *reqs):
        for req in reqs:
            req.defer_output = True
        self.held.extend(reqs)

    def poll(self):
        ready, self.ready = self.ready, []
        self.held = [req for req in self.held if req not in ready]
        return ready

    def has_pending(self):
        return bool(self.held)


def _req(rid, return_logprob=False):
    return SimpleNamespace(rid=rid, defer_output=False, return_logprob=return_logprob)


def _streamer(defer_outputs):
    streamer = object.__new__(SchedulerOutputStreamer)
    streamer.is_generation = True
    streamer.defer_outputs = defer_outputs
    return streamer


def test_streamer_skips_held_requests():
    held, free = _req("held"), _req("free")
    held.defer_output = True

    with patch.object(SchedulerOutputStreamer, "_stream_output_generation") as stream:
        _streamer(defer_outputs=True).stream_output([held, free], False)

    stream.assert_called_once_with([free], False, None)


def test_streamer_is_unchanged_without_a_source():
    reqs = [_req("a"), _req("b")]

    with patch.object(SchedulerOutputStreamer, "_stream_output_generation") as stream:
        _streamer(defer_outputs=False).stream_output(reqs, True)

    stream.assert_called_once_with(reqs, True, None)


def _scheduler():
    scheduler = object.__new__(Scheduler)
    scheduler.output_streamer = SimpleNamespace(
        defer_outputs=False, stream_output=Mock()
    )
    return scheduler


def test_registering_a_source_enables_the_streamer_filter():
    scheduler = _scheduler()
    source = Source()

    scheduler.register_deferred_output_source(source)

    assert scheduler.deferred_output_sources == (source,)
    assert scheduler.output_streamer.defer_outputs is True
    assert Scheduler.deferred_output_sources == ()


def test_released_requests_stream_exactly_once():
    scheduler = _scheduler()
    source = Source()
    scheduler.register_deferred_output_source(source)
    first, second = _req("first", return_logprob=True), _req("second")
    source.hold(first, second)
    source.ready = [first]

    scheduler.stream_released_deferred_outputs()

    scheduler.output_streamer.stream_output.assert_called_once_with([first], True)
    assert first.defer_output is False
    assert second.defer_output is True
    assert scheduler.has_pending_deferred_outputs()

    scheduler.output_streamer.stream_output.reset_mock()
    scheduler.stream_released_deferred_outputs()

    scheduler.output_streamer.stream_output.assert_not_called()


def _idle_scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._engine_paused = False
    scheduler.enable_overlap = False
    scheduler.last_batch = None
    scheduler.chunked_req = None
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.running_batch = MagicMock()
    scheduler.running_batch.is_empty.return_value = True
    scheduler.result_queue = deque()
    scheduler.waiting_queue = []
    scheduler.dllm_manager = MagicMock()
    scheduler.dllm_manager.any_staging_reqs.return_value = False
    scheduler._pp_microbatches_drained = MagicMock(return_value=True)
    scheduler.grammar_manager = MagicMock()
    scheduler.grammar_manager.grammar_queue = []
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_lmcache = False
    scheduler.enable_hisparse = False
    scheduler.output_streamer = SimpleNamespace(defer_outputs=False)
    return scheduler


def test_scheduler_is_busy_while_a_response_is_held():
    scheduler = _idle_scheduler()
    source = Source()
    scheduler.register_deferred_output_source(source)
    assert scheduler.is_fully_idle()

    source.hold(_req("held"))

    assert not scheduler.is_fully_idle()
    # Health checks still need a forward; a held response does not prove one.
    assert scheduler.is_fully_idle(for_health_check=True)

    source.ready = list(source.held)
    source.poll()

    assert scheduler.is_fully_idle()


def test_batch_results_release_held_responses_first():
    scheduler = MagicMock()
    scheduler.deferred_output_sources = (Source(),)
    scheduler.enable_fpm = False
    order = Mock()
    order.attach_mock(scheduler.stream_released_deferred_outputs, "release")
    order.attach_mock(
        scheduler.batch_result_processor.process_batch_result_decode, "process"
    )
    batch = SimpleNamespace(
        reqs=[],
        forward_mode=SimpleNamespace(is_decode=lambda: True, is_extend=lambda: False),
    )

    with patch("sglang.srt.managers.scheduler.flush_trace_batch"):
        Scheduler.process_batch_result.__wrapped__(scheduler, batch, Mock())

    assert order.mock_calls[:2] == [call.release(), call.process(batch, ANY)]


def test_idle_scheduler_releases_held_responses_and_yields():
    scheduler = MagicMock()
    scheduler.deferred_output_sources = (Source(),)
    scheduler.is_fully_idle.return_value = False
    scheduler._last_stall_publish_ts = float("inf")
    scheduler.enable_hicache_storage = False
    scheduler.enable_lmcache = False
    scheduler.disaggregation_mode = DisaggregationMode.NULL

    with patch("sglang.srt.managers.scheduler.time.sleep") as sleep:
        Scheduler.on_idle.__wrapped__(scheduler)

    scheduler.stream_released_deferred_outputs.assert_called_once_with()
    sleep.assert_called_once_with(0)
