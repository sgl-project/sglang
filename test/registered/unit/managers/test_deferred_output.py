"""Finished responses remain scheduler-owned while external work completes."""

from array import array
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.runtime_context import get_serving
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.weight_versions import (
    WeightVersionSpan,
    compute_weight_version_spans,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def topology():
    with published_topology(weight_version="v1"):
        yield


def _req(rid, return_logprob=False):
    req = Req(rid, "", array("q", [1]), SamplingParams(), return_logprob=return_logprob)
    req.output_ids.append(2)
    req.finished_reason = FINISH_LENGTH(1)
    return req


def _scheduler():
    scheduler = object.__new__(Scheduler)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.waiting_queue = []
    scheduler.chunked_req = None
    scheduler.hisparse_coordinator = None
    scheduler.collect_inflight_reqs = lambda: set()
    scheduler.ipc_channels = Mock()
    scheduler.rust_server = None
    streamer = object.__new__(SchedulerOutputStreamer)
    streamer.is_generation = True
    streamer.defer_outputs = False
    scheduler.output_streamer = streamer
    return scheduler


def test_response_is_filtered_until_completion_and_released_once():
    scheduler = _scheduler()
    first, second = _req("first", return_logprob=True), _req("second")
    first_done, second_done = Future(), Future()
    scheduler.defer_response(first, first_done)
    scheduler.defer_response(second, second_done)

    with patch.object(SchedulerOutputStreamer, "_stream_output_generation") as stream:
        scheduler.output_streamer.stream_output([first, second, _req("free")], True)
        assert [req.rid for req in stream.call_args.args[0]] == ["free"]
        stream.reset_mock()
        scheduler.stream_released_deferred_outputs()
        stream.assert_not_called()

        first_done.set_result(None)
        scheduler.stream_released_deferred_outputs()
        stream.assert_called_once_with([first], True, None)
        assert not first.defer_output and second.defer_output
        assert scheduler.has_pending_deferred_outputs()
        scheduler.stream_released_deferred_outputs()
        stream.assert_called_once()

        second_done.set_result(None)
        scheduler.stream_released_deferred_outputs()
        assert stream.call_args == call([second], False, None)
        assert not scheduler.has_pending_deferred_outputs()
        assert Scheduler._deferred_responses is None


@pytest.mark.parametrize(
    "invalid", ["unfinished", "sent", "duplicate", "prefill", "decode"]
)
def test_defer_response_rejects_invalid_ownership(invalid):
    scheduler, req = _scheduler(), _req("request")
    if invalid == "unfinished":
        req.finished_reason = None
    elif invalid == "sent":
        req.finished_output = True
    elif invalid == "duplicate":
        scheduler.defer_response(req, Future())
        req = _req(req.rid)
    else:
        scheduler.disaggregation_mode = DisaggregationMode(invalid)

    with pytest.raises(ValueError):
        scheduler.defer_response(req, Future())

    assert not req.defer_output


@pytest.mark.parametrize(
    ("cancelled", "rust_frontend"),
    [(False, False), (True, False), (False, True)],
    ids=["error", "cancelled", "rust-error"],
)
def test_failed_completion_releases_an_error_response(cancelled, rust_frontend):
    """Failed publication reaches the active frontend, including Rust egress."""
    scheduler, req, completion = _scheduler(), _req("failed"), Future()
    if rust_frontend:
        scheduler.rust_server = Mock()
    scheduler.defer_response(req, completion)
    if cancelled:
        completion.cancel()
    else:
        completion.set_exception(RuntimeError("publish failed"))

    with patch.object(SchedulerOutputStreamer, "_stream_output_generation") as stream:
        scheduler.stream_released_deferred_outputs()
        scheduler.stream_released_deferred_outputs()

    stream.assert_not_called()
    send = (
        scheduler.rust_server.push_abort
        if rust_frontend
        else scheduler.ipc_channels.send_to_tokenizer.send_output
    )
    send.assert_called_once()
    abort, sent_req = send.call_args.args
    assert sent_req is req and abort.rid == req.rid
    assert abort.finished_reason["type"] == "abort"
    assert abort.finished_reason["status_code"] == 500
    assert req.finished_output
    assert not req.defer_output and not scheduler.has_pending_deferred_outputs()


@pytest.mark.parametrize(
    ("abort_all", "rust_frontend"),
    [(False, False), (True, False), (False, True)],
    ids=["matching", "all", "rust-matching"],
)
def test_abort_cancels_only_matching_held_responses(abort_all, rust_frontend):
    """Cancellation retires matching responses through the active frontend."""
    scheduler = _scheduler()
    if rust_frontend:
        scheduler.rust_server = Mock()
    scheduler.mm_receiver = None
    scheduler.dllm_config = None
    scheduler.grammar_manager = Mock()
    reqs = [_req("group-0"), _req("other")]
    completions = [Future(), Future()]
    if not abort_all:
        completions[0].set_running_or_notify_cancel()
    for req, completion in zip(reqs, completions):
        scheduler.defer_response(req, completion)

    scheduler.abort_request(AbortReq(rid="group-", abort_all=abort_all))
    assert [completion.cancelled() for completion in completions] == [
        abort_all,
        abort_all,
    ]
    assert [req.defer_output for req in reqs] == [False, not abort_all]
    assert scheduler.has_pending_deferred_outputs() is (not abort_all)
    send = (
        scheduler.rust_server.push_abort
        if rust_frontend
        else scheduler.ipc_channels.send_to_tokenizer.send_output
    )
    sent = send.call_args_list
    assert [args[0].rid for args, _ in sent] == [
        req.rid for req in reqs[: 2 if abort_all else 1]
    ]
    if not abort_all:
        completions[0].set_result(None)
    with patch.object(SchedulerOutputStreamer, "_stream_output_generation") as stream:
        scheduler.stream_released_deferred_outputs()
    stream.assert_not_called()


def test_held_tokens_keep_their_weight_version():
    scheduler, req = _scheduler(), _req("held")
    scheduler.defer_response(req, Future())

    scheduler.record_weight_version_change("v2")

    assert compute_weight_version_spans(
        req.weight_version_events, get_serving().weight_version, len(req.output_ids)
    ) == [WeightVersionSpan(version="v1", start=0, end=1)]


def test_held_responses_do_not_change_idle_state():
    # Only the output rank holds responses under TP; cache flush and memory
    # release must still see the same idle state on every rank.
    scheduler = _scheduler()
    scheduler._engine_paused = False
    scheduler.enable_overlap = False
    scheduler.last_batch = None
    scheduler.running_batch = SimpleNamespace(is_empty=lambda: True)
    scheduler.result_queue = deque()
    scheduler.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
    scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_lmcache = False
    scheduler.enable_hisparse = False
    scheduler.defer_response(_req("held"), Future())

    assert scheduler.has_pending_deferred_outputs()
    assert scheduler.is_fully_idle()
    assert scheduler.is_fully_idle(for_health_check=True)


@pytest.mark.parametrize("state", ["batch", "busy", "idle", "paused"])
def test_scheduler_polls_until_a_held_response_completes(state):
    scheduler = MagicMock()
    scheduler._deferred_responses = None
    scheduler.output_streamer = _scheduler().output_streamer
    scheduler.stream_released_deferred_outputs = lambda: (
        Scheduler.stream_released_deferred_outputs(scheduler)
    )
    scheduler.has_pending_deferred_outputs = lambda: (
        Scheduler.has_pending_deferred_outputs(scheduler)
    )
    scheduler.is_fully_idle.return_value = state == "idle"
    scheduler.enable_fpm = False
    scheduler._last_stall_publish_ts = float("inf")
    scheduler.enable_hicache_storage = False
    scheduler.enable_lmcache = False
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.enable_hisparse = True  # skips the pool accounting checks

    def poll():
        if state == "batch":
            Scheduler.process_batch_result.__wrapped__(scheduler, Mock(reqs=[]), Mock())
        elif state == "paused":
            Scheduler._record_scheduler_state_for_paused_engine(scheduler)
        else:
            Scheduler.on_idle.__wrapped__(scheduler)

    req, completion = _req("held"), Future()
    Scheduler.defer_response(scheduler, req, completion)
    with (
        patch.object(SchedulerOutputStreamer, "_stream_output_generation") as stream,
        patch("sglang.srt.managers.scheduler.time.sleep"),
    ):
        poll()
        assert req.defer_output
        stream.assert_not_called()
        scheduler.maybe_sleep_on_idle.assert_not_called()

        completion.set_result(None)
        poll()
        stream.assert_called_once_with([req], False, None)
        assert not req.defer_output and not scheduler.has_pending_deferred_outputs()
        if state == "idle":
            scheduler.maybe_sleep_on_idle.assert_called_once_with()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
