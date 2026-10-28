"""Finished responses remain scheduler-owned while external work completes."""

from array import array
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.deferred_responses import (
    DeferredResponses,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_request_lists import (
    make_request_registry,
    make_request_retirer,
    partial_scheduler,
)
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


def _deferred(disaggregation_mode=DisaggregationMode.NULL, rust_frontend=False):
    streamer = object.__new__(SchedulerOutputStreamer)
    streamer.is_generation = True
    streamer.defer_outputs = False
    streamer.send_to_tokenizer = Mock()
    streamer.rust_server = Mock() if rust_frontend else None
    return DeferredResponses(
        disaggregation_mode=disaggregation_mode,
        output_streamer=streamer,
        request_retirer=make_request_retirer(output_streamer=streamer),
    )


def _sent_aborts(deferred):
    """The sender of the active frontend: the Rust egress or the tokenizer."""
    streamer = deferred.output_streamer
    if streamer.rust_server is not None:
        return streamer.rust_server.push_abort
    return streamer.send_to_tokenizer.send_output


def _scheduler(deferred):
    scheduler = partial_scheduler()
    scheduler.parts.deferred_responses = deferred
    scheduler.hub.request_registry = make_request_registry()
    return scheduler


def test_response_is_filtered_until_completion_and_released_once():
    deferred = _deferred()
    first, second = _req("first", return_logprob=True), _req("second")
    first_done, second_done = Future(), Future()
    deferred.defer(first, first_done)
    deferred.defer(second, second_done)

    with patch.object(SchedulerOutputStreamer, "_stream_output_generation") as stream:
        deferred.output_streamer.stream_output([first, second, _req("free")], True)
        assert [req.rid for req in stream.call_args.args[0]] == ["free"]
        stream.reset_mock()
        deferred.stream_released()
        stream.assert_not_called()

        first_done.set_result(None)
        deferred.stream_released()
        stream.assert_called_once_with([first], True, None)
        assert not first.defer_output and second.defer_output
        assert deferred.has_pending()
        deferred.stream_released()
        stream.assert_called_once()

        second_done.set_result(None)
        deferred.stream_released()
        assert stream.call_args == call([second], False, None)
        assert not deferred.has_pending()


@pytest.mark.parametrize(
    "invalid", ["unfinished", "sent", "duplicate", "prefill", "decode"]
)
def test_defer_response_rejects_invalid_ownership(invalid):
    deferred, req = _deferred(), _req("request")
    if invalid == "unfinished":
        req.finished_reason = None
    elif invalid == "sent":
        req.finished_output = True
    elif invalid == "duplicate":
        deferred.defer(req, Future())
        req = _req(req.rid)
    else:
        deferred = _deferred(DisaggregationMode(invalid))

    with pytest.raises(ValueError):
        deferred.defer(req, Future())

    assert not req.defer_output


@pytest.mark.parametrize(
    ("cancelled", "rust_frontend"),
    [(False, False), (True, False), (False, True)],
    ids=["error", "cancelled", "rust-error"],
)
def test_failed_completion_releases_an_error_response(cancelled, rust_frontend):
    """Failed publication reaches the active frontend, including Rust egress."""
    deferred = _deferred(rust_frontend=rust_frontend)
    req, completion = _req("failed"), Future()
    deferred.defer(req, completion)
    if cancelled:
        completion.cancel()
    else:
        completion.set_exception(RuntimeError("publish failed"))

    with patch.object(SchedulerOutputStreamer, "_stream_output_generation") as stream:
        deferred.stream_released()
        deferred.stream_released()

    stream.assert_not_called()
    send = _sent_aborts(deferred)
    send.assert_called_once()
    abort, sent_req = send.call_args.args
    assert sent_req is req and abort.rid == req.rid
    assert abort.finished_reason["type"] == "abort"
    assert abort.finished_reason["status_code"] == 500
    assert req.finished_output
    assert not req.defer_output and not deferred.has_pending()


@pytest.mark.parametrize(
    ("abort_all", "rust_frontend"),
    [(False, False), (True, False), (False, True)],
    ids=["matching", "all", "rust-matching"],
)
def test_abort_cancels_only_matching_held_responses(abort_all, rust_frontend):
    """Cancellation retires matching responses through the active frontend."""
    deferred = _deferred(rust_frontend=rust_frontend)
    scheduler = _scheduler(deferred)
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
    sent = _sent_aborts(deferred).call_args_list
    assert [args[0].rid for args, _ in sent] == [
        req.rid for req in reqs[: 2 if abort_all else 1]
    ]
    if not abort_all:
        completions[0].set_result(None)
    with patch.object(SchedulerOutputStreamer, "_stream_output_generation") as stream:
        scheduler.stream_released_deferred_outputs()
    stream.assert_not_called()


def test_held_responses_do_not_change_idle_state():
    # Only the output rank holds responses under TP; cache flush and memory
    # release must still see the same idle state on every rank.
    deferred = _deferred()
    scheduler = _scheduler(deferred)
    scheduler.defer_response(_req("held"), Future())

    assert scheduler.has_pending_deferred_outputs()
    registry = scheduler.hub.request_registry
    assert registry.is_fully_idle()
    assert registry.is_fully_idle(for_health_check=True)


@pytest.mark.parametrize("state", ["batch", "busy", "idle", "paused"])
def test_scheduler_polls_until_a_held_response_completes(state):
    scheduler = MagicMock()
    scheduler.parts.deferred_responses = _deferred()
    scheduler.hub.request_registry.is_fully_idle.return_value = state == "idle"
    scheduler.parts.metrics_sink.enable_fpm = False
    scheduler.mode.disaggregation_mode = DisaggregationMode.NULL
    scheduler._last_stall_publish_ts = float("inf")
    # HiSparse skips the pool accounting checks.
    memory = SimpleNamespace(
        enable_hisparse=True, enable_unified_memory=False, enable_lmcache=False
    )

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
        patch("sglang.srt.managers.scheduler.get_memory", return_value=memory),
        patch(
            "sglang.srt.managers.scheduler.hicache_storage_enabled",
            return_value=False,
        ),
        patch("sglang.srt.managers.scheduler.commit_step_counters"),
    ):
        poll()
        assert req.defer_output
        stream.assert_not_called()
        scheduler.maybe_sleep_on_idle.assert_not_called()

        completion.set_result(None)
        poll()
        stream.assert_called_once_with([req], False, None)
        assert not req.defer_output
        assert not Scheduler.has_pending_deferred_outputs(scheduler)
        if state == "idle":
            scheduler.maybe_sleep_on_idle.assert_called_once_with()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
