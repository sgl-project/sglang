"""Unit tests for aborting queued rollout requests (POST /rollout/abort)."""

import asyncio
from collections import deque
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime import scheduler_client
from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    ABORTED_ERROR,
    AbortReq,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training import rollout_api
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    RolloutAbortRequest,
    RolloutRequest,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req


def _req(request_id, *, is_warmup=False):
    req = Req(sampling_params=SamplingParams(prompt="p", request_id=request_id))
    req.is_warmup = is_warmup
    return req


def _scheduler(queued):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.metrics = None
    scheduler.return_result = Mock()
    scheduler.waiting_queue = deque(
        (f"client-{req.request_id}".encode(), req, 0.0) for req in queued
    )
    return scheduler


def _replies(scheduler):
    return {
        identity: output.error or output.output
        for (output, identity), _ in scheduler.return_result.call_args_list
    }


def test_abort_drops_only_matching_queued_requests():
    a, b, c = _req("a"), _req("b"), _req("c")
    scheduler = _scheduler([a, b, c])

    remaining = scheduler._apply_aborts(
        [
            (b"client-x", AbortReq(request_ids=["a", "c", "unknown"])),
            (b"client-d", _req("d")),
        ]
    )

    # the abort never enters the queue, so it cannot wait behind the requests it cancels
    assert [req.request_id for _, req in remaining] == ["d"]
    # survivors keep FIFO order
    assert [entry[1] for entry in scheduler.waiting_queue] == [b]
    assert _replies(scheduler) == {
        b"client-a": ABORTED_ERROR,
        b"client-c": ABORTED_ERROR,
        b"client-x": {"success": True, "num_aborted": 2},
    }


def test_abort_all_spares_warmup_and_grouped_requests_are_matched_by_first_req():
    warmup = _req("w", is_warmup=True)
    group = [_req("g"), _req("g2")]
    scheduler = _scheduler([warmup, _req("a")])
    scheduler.waiting_queue.append((b"client-g", group, 0.0))

    scheduler._apply_aborts([(b"client-x", AbortReq(abort_all=True))])

    assert [entry[1] for entry in scheduler.waiting_queue] == [warmup]
    assert _replies(scheduler)[b"client-x"] == {"success": True, "num_aborted": 2}


def test_abort_drops_generate_from_the_same_batch():
    scheduler = _scheduler([])
    x, y = _req("x"), _req("y")

    remaining = scheduler._apply_aborts(
        [
            (b"client-x", x),
            (b"client-abort", AbortReq(request_ids=["x"])),
            (b"client-y", y),
        ]
    )

    assert remaining == [(b"client-y", y)]
    assert not scheduler.waiting_queue
    assert _replies(scheduler) == {
        b"client-x": ABORTED_ERROR,
        b"client-abort": {"success": True, "num_aborted": 1},
    }


def test_abort_fans_out_to_every_replica():
    # a request may be queued on any dp replica, so the abort must reach all of them
    assert AbortReq in scheduler_client._CONTROL_REQ_TYPES


def test_abort_reply_sums_num_aborted_across_replicas():
    client = scheduler_client.AsyncSchedulerClient()
    client.context = Mock()
    client.request_logger = Mock()
    client.server_args = Mock(scheduler_endpoints=["ep0", "ep1"])
    replies = {
        "ep0": OutputBatch(output={"success": True, "num_aborted": 0}),
        "ep1": OutputBatch(output={"success": True, "num_aborted": 2}),
    }
    with patch.object(
        client, "_forward_one", AsyncMock(side_effect=lambda ep, *_: replies[ep])
    ):
        out = asyncio.run(client.forward(AbortReq(request_ids=["a", "b"])))
    # the request was queued on replica 1, so replica 0 alone would report 0
    assert out.output == {"success": True, "num_aborted": 2}


def test_aborted_rollout_returns_499_not_500():
    request = RolloutRequest(prompt="p", request_id="miles-0-1")
    with (
        patch.object(rollout_api, "get_global_server_args"),
        patch.object(rollout_api, "build_sampling_params") as build,
        patch.object(rollout_api, "prepare_request"),
        patch.object(
            rollout_api.async_scheduler_client,
            "forward",
            AsyncMock(return_value=OutputBatch(error=ABORTED_ERROR)),
        ),
        pytest.raises(HTTPException) as exc,
    ):
        asyncio.run(rollout_api.rollout_generate(request))
    assert exc.value.status_code == 499
    # the client's id is what the scheduler matches aborts against
    assert build.call_args.args[0] == "miles-0-1"


@pytest.mark.parametrize("request_id", ["../etc/passwd", "a/b", "", "x" * 129])
def test_client_request_id_is_safe_for_file_names(request_id):
    with pytest.raises(ValidationError):
        RolloutRequest(prompt="p", request_id=request_id)


def test_abort_requires_ids_or_abort_all():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(rollout_api.rollout_abort(RolloutAbortRequest()))
    assert exc.value.status_code == 400
