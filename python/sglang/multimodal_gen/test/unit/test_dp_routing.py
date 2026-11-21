"""Routing invariants for DP serving.

Generation requests go to exactly one replica, control ops reach every replica
unless a fatal device error aborts them, and a realtime session stays on its replica.
"""

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    AutoResidencyReq,
    ReleaseRealtimeSessionReq,
    SetLoraReq,
    ShutdownReq,
    StartProfileReq,
    StopProfileReq,
)
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import (
    _CONTROL_REQ_TYPES,
    AsyncSchedulerClient,
    SchedulerClient,
    _is_warmup_batch,
    _merge_auto_residency_results,
    _merge_fanout_results,
    _select_replica,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _req(**kwargs) -> Req:
    return Req(prompt="a pear", **kwargs)


def test_round_robin_covers_every_replica():
    counter = itertools.count()
    picks = [_select_replica([_req()], 3, counter) for _ in range(6)]
    assert picks == [0, 1, 2, 0, 1, 2]


def test_dp1_always_selects_replica_zero():
    counter = itertools.count()
    assert {_select_replica([_req()], 1, counter) for _ in range(5)} == {0}


@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_session_requests_stick_to_one_replica(dp_size):
    counter = itertools.count()
    session_req = _req(realtime_session_id="session-abc")
    release_req = ReleaseRealtimeSessionReq(session_id="session-abc")
    picks = {
        _select_replica(req, dp_size, counter)
        for req in [session_req, [session_req], release_req, release_req]
    }
    assert len(picks) == 1
    # and the counter was never consumed by session traffic
    assert next(counter) == 0
    assert not isinstance(release_req, _CONTROL_REQ_TYPES)


def test_control_reqs_are_recognized():
    assert isinstance(AutoResidencyReq(), _CONTROL_REQ_TYPES)
    assert isinstance(SetLoraReq(lora_nickname="x", lora_path="y"), _CONTROL_REQ_TYPES)
    assert isinstance(ShutdownReq(), _CONTROL_REQ_TYPES)
    assert not isinstance([_req()], _CONTROL_REQ_TYPES)
    assert not isinstance(_req(), _CONTROL_REQ_TYPES)


def test_warmup_reqs_are_fanned_out_to_every_replica():
    assert _is_warmup_batch(_req(is_warmup=True))
    assert _is_warmup_batch([_req(is_warmup=True), _req(is_warmup=True)])
    assert not _is_warmup_batch(_req())
    assert not _is_warmup_batch([])


def test_fanout_merge_surfaces_the_failing_replica():
    ok = OutputBatch(output=None)
    bad = OutputBatch(error="replica 1 exploded")
    assert _merge_fanout_results([ok, bad]) is bad
    assert _merge_fanout_results([ok, OutputBatch(output=None)]) is ok


def test_auto_residency_fanout_rewarms_when_any_replica_adjusts():
    skipped = OutputBatch(output={"status": "skipped"})
    adjusted = OutputBatch(output={"status": "adjusted"})
    assert _merge_auto_residency_results([skipped, adjusted]) is adjusted


def test_auto_residency_fanout_tolerates_a_locally_rolled_back_replica():
    restored = OutputBatch(
        output={"status": "rolled_back"}, error="original placement restored"
    )
    adjusted = OutputBatch(output={"status": "adjusted"})
    assert _merge_auto_residency_results([restored, adjusted]) is adjusted


def test_auto_residency_fanout_never_hides_failed_rollback():
    failed = OutputBatch(output={"status": "rollback_failed"}, error="restart required")
    adjusted = OutputBatch(output={"status": "adjusted"})
    assert _merge_auto_residency_results([adjusted, failed]) is failed


def test_auto_residency_fanout_reports_successful_rollback():
    skipped = OutputBatch(output={"status": "skipped"})
    restored = OutputBatch(output={"status": "rolled_back"})
    assert _merge_auto_residency_results([skipped, restored]) is restored


@pytest.mark.parametrize("client_type", [SchedulerClient, AsyncSchedulerClient])
@pytest.mark.parametrize(
    "error",
    [
        None,
        "CUDA out of memory",
        "CUDA error: an illegal memory access was encountered",
    ],
)
def test_fanout_preserves_fatal_reply_before_later_timeout(client_type, error):
    """A later timeout must preserve fatal replies but still fail other calls."""
    client = client_type()
    client.server_args = SimpleNamespace(
        scheduler_endpoints=["tcp://replica-0", "tcp://replica-1"],
        scheduler_rpc_timeout=None,
    )
    client.request_logger = Mock()
    reply = OutputBatch(error=error)
    timeout = TimeoutError("Scheduler did not respond in time.")
    mock_type = AsyncMock if client_type is AsyncSchedulerClient else Mock
    client._forward_one = mock_type(side_effect=[reply, timeout])
    # Async forward checks initialization even when the transport is mocked.
    client.context = Mock()

    def forward():
        request = SetLoraReq(lora_nickname="x", lora_path="y")
        if client_type is AsyncSchedulerClient:
            return asyncio.run(client.forward(request))
        return client.forward(request)

    if error and "illegal memory access" in error:
        assert forward() is reply
        assert client._forward_one.call_count == 1
    else:
        with pytest.raises(TimeoutError) as exc_info:
            forward()
        assert exc_info.value is timeout
        assert client._forward_one.call_count == 2


@pytest.mark.parametrize("client_type", [SchedulerClient, AsyncSchedulerClient])
@pytest.mark.parametrize(
    "prior_errors",
    [(), (None,), ("CUDA out of memory",), ("invalid request",)],
    ids=["fatal-first", "success-first", "oom-first", "ordinary-error-first"],
)
@pytest.mark.parametrize("auto_residency", [False, True])
def test_fanout_stops_on_fatal_reply_without_deadline(
    client_type, prior_errors, auto_residency
):
    fatal = OutputBatch(error="CUDA error: an illegal memory access was encountered")
    replies = [
        OutputBatch(
            error=error,
            output={"status": "adjusted"} if auto_residency else None,
        )
        for error in prior_errors
    ] + [fatal]
    endpoints = [f"tcp://replica-{i}" for i in range(len(replies) + 1)]
    client = client_type()
    client.server_args = SimpleNamespace(
        scheduler_endpoints=endpoints,
        scheduler_rpc_timeout=None,
    )
    client.request_logger = Mock()
    client.context = Mock()

    def reply(endpoint, batch, timeout_ms):
        assert timeout_ms is None
        if endpoint == endpoints[-1]:
            # Fail immediately if an RPC that could wait forever is started.
            pytest.fail("Contacted another replica after receiving a fatal reply")
        return replies[endpoints.index(endpoint)]

    mock_type = AsyncMock if client_type is AsyncSchedulerClient else Mock
    client._forward_one = mock_type(side_effect=reply)
    request = (
        AutoResidencyReq()
        if auto_residency
        else SetLoraReq(lora_nickname="x", lora_path="y")
    )
    if client_type is AsyncSchedulerClient:
        result = asyncio.run(asyncio.wait_for(client.forward(request), timeout=1))
    else:
        result = client.forward(request)

    assert result is fatal
    assert [call.args[0] for call in client._forward_one.call_args_list] == endpoints[
        :-1
    ]
    client.request_logger.log_finished_request.assert_called_once_with(request, fatal)


@pytest.mark.parametrize("client_type", [SchedulerClient, AsyncSchedulerClient])
@pytest.mark.parametrize("error", [None, "CUDA out of memory", "invalid request"])
@pytest.mark.parametrize(
    "control_request",
    [
        SetLoraReq(lora_nickname="x", lora_path="y"),
        StartProfileReq(profile_id="test-profile"),
        StopProfileReq(),
    ],
)
def test_nonfatal_control_fanout_reaches_every_replica(
    client_type, error, control_request
):
    replies = [OutputBatch(error=error), OutputBatch(), OutputBatch()]
    endpoints = [f"tcp://replica-{i}" for i in range(len(replies))]
    client = client_type()
    client.server_args = SimpleNamespace(
        scheduler_endpoints=endpoints,
        scheduler_rpc_timeout=None,
    )
    client.request_logger = Mock()
    client.context = Mock()
    mock_type = AsyncMock if client_type is AsyncSchedulerClient else Mock
    client._forward_one = mock_type(side_effect=replies)
    if client_type is AsyncSchedulerClient:
        result = asyncio.run(client.forward(control_request))
    else:
        result = client.forward(control_request)

    assert result is replies[0]
    assert [call.args[0] for call in client._forward_one.call_args_list] == endpoints


def test_scheduler_endpoints_one_per_replica():
    args = ServerArgs.__new__(ServerArgs)
    args.host = "localhost"
    args.dp_size = 3
    args.scheduler_port = 6000
    args.scheduler_ports = None
    assert ServerArgs.scheduler_endpoints.fget(args) == [
        "tcp://127.0.0.1:6000",
        "tcp://127.0.0.1:6001",
        "tcp://127.0.0.1:6002",
    ]
    # settled ports need not be consecutive; the endpoint list follows them
    args.scheduler_ports = [6000, 7005, 7100]
    assert ServerArgs.scheduler_endpoint_for(args, 2) == "tcp://127.0.0.1:7100"
