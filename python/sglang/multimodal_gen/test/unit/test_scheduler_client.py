import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import zmq

from sglang.multimodal_gen.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClient,
)
from sglang.multimodal_gen.runtime.server_args import (
    MAX_SCHEDULER_RECV_TIMEOUT_MS,
)


def test_sync_scheduler_client_uses_configured_timeout():
    response = object()
    socket = MagicMock()
    socket.recv_pyobj.return_value = response
    client = SchedulerClient()
    client.context = SimpleNamespace(socket=lambda _socket_type: socket)
    client.server_args = SimpleNamespace(scheduler_recv_timeout_ms=1234)

    assert client._forward_one("tcp://scheduler", object(), None) is response

    socket.setsockopt.assert_any_call(zmq.RCVTIMEO, 1234)
    socket.close.assert_called_once_with()


def test_async_scheduler_client_accepts_indefinite_override():
    response = {"status": "ok"}
    socket = MagicMock()
    socket.send = AsyncMock()
    socket.recv = AsyncMock(return_value=pickle.dumps(response))
    client = AsyncSchedulerClient()
    client.context = SimpleNamespace(socket=lambda _socket_type: socket)
    client.server_args = SimpleNamespace(scheduler_recv_timeout_ms=1234)

    result = asyncio.run(
        client._forward_one("tcp://scheduler", object(), timeout_ms=-1)
    )

    assert result == response
    recv_timeout_calls = [
        call
        for call in socket.setsockopt.call_args_list
        if call.args[0] == zmq.RCVTIMEO
    ]
    assert recv_timeout_calls == []
    socket.close.assert_called_once_with()


@pytest.mark.parametrize(
    "invalid_timeout_ms",
    [0, -2, MAX_SCHEDULER_RECV_TIMEOUT_MS + 1, True],
)
def test_scheduler_client_rejects_invalid_override_and_closes_socket(
    invalid_timeout_ms,
):
    socket = MagicMock()
    client = SchedulerClient()
    client.context = SimpleNamespace(socket=lambda _socket_type: socket)
    client.server_args = SimpleNamespace(scheduler_recv_timeout_ms=1234)

    with pytest.raises(ValueError, match="timeout_ms must be -1"):
        client._forward_one("tcp://scheduler", object(), timeout_ms=invalid_timeout_ms)

    socket.close.assert_called_once_with()
