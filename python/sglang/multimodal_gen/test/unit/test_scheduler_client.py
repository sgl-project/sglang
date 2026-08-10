import asyncio
import pickle
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import zmq
import zmq.asyncio

from sglang.multimodal_gen.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClient,
    run_zeromq_broker,
)
from sglang.multimodal_gen.runtime.server_args import MAX_SCHEDULER_RPC_TIMEOUT_S


def test_sync_scheduler_client_converts_configured_seconds_to_milliseconds():
    response = object()
    socket = MagicMock()
    socket.recv_pyobj.return_value = response
    client = SchedulerClient()
    client.context = SimpleNamespace(socket=lambda _socket_type: socket)
    client.server_args = SimpleNamespace(scheduler_rpc_timeout=2)

    assert client._forward_one("tcp://scheduler", object(), None) is response

    socket.setsockopt.assert_any_call(zmq.RCVTIMEO, 2000)
    socket.close.assert_called_once_with()


def test_async_scheduler_client_has_no_transport_deadline_by_default():
    response = {"status": "ok"}
    socket = MagicMock()
    socket.send = AsyncMock()
    socket.recv = AsyncMock(return_value=pickle.dumps(response))
    client = AsyncSchedulerClient()
    client.context = SimpleNamespace(socket=lambda _socket_type: socket)
    client.server_args = SimpleNamespace(scheduler_rpc_timeout=None)

    result = asyncio.run(client._forward_one("tcp://scheduler", object(), None))

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
    [0, -1, MAX_SCHEDULER_RPC_TIMEOUT_S * 1000 + 1, True],
)
def test_scheduler_client_rejects_invalid_override_and_closes_socket(
    invalid_timeout_ms,
):
    socket = MagicMock()
    client = SchedulerClient()
    client.context = SimpleNamespace(socket=lambda _socket_type: socket)
    client.server_args = SimpleNamespace(scheduler_rpc_timeout=None)

    with pytest.raises(ValueError, match="timeout_ms must be None"):
        client._forward_one("tcp://scheduler", object(), timeout_ms=invalid_timeout_ms)

    socket.close.assert_called_once_with()


def test_async_scheduler_client_waits_for_delayed_response():
    async def run_test():
        context = zmq.asyncio.Context()
        endpoint = f"inproc://scheduler-{uuid.uuid4().hex}"
        server = context.socket(zmq.REP)
        server.bind(endpoint)
        client = AsyncSchedulerClient()
        client.context = context
        client.server_args = SimpleNamespace(scheduler_rpc_timeout=None)

        async def reply():
            await server.recv()
            await asyncio.sleep(0.02)
            await server.send(pickle.dumps({"status": "ok"}))

        reply_task = asyncio.create_task(reply())
        try:
            result = await client._forward_one(endpoint, object(), timeout_ms=1000)
            await reply_task
            assert result == {"status": "ok"}
        finally:
            server.close(linger=0)
            context.destroy(linger=0)

    asyncio.run(run_test())


def test_async_scheduler_client_honors_explicit_deadline():
    async def run_test():
        context = zmq.asyncio.Context()
        endpoint = f"inproc://scheduler-{uuid.uuid4().hex}"
        server = context.socket(zmq.REP)
        server.bind(endpoint)
        client = AsyncSchedulerClient()
        client.context = context
        client.server_args = SimpleNamespace(scheduler_rpc_timeout=None)

        try:
            with pytest.raises(TimeoutError, match="did not respond"):
                await client._forward_one(endpoint, object(), timeout_ms=10)
        finally:
            server.close(linger=0)
            context.destroy(linger=0)

    asyncio.run(run_test())


def test_async_scheduler_client_closes_socket_when_cancelled():
    async def run_test():
        socket = MagicMock()
        socket.send = AsyncMock()
        socket.recv = AsyncMock(side_effect=asyncio.Event().wait)
        client = AsyncSchedulerClient()
        client.context = SimpleNamespace(socket=lambda _socket_type: socket)
        client.server_args = SimpleNamespace(scheduler_rpc_timeout=None)

        task = asyncio.create_task(
            client._forward_one("tcp://scheduler", object(), timeout_ms=None)
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        socket.close.assert_called_once_with()

    asyncio.run(run_test())


def test_broker_closes_socket_and_context_when_cancelled(monkeypatch):
    async def run_test():
        socket = MagicMock()
        socket.recv = AsyncMock(side_effect=asyncio.Event().wait)
        context = MagicMock()
        context.socket.return_value = socket
        monkeypatch.setattr(zmq.asyncio, "Context", lambda: context)

        task = asyncio.create_task(
            run_zeromq_broker(SimpleNamespace(broker_port=12345))
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        socket.close.assert_called_once_with(linger=0)
        context.destroy.assert_called_once_with(linger=0)

    asyncio.run(run_test())
