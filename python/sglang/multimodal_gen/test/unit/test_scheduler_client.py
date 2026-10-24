import asyncio
import pickle
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import zmq
import zmq.asyncio

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.distributed.ipc_cuda import CudaIpcRef
from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    ReleaseRealtimeSessionReq,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSessionCache
from sglang.multimodal_gen.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClient,
    run_zeromq_broker,
)
from sglang.multimodal_gen.runtime.server_args import MAX_SCHEDULER_RPC_TIMEOUT_S


@pytest.mark.parametrize("client_type", [SchedulerClient, AsyncSchedulerClient])
def test_session_release_reaches_owner_over_zmq(client_type):
    async def run_test():
        context = zmq.asyncio.Context()
        servers = [context.socket(zmq.REP) for _ in range(2)]
        endpoints = [
            f"tcp://127.0.0.1:{server.bind_to_random_port('tcp://127.0.0.1')}"
            for server in servers
        ]
        caches = [RealtimeSessionCache() for _ in servers]
        client = client_type()
        client.context = (
            zmq.Context() if client_type is SchedulerClient else zmq.asyncio.Context()
        )
        client.server_args = SimpleNamespace(
            scheduler_endpoints=endpoints, scheduler_rpc_timeout=2
        )
        client.request_logger = MagicMock()

        async def serve(replica):
            socket, cache = servers[replica], caches[replica]
            while True:
                request = await socket.recv_pyobj()
                released = None
                if isinstance(request, ReleaseRealtimeSessionReq):
                    released = cache.release(request.session_id)
                else:
                    for req in request:
                        cache.attach(req)
                await socket.send_pyobj(
                    OutputBatch(output={"replica": replica, "released": released})
                )

        async def forward(request):
            if client_type is SchedulerClient:
                return await asyncio.to_thread(client.forward, request)
            return await client.forward(request)

        tasks = [asyncio.create_task(serve(replica)) for replica in range(2)]
        try:
            owners = set()
            for index in range(16):
                session_id = f"session-{index}"
                chunks = [
                    await forward(
                        [
                            Req(
                                prompt="a pear",
                                realtime_session_id=session_id,
                                block_idx=block,
                            )
                        ]
                    )
                    for block in range(2)
                ]
                owner = chunks[0].output["replica"]
                owners.add(owner)
                assert chunks[1].output["replica"] == owner

                # unrelated traffic must not change session cleanup ownership
                ordinary = await forward([Req(prompt="a pear")])
                assert ordinary.output["replica"] == index % 2
                for expected_released in (True, False):
                    release = await forward(ReleaseRealtimeSessionReq(session_id))
                    assert release.output == {
                        "replica": owner,
                        "released": expected_released,
                    }
                assert all(not cache._sessions for cache in caches)
            assert owners == {0, 1}
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            client.close()
            for server in servers:
                server.close(linger=0)
            context.destroy(linger=0)

    asyncio.run(asyncio.wait_for(run_test(), timeout=30))


@pytest.mark.parametrize(
    "invalid", ["pickle", "truncated", "protocol", "module", "empty", "cuda"]
)
@pytest.mark.parametrize("reply_blocked", [False, True])
def test_scheduler_isolates_invalid_messages_and_replies(invalid, reply_blocked):
    bad_ref = CudaIpcRef(
        dtype="invalid_dtype",
        shape=(1,),
        stride=(1,),
        storage_offset=0,
        device_index=0,
        handle=b"",
        storage_size_bytes=1,
        storage_offset_bytes=0,
        ref_counter_handle=b"",
        ref_counter_offset=0,
        event_handle=b"",
        event_sync_required=False,
    )
    payload = {
        "pickle": b"invalid pickle",
        "truncated": b"",
        "protocol": b"\x80\xff.",
        "module": b"cnot_installed_sglang_extension\nMissingRequest\n.",
        "empty": pickle.dumps([]),
        "cuda": pickle.dumps({"tensor": bad_ref}),
    }[invalid]
    context = zmq.Context()
    receiver = context.socket(zmq.ROUTER)
    client = context.socket(zmq.DEALER)
    endpoint = f"inproc://message-isolation-{uuid.uuid4().hex}"
    receiver.bind(endpoint)
    client.connect(endpoint)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.receiver = receiver
    scheduler._socket_lock = threading.Lock()
    scheduler.server_args = SimpleNamespace(
        comfyui_mode=False,
        sp_degree=1,
        enable_cfg_parallel=False,
        tp_size=1,
    )
    try:
        before = Req(sampling_params=SamplingParams(prompt="before"))
        after = Req(sampling_params=SamplingParams(prompt="after"))
        for message in (pickle.dumps([before]), payload, pickle.dumps([after])):
            client.send_multipart([b"", message])
        if reply_blocked:
            scheduler.receiver = MagicMock(wraps=receiver)
            scheduler.receiver.send_multipart.side_effect = zmq.Again()
        items = scheduler.recv_reqs()
        assert [req.prompt for _, req in items] == ["before", "after"]
        if reply_blocked:
            scheduler.receiver.send_multipart.assert_called_once()
        else:
            assert client.poll(1000)
            reply = client.recv_multipart()
            assert reply[0] == b""
            result = pickle.loads(reply[1])
            assert isinstance(result, OutputBatch)
            assert result.error.startswith("Invalid scheduler request:")
        client.send_multipart([b"", pickle.dumps([after])])
        assert scheduler.recv_reqs()[0][1].prompt == "after"
    finally:
        client.close(linger=0)
        receiver.close(linger=0)
        context.term()


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
