import asyncio
import pickle
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import zmq
import zmq.asyncio

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime import ipc_array
from sglang.multimodal_gen.runtime.distributed.ipc_cuda import CudaIpcRef
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClient,
    async_scheduler_client,
    run_zeromq_broker,
)
from sglang.multimodal_gen.runtime.server_args import MAX_SCHEDULER_RPC_TIMEOUT_S
from sglang.srt.utils.network import get_free_port


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


def test_worker_failure_aborts_concurrent_requests_without_rpc_deadline():
    async def run_test():
        context = zmq.asyncio.Context()
        endpoint = f"inproc://scheduler-{uuid.uuid4().hex}"
        server = context.socket(zmq.ROUTER)
        server.bind(endpoint)
        client = AsyncSchedulerClient()
        client.context = context
        client.server_args = SimpleNamespace(scheduler_rpc_timeout=None)
        tasks = [
            asyncio.create_task(client._forward_one(endpoint, i, None))
            for i in range(3)
        ]
        try:
            # All requests reached the worker; none has received a reply.
            for _ in tasks:
                await asyncio.wait_for(server.recv_multipart(), timeout=5)
            client.mark_failed("Scheduler worker exited; restart the server.")
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=5
            )
            assert all(isinstance(result, RuntimeError) for result in results)
            assert all("restart the server" in str(result) for result in results)
            with pytest.raises(RuntimeError, match="restart the server"):
                await client._forward_one(endpoint, "new request", None)
            assert not client._pending_requests
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            server.close(linger=0)
            context.destroy(linger=0)

    asyncio.run(run_test())


@pytest.mark.parametrize("failure", [None, "bind", "recv", "send"])
def test_broker_closes_socket_and_context(monkeypatch, failure):
    async def run_test():
        socket = MagicMock()
        socket.recv = AsyncMock(side_effect=asyncio.Event().wait)
        context = MagicMock()
        context.socket.return_value = socket
        monkeypatch.setattr(zmq.asyncio, "Context", lambda: context)
        if failure == "bind":
            socket.bind.side_effect = zmq.ZMQError(zmq.EADDRINUSE)
        elif failure == "recv":
            # bound the old retry loop so a regression fails instead of hanging CI
            socket.recv.side_effect = [
                zmq.ZMQError(zmq.ENOTSOCK),
                asyncio.CancelledError(),
            ]
        elif failure == "send":
            socket.recv.side_effect = [
                pickle.dumps("request"),
                asyncio.CancelledError(),
            ]
            socket.send = AsyncMock(side_effect=zmq.ZMQError(zmq.ENOTSOCK))
            monkeypatch.setattr(
                async_scheduler_client, "forward", AsyncMock(return_value={"ok": True})
            )

        task = asyncio.create_task(
            run_zeromq_broker(SimpleNamespace(broker_port=12345))
        )
        await asyncio.sleep(0)
        if failure is None:
            task.cancel()
        with pytest.raises(zmq.ZMQError if failure else asyncio.CancelledError):
            await task

        socket.close.assert_called_once_with(linger=0)
        context.destroy.assert_called_once_with(linger=0)

    asyncio.run(run_test())


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "bad_pickle",
        "bad_array",
        "timeout_before",
        "timeout_after",
        "cancel_before",
        "cancel_after",
    ],
)
def test_reply_file_lifetime(monkeypatch, tmp_path, sync, outcome):
    if sync and outcome.startswith("cancel"):
        pytest.skip("sync callers use a transport timeout, not asyncio cancellation")
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ipc_array, "_MIN_FILE_REF_BYTES", 16)

    async def run_test():
        context = zmq.asyncio.Context()
        server = context.socket(zmq.ROUTER)
        port = server.bind_to_random_port("tcp://127.0.0.1")
        client = SchedulerClient() if sync else AsyncSchedulerClient()
        client.context = zmq.Context() if sync else zmq.asyncio.Context()
        client.server_args = SimpleNamespace(scheduler_rpc_timeout=None)
        endpoint = f"tcp://127.0.0.1:{port}"
        args = (endpoint, {}, 1000)
        request = asyncio.create_task(
            asyncio.to_thread(client._forward_one, *args)
            if sync
            else client._forward_one(*args)
        )
        try:
            identity, _, _ = await asyncio.wait_for(server.recv_multipart(), 5)
            assert list(tmp_path.iterdir())
            expected = np.arange(16, dtype=np.uint8)

            async def finish():
                if outcome.startswith("cancel"):
                    request.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request
                elif outcome.startswith("timeout"):
                    with pytest.raises(TimeoutError):
                        await request
                elif outcome == "success":
                    result = await request
                    assert all(np.array_equal(item, expected) for item in result.output)
                else:
                    with pytest.raises((ValueError, pickle.UnpicklingError)):
                        await request
                assert not list(tmp_path.iterdir())

            if outcome.endswith("before"):
                await finish()
                assert (
                    ipc_array.spill_large_arrays_to_file_refs(expected, identity)
                    is expected
                )
            else:
                output = OutputBatch(
                    output=ipc_array.spill_large_arrays_to_file_refs(
                        [expected] * 2, identity
                    )
                )
                assert len(list(tmp_path.rglob("*.npy"))) == 2
                if outcome == "bad_array":
                    Path(output.output[0].path).write_bytes(b"bad npy file")
                if outcome in {"success", "bad_pickle", "bad_array"}:
                    payload = (
                        b"bad pickle"
                        if outcome == "bad_pickle"
                        else pickle.dumps(output)
                    )
                    await server.send_multipart([identity, b"", payload])
                await finish()
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            server.close(linger=0)
            context.destroy(linger=0)
            client.close()

    asyncio.run(run_test())


@pytest.mark.parametrize("failure", ["decode", "forward", "encode"])
def test_broker_serves_next_request_after_job_error(monkeypatch, failure):
    async def run_test():
        async def forward(request):
            if request == "bad":
                if failure == "forward":
                    raise ValueError("rejected job")
                return lambda: None  # cannot serialize a reply
            return {"request": request}

        monkeypatch.setattr(async_scheduler_client, "forward", forward)
        port = get_free_port()
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://127.0.0.1:{port}")
        broker = asyncio.create_task(
            run_zeromq_broker(SimpleNamespace(broker_port=port))
        )
        try:
            payload = b"invalid pickle" if failure == "decode" else pickle.dumps("bad")
            await socket.send(payload)
            error = pickle.loads(await asyncio.wait_for(socket.recv(), 5))
            assert error["status"] == "error" and error["message"]
            await socket.send(pickle.dumps("next"))
            reply = pickle.loads(await asyncio.wait_for(socket.recv(), 5))
            assert reply == {"request": "next"}
            assert not broker.done()
        finally:
            broker.cancel()
            await asyncio.gather(broker, return_exceptions=True)
            socket.close(linger=0)
            context.destroy(linger=0)

    asyncio.run(run_test())
