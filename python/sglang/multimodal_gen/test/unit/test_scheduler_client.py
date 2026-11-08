import asyncio
import pickle
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import numpy as np
import pytest
import zmq
import zmq.asyncio

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime import (
    ipc_array,
)
from sglang.multimodal_gen.runtime import scheduler_client as scheduler_client_module
from sglang.multimodal_gen.runtime.distributed.ipc_cuda import CudaIpcRef
from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    ReleaseRealtimeSessionReq,
    ShutdownReq,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSessionCache
from sglang.multimodal_gen.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClient,
    async_scheduler_client,
    run_zeromq_broker,
)
from sglang.multimodal_gen.runtime.server_args import MAX_SCHEDULER_RPC_TIMEOUT_S
from sglang.multimodal_gen.runtime.server_warmup import (
    run_async_client_warmup,
    run_sync_client_warmup,
)
from sglang.srt.utils.network import get_free_port


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


@pytest.mark.parametrize("sync", [False, True])
def test_server_warmup_reaches_every_replica_without_skewing_round_robin(
    monkeypatch, sync
):
    endpoints = ["tcp://replica-0", "tcp://replica-1"]
    warmup_reqs = [Req(prompt=f"warmup-{i}", is_warmup=True) for i in range(2)]
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_warmup.build_warmup_reqs",
        lambda *args, **kwargs: warmup_reqs,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_warmup.should_include_warmup_image",
        lambda *args, **kwargs: False,
    )
    server_args = SimpleNamespace(
        scheduler_endpoints=endpoints,
        scheduler_rpc_timeout=None,
        warmup_resolutions=["512x512"],
    )
    client = SchedulerClient() if sync else AsyncSchedulerClient()
    client.context = object()
    client.server_args = server_args
    client.request_logger = MagicMock()
    received = {endpoint: [] for endpoint in endpoints}

    def record(endpoint, batch):
        received[endpoint].append(batch)
        return OutputBatch()

    if sync:
        client._forward_one = lambda endpoint, batch, timeout_ms: record(
            endpoint, batch
        )
    else:

        async def forward_one(endpoint, batch, timeout_ms):
            return record(endpoint, batch)

        client._forward_one = forward_one

    async def run_test():
        if sync:
            await asyncio.to_thread(run_sync_client_warmup, server_args, client.forward)
        else:
            await run_async_client_warmup(server_args, client.forward)
        for endpoint in endpoints:
            assert [req.prompt for req in received[endpoint]] == [
                "warmup-0",
                "warmup-1",
            ]
            # every replica counts the full list, not a share of it
            assert {req.extra["warmup_total"] for req in received[endpoint]} == {2}
            received[endpoint].clear()

        # warmup left the counter alone: real traffic still starts at replica 0
        for endpoint in endpoints:
            batch = [Req(prompt="real")]
            if sync:
                await asyncio.to_thread(client.forward, batch)
            else:
                await client.forward(batch)
            assert len(received[endpoint]) == 1

    asyncio.run(run_test())


def test_sync_scheduler_client_converts_configured_seconds_to_milliseconds():
    response = object()
    socket = MagicMock()
    socket.recv_pyobj.return_value = response
    client = SchedulerClient()
    client.context = SimpleNamespace(socket=lambda _socket_type: socket)
    client.server_args = SimpleNamespace(scheduler_rpc_timeout=2)

    assert client._forward_one("tcp://scheduler", object(), None) is response

    socket.setsockopt.assert_any_call(zmq.RCVTIMEO, 2000)
    socket.poll.assert_not_called()
    socket.close.assert_called_once_with()


def test_sync_scheduler_client_worker_failure_interrupts_wait_without_deadline():
    socket = MagicMock()
    socket.poll.return_value = 0
    client = SchedulerClient()
    client.context = SimpleNamespace(socket=lambda _socket_type: socket)
    client.server_args = SimpleNamespace(scheduler_rpc_timeout=None)
    check_worker_health = MagicMock(
        side_effect=[None, RuntimeError("Scheduler worker exited")]
    )

    with pytest.raises(RuntimeError, match="Scheduler worker exited"):
        client._forward_one(
            "tcp://scheduler",
            object(),
            None,
            check_worker_health=check_worker_health,
        )

    socket.send_pyobj.assert_called_once()
    socket.poll.assert_called_once_with(1000, zmq.POLLIN)
    socket.recv_pyobj.assert_not_called()
    socket.close.assert_called_once_with()


def test_sync_scheduler_client_can_wait_across_multiple_health_polls():
    response = object()
    socket = MagicMock()
    socket.poll.side_effect = [0, 0, zmq.POLLIN]
    socket.recv_pyobj.return_value = response
    client = SchedulerClient()
    client.context = SimpleNamespace(socket=lambda _socket_type: socket)
    client.server_args = SimpleNamespace(scheduler_rpc_timeout=None)
    check_worker_health = MagicMock()

    assert (
        client._forward_one(
            "tcp://scheduler",
            object(),
            None,
            check_worker_health=check_worker_health,
        )
        is response
    )

    assert check_worker_health.call_count == 3
    assert socket.poll.call_args_list == [call(1000, zmq.POLLIN)] * 3
    socket.send_pyobj.assert_called_once()
    socket.recv_pyobj.assert_called_once_with()
    socket.close.assert_called_once_with()


def test_sync_scheduler_client_health_polls_share_one_transport_deadline(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    socket = MagicMock()

    def poll(timeout_ms, flags):
        clock.now += timeout_ms / 1000
        return 0

    socket.poll.side_effect = poll
    monkeypatch.setattr(scheduler_client_module.time, "monotonic", lambda: clock.now)
    client = SchedulerClient()
    client.context = SimpleNamespace(socket=lambda _socket_type: socket)
    client.server_args = SimpleNamespace(scheduler_rpc_timeout=9)
    check_worker_health = MagicMock()

    with pytest.raises(TimeoutError, match="did not respond"):
        client._forward_one(
            "tcp://scheduler",
            object(),
            timeout_ms=2500,
            check_worker_health=check_worker_health,
        )

    assert socket.poll.call_args_list == [
        call(1000, zmq.POLLIN),
        call(1000, zmq.POLLIN),
        call(500, zmq.POLLIN),
    ]
    assert clock.now == 2.5
    socket.send_pyobj.assert_called_once()
    socket.recv_pyobj.assert_not_called()
    socket.close.assert_called_once_with()


def test_sync_scheduler_client_forward_preserves_default_routing_call_shape():
    batch = object()
    client = SchedulerClient()
    client._forward_routed = MagicMock(return_value="response")

    assert client.forward(batch) == "response"

    client._forward_routed.assert_called_once_with(batch, None, None)


@pytest.mark.parametrize("with_health_check", [False, True])
@pytest.mark.parametrize("control_request", [False, True])
def test_sync_scheduler_client_routes_optional_worker_health_check(
    with_health_check, control_request
):
    batch = ShutdownReq() if control_request else object()
    client = SchedulerClient()
    client.server_args = SimpleNamespace(scheduler_endpoints=["tcp://scheduler"])
    client.request_logger = MagicMock()
    client._forward_one = MagicMock(return_value="response")
    check_worker_health = MagicMock() if with_health_check else None

    assert client.forward(batch, check_worker_health=check_worker_health) == "response"

    expected_kwargs = (
        {"check_worker_health": check_worker_health} if with_health_check else {}
    )
    client._forward_one.assert_called_once_with(
        "tcp://scheduler", batch, None, **expected_kwargs
    )


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
