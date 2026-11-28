"""UniflowTransport tests over a fake uniflow._core, on CPU in one process."""

from __future__ import annotations

import base64
import gc
import logging
import struct
import sys
import threading
import weakref
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from sglang.srt.disaggregation.transport.base import Copy, Status, Transport
from sglang.srt.disaggregation.uniflow import transport as uniflow_transport
from sglang.srt.disaggregation.uniflow.buffers import (
    BufferKeys,
    BufferKind,
    UniflowBufferKey,
    copy_ranges,
    group_units,
)
from sglang.srt.disaggregation.uniflow.transport import (
    MAX_ACCEPTED_CONNECTIONS,
    Registration,
    UniflowRemoteBuffers,
    UniflowTransport,
    import_uniflow_core,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.uniflow_utils import FakeResult, FakeSegment, fake_uniflow_core

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

KV0 = UniflowBufferKey(BufferKind.KV, 0)
KV1 = UniflowBufferKey(BufferKind.KV, 1)
AUX0 = UniflowBufferKey(BufferKind.AUX, 0)
GPU_ID = 3


def _transport() -> UniflowTransport:
    return UniflowTransport.create(
        fake_uniflow_core(), device_id=GPU_ID, name="test", host="127.0.0.1"
    )


def _export_id(ptr: int, length: int) -> str:
    return base64.b64encode(struct.pack("<QQ", ptr, length)).decode("ascii")


class _ScriptedFuture:
    """A put future that becomes ready after `ready_after` polls."""

    def __init__(self, *, ready_after: int = 0, error: str | None = None) -> None:
        self.ready_after = ready_after
        self.error = error
        self.waits: list[int] = []
        self.gets = 0

    def wait_for(self, timeout_ms: int) -> bool:
        self.waits.append(timeout_ms)
        return len(self.waits) > self.ready_after

    def get(self) -> FakeResult:
        self.gets += 1
        return FakeResult(error=self.error)


class _RecordingConnection:
    def __init__(self, *futures: Any) -> None:
        self.futures = list(futures)
        self.puts: list[list[tuple[Any, Any]]] = []
        self.is_shut_down = False
        self.shutdown_error: Exception | None = None

    def put(self, requests: list[Any]) -> Any:
        self.puts.append([(request.src, request.dst) for request in requests])
        return self.futures.pop(0)

    def shutdown(self) -> None:
        self.is_shut_down = True
        if self.shutdown_error is not None:
            raise self.shutdown_error


def _remote(connection: Any, **segments: tuple[int, int]) -> UniflowRemoteBuffers:
    keys = {"kv0": KV0, "kv1": KV1, "aux0": AUX0}
    return UniflowRemoteBuffers(
        agent_id="decode",
        connection=connection,
        segments={
            keys[name]: FakeSegment(*region) for name, region in segments.items()
        },
    )


def test_buffer_keys_follow_the_register_layout() -> None:
    keys = BufferKeys.for_buffers([10, 20], [30], [[40, 50]])
    state00 = UniflowBufferKey(BufferKind.STATE, 0, 0)
    state01 = UniflowBufferKey(BufferKind.STATE, 0, 1)

    assert keys.kv == (KV0, KV1) and keys.aux == (AUX0,)
    assert keys.state == ((state00, state01),)
    assert keys.pair([10, 20], [30], [[40, 50]]) == [
        (KV0, 10),
        (KV1, 20),
        (AUX0, 30),
        (state00, 40),
        (state01, 50),
    ]
    assert UniflowBufferKey(BufferKind.KV, 0) == KV0
    assert UniflowBufferKey(BufferKind.AUX, 0) != KV0
    assert len({KV0, UniflowBufferKey(BufferKind.KV, 0), AUX0}) == 2
    assert copy_ranges([(1, 2, 3)]).tolist() == [[1, 2, 3]]
    # Pages 0-1 -> 4-5 and 3 -> 7 are two runs contiguous on both sides.
    groups = ([[0, 1], [3]], [[4, 5], [7]])
    assert group_units(*groups).tolist() == [[0, 4, 2], [3, 7, 1]]


def test_create_builds_the_agent_on_an_unbracketed_address() -> None:
    transport = UniflowTransport.create(
        fake_uniflow_core(), device_id=GPU_ID, name="prefill_0", host="fdbd::1"
    )

    assert isinstance(transport, Transport)
    config = transport._agent.config
    assert (config.device_id, config.name) == (GPU_ID, "prefill_0")
    # UniFlow splits host:port on the last colon and rejects brackets.
    assert config.listen_address == "fdbd::1:0"
    assert transport.agent_id.startswith("fake-prefill_0-")


def test_import_uniflow_core_names_the_backend_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = fake_uniflow_core()
    monkeypatch.setitem(sys.modules, "uniflow", ModuleType("uniflow"))
    monkeypatch.setitem(sys.modules, "uniflow._core", core)
    assert import_uniflow_core() is core

    monkeypatch.setitem(sys.modules, "uniflow._core", None)
    with pytest.raises(ImportError, match="--disaggregation-transfer-backend uniflow"):
        import_uniflow_core()


def test_register_places_device_memory_in_vram_and_host_memory_in_dram() -> None:
    transport = _transport()

    transport.register(KV0, 0x1000, 64, GPU_ID)
    transport.register(AUX0, 0x2000, 32, -1)

    placements = [(s.memory_type, s.device_id) for s in transport._agent.registered]
    assert placements == [("VRAM", GPU_ID), ("DRAM", -1)]
    assert transport.export_ids([KV0, AUX0]) == [
        _export_id(0x1000, 64),
        _export_id(0x2000, 32),
    ]


def test_zero_length_buffer_is_not_registered() -> None:
    transport = _transport()

    transport.register(KV1, 0x3000, 0, GPU_ID)

    assert transport._agent.registered == []
    assert transport.export_ids([KV1]) == [""]


def test_register_all_registers_every_entry_in_order() -> None:
    transport = _transport()
    transport.register(KV0, 0x1000, 64, GPU_ID)

    transport.register_all(
        [
            Registration(key=KV0, ptr=0x5000, length=128, device=GPU_ID),
            Registration(key=KV1, ptr=0x6000, length=0, device=GPU_ID),
            Registration(key=AUX0, ptr=0x7000, length=32, device=-1),
        ]
    )

    regions = [(s.ptr, s.length, s.memory_type) for s in transport._agent.registered]
    assert regions == [
        (0x1000, 64, "VRAM"),
        (0x5000, 128, "VRAM"),
        (0x7000, 32, "DRAM"),
    ]
    # KV0 now names the new region; the zero-length KV1 keeps an empty export id.
    assert transport.export_ids([KV0, KV1, AUX0]) == [
        _export_id(0x5000, 128),
        "",
        _export_id(0x7000, 32),
    ]


def test_reregistering_a_key_as_zero_length_releases_its_segment() -> None:
    transport = _transport()
    transport.register(KV0, 0x1000, 64, GPU_ID)
    first = weakref.ref(transport._agent.registered.pop())

    transport.register_all([Registration(key=KV0, ptr=0x1000, length=0, device=GPU_ID)])
    gc.collect()

    assert first() is None
    assert transport.export_ids([KV0]) == [""]


def test_deregister_and_reregister_release_the_segment() -> None:
    transport = _transport()
    transport.register(KV0, 0x1000, 64, GPU_ID)
    [segment] = transport._agent.registered
    first = weakref.ref(segment)
    transport._agent.registered.clear()
    del segment

    # Registering a key again replaces its segment and releases the old one.
    transport.register(KV0, 0x4000, 128, GPU_ID)
    gc.collect()
    assert first() is None
    assert transport.export_ids([KV0]) == [_export_id(0x4000, 128)]

    second = weakref.ref(transport._agent.registered.pop())
    assert second() is not None
    # Destroying the segment handle is UniFlow's deregistration.
    transport.deregister([KV0])
    gc.collect()
    assert second() is None
    with pytest.raises(KeyError):
        transport.export_ids([KV0])


def test_write_all_posts_one_batch_in_copy_and_row_order() -> None:
    transport = _transport()
    transport.register(KV0, 0x1000, 4096, GPU_ID)
    transport.register(KV1, 0x2000, 4096, GPU_ID)
    connection = _RecordingConnection(_ScriptedFuture())
    remote = _remote(connection, kv0=(0x9000, 4096), kv1=(0xA000, 4096))

    handles = transport.write_all(
        remote,
        [
            Copy(src=KV0, dst=KV0, ranges=copy_ranges([(0, 64, 16), (32, 128, 8)])),
            Copy(src=KV1, dst=KV1, ranges=copy_ranges([(16, 0, 4)])),
        ],
    )

    assert connection.puts == [
        [
            ((0x1000, 16), (0x9040, 16)),
            ((0x1020, 8), (0x9080, 8)),
            ((0x2010, 4), (0xA000, 4)),
        ]
    ]
    assert len(handles) == 1
    assert transport.status(handles[0]).status is Status.DONE
    assert transport.status_all(handles).status is Status.DONE


def test_write_without_ranges_posts_nothing() -> None:
    transport = _transport()
    transport.register(KV0, 0x1000, 64, GPU_ID)
    connection = _RecordingConnection()
    remote = _remote(connection, kv0=(0x9000, 64))
    empty = copy_ranges([])

    assert transport.write_all(remote, []) == []
    assert transport.write_all(remote, [Copy(src=KV0, dst=KV0, ranges=empty)]) == []
    handle = transport.write(remote, KV0, KV0, empty)
    assert transport.status(handle).status is Status.DONE
    assert connection.puts == []


def test_status_polls_without_blocking_and_repeats_final_results() -> None:
    transport = _transport()
    transport.register(KV0, 0x1000, 64, GPU_ID)
    future = _ScriptedFuture(ready_after=1)
    remote = _remote(_RecordingConnection(future), kv0=(0x9000, 64))
    handle = transport.write(remote, KV0, KV0, copy_ranges([(0, 0, 8)]))

    assert transport.status(handle).status is Status.PENDING
    assert transport.status(handle).status is Status.DONE
    # DONE is final: later polls repeat it without touching the future.
    assert transport.status(handle).status is Status.DONE
    assert future.waits == [0, 0] and future.gets == 1


def test_put_errors_fail_the_handle_with_the_binding_message() -> None:
    transport = _transport()
    transport.register(KV0, 0x1000, 64, GPU_ID)
    ranges = copy_ranges([(0, 0, 8)])

    failed = transport.write(
        _remote(_RecordingConnection(_ScriptedFuture(error="link down")), kv0=(0, 64)),
        KV0,
        KV0,
        ranges,
    )
    result = transport.status(failed)
    assert (result.status, result.reason) == (Status.FAILED, "link down")
    assert transport.status(failed) is result

    class _RefusingConnection:
        def put(self, requests: list[Any]) -> None:
            raise RuntimeError("connection closed")

    # A rejected post is final at once; part of the batch may be on the wire.
    rejected = transport.write(
        _remote(_RefusingConnection(), kv0=(0, 64)), KV0, KV0, ranges
    )
    assert rejected.future is None
    assert transport.status(rejected).reason == "connection closed"

    def broken_get() -> None:
        raise RuntimeError("broken promise")

    broken = SimpleNamespace(wait_for=lambda timeout_ms: True, get=broken_get)
    handle = transport.write(
        _remote(_RecordingConnection(broken), kv0=(0, 64)), KV0, KV0, ranges
    )
    result = transport.status(handle)
    assert (result.status, result.reason) == (Status.FAILED, "broken promise")


def test_wait_all_honors_the_deadline_and_stops_at_the_first_failure() -> None:
    transport = _transport()
    transport.register(KV0, 0x1000, 64, GPU_ID)
    ranges = copy_ranges([(0, 0, 8)])

    def post(future: Any) -> Any:
        remote = _remote(_RecordingConnection(future), kv0=(0, 64))
        return transport.write(remote, KV0, KV0, ranges)

    slow = _ScriptedFuture(ready_after=1)
    # The first batch gets the whole deadline, as one put waited before.
    assert transport.wait_all([post(slow)], timeout_ms=25).status is Status.PENDING
    assert slow.waits == [25]

    done, failed, never_waited = (
        _ScriptedFuture(),
        _ScriptedFuture(error="link down"),
        _ScriptedFuture(),
    )
    result = transport.wait_all(
        [post(done), post(failed), post(never_waited)], timeout_ms=1000
    )
    assert (result.status, result.reason) == (Status.FAILED, "link down")
    assert never_waited.waits == []

    handles = [post(_ScriptedFuture()), post(_ScriptedFuture())]
    assert transport.wait_all(handles, timeout_ms=1000).status is Status.DONE
    assert transport.status_all(handles).status is Status.DONE


def test_import_registration_imports_segments_before_connecting() -> None:
    transport = _transport()

    remote = transport.import_registration(
        agent_id="decode-agent",
        export_ids={KV0: _export_id(0x9000, 256), KV1: ""},
    )

    assert remote.agent_id == "decode-agent"
    assert set(remote.segments) == {KV0}
    assert remote.segments[KV0].span(16, 8) == (0x9010, 8)
    assert transport._agent.connections == [remote.connection]

    def refuse_import(export_id: bytes) -> FakeResult:
        return FakeResult(error="stale export id")

    transport._agent.import_segment = refuse_import
    with pytest.raises(RuntimeError, match="stale export id"):
        transport.import_registration(
            agent_id="decode-agent", export_ids={KV0: _export_id(0x9000, 256)}
        )
    # A failed import opens no connection.
    assert len(transport._agent.connections) == 1


def test_disconnect_and_close_shut_down_opened_connections(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = _transport()
    transport.register(KV0, 0x1000, 64, GPU_ID)
    export_ids = {KV0: _export_id(0x9000, 64)}
    first, second, third = (
        transport.import_registration(agent_id=f"decode-{i}", export_ids=export_ids)
        for i in range(3)
    )

    transport.disconnect(first)
    assert first.connection.is_shut_down

    def broken_shutdown() -> None:
        raise RuntimeError("shutdown failed")

    second.connection.shutdown = broken_shutdown
    with caplog.at_level(logging.ERROR, logger=uniflow_transport.logger.name):
        transport.close()

    # close() goes on past a failed shutdown and drops every registration.
    assert "UniFlow shutdown of peer decode-1 failed" in caplog.text
    assert third.connection.is_shut_down
    with pytest.raises(KeyError):
        transport.export_ids([KV0])
    with pytest.raises(RuntimeError, match="shutdown failed"):
        transport.disconnect(second)


def test_accept_loop_keeps_connections_and_survives_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = _transport()
    accepted = object()
    drained = threading.Event()
    script: list[Any] = [
        FakeResult(error="listener down"),
        FakeResult(error="listener down"),
        RuntimeError("binding glitch"),
        FakeResult(accepted),
    ]

    def accept() -> FakeResult:
        if not script:
            drained.set()
            threading.Event().wait()
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    transport._agent.accept = accept
    with caplog.at_level(logging.DEBUG, logger=uniflow_transport.logger.name):
        thread = transport.start_accepting()
        assert drained.wait(timeout=10.0)

    assert thread.daemon and thread.is_alive()
    assert list(transport._accepted) == [accepted]
    assert transport._accepted.maxlen == MAX_ACCEPTED_CONNECTIONS
    levels = [
        record.levelno
        for record in caplog.records
        if "UniFlow accept failed" in record.getMessage()
    ]
    # A broken listener repeats its error; only the first one warns.
    assert levels == [logging.WARNING, logging.DEBUG]
    assert "UniFlow accept loop iteration failed" in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
