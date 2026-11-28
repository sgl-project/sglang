"""A fake uniflow._core for CPU tests of the UniFlow PD transfer backend."""

import ctypes
import struct
import threading
from types import ModuleType, SimpleNamespace

from sglang.srt.disaggregation.uniflow.buffers import BufferKeys, BufferKind
from sglang.srt.disaggregation.uniflow.conn import PeerBuffers
from sglang.srt.disaggregation.uniflow.transport import (
    UniflowRemoteBuffers,
    UniflowTransport,
)


class FakeResult:
    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error

    def has_value(self):
        return self._error is None

    def has_error(self):
        return self._error is not None

    def value(self):
        return self._value

    def error(self):
        return self._error


class FakeSegment:
    """A segment whose export id is its host address and length; a span is
    (address, length)."""

    def __init__(self, ptr, length, memory_type=None, device_id=None):
        self.ptr = ptr
        self.length = length
        self.memory_type = memory_type
        self.device_id = device_id

    def export_id(self):
        return FakeResult(struct.pack("<QQ", self.ptr, self.length))

    def span(self, offset, length):
        assert 0 <= offset and offset + length <= self.length
        return self.ptr + offset, length


class FakeFuture:
    def wait_for(self, timeout_ms):
        return True

    def get(self):
        return FakeResult()


class FakeConnection:
    """Copies host memory at put() and completes at once."""

    def __init__(self):
        self.is_shut_down = False

    def shutdown(self):
        self.is_shut_down = True

    def put(self, requests):
        for request in requests:
            (src, src_len), (dst, dst_len) = request.src, request.dst
            assert src_len == dst_len
            ctypes.memmove(dst, src, src_len)
        return FakeFuture()


class FakeAgent:
    def __init__(self, config):
        self.config = config
        # Every registered segment and opened connection, in order.
        self.registered = []
        self.connections = []

    def get_unique_id(self):
        return FakeResult(f"fake-{self.config.name}-{id(self)}")

    def register_segment(self, segment):
        self.registered.append(segment)
        return FakeResult(segment)

    def import_segment(self, export_id):
        return FakeResult(FakeSegment(*struct.unpack("<QQ", export_id)))

    def connect(self, agent_id):
        connection = FakeConnection()
        self.connections.append(connection)
        return FakeResult(connection)

    def accept(self):
        threading.Event().wait()


def fake_uniflow_core():
    """A uniflow._core stand-in whose agent is a FakeAgent."""
    core = ModuleType("uniflow._core")
    core.MemoryType = SimpleNamespace(VRAM="VRAM", DRAM="DRAM")
    core.Segment = FakeSegment
    core.TransferRequest = lambda src, dst: SimpleNamespace(src=src, dst=dst)
    core.UniflowAgentConfig = lambda device_id, name, listen_address: SimpleNamespace(
        device_id=device_id, name=name, listen_address=listen_address
    )
    core.UniflowAgent = FakeAgent
    return core


def _layout(kv, aux, state):
    return list(kv), list(aux), [list(s) for s in state]


def attach_uniflow_transport(manager, *, kv=(), aux=(), state=(), gpu_id=0):
    """Give `manager` a real UniflowTransport over a FakeAgent with this side's
    buffers, each a host (ptr, length), registered; aux as host memory."""
    transport = UniflowTransport.create(
        fake_uniflow_core(), device_id=gpu_id, name="test", host="127.0.0.1"
    )
    local = _layout(kv, aux, state)
    keys = BufferKeys.for_buffers(*local)
    for key, (ptr, length) in keys.pair(*local):
        device = -1 if key.kind == BufferKind.AUX else gpu_id
        transport.register(key, ptr, length, device)
    manager.transport = transport
    manager._local_keys = keys
    return transport


def peer_buffers(connection, *, agent_id, kv=(), aux=(), state=()):
    """A decode peer written through `connection`, its buffers given as host
    (ptr, length) regions."""
    layout = _layout(kv, aux, state)
    keys = BufferKeys.for_buffers(*layout)
    segments = {
        key: FakeSegment(ptr, length) for key, (ptr, length) in keys.pair(*layout)
    }
    remote = UniflowRemoteBuffers(
        agent_id=agent_id, connection=connection, segments=segments
    )
    return PeerBuffers(remote=remote, keys=keys)
