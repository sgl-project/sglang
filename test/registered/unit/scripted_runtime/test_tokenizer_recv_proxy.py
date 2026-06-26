from __future__ import annotations

import unittest
from collections import deque
from dataclasses import dataclass

import zmq

from sglang.srt.managers.io_struct import msgpack_decode, msgpack_encode, wrap_as_pickle
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime.tokenizer_recv_proxy import ScriptedTokenizerRecvProxy
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclass
class _ControlMsg:

    tag: str = "flush"


@dataclass
class _StartReq:

    rid: str


class _FakeUnderlyingSocket:

    def __init__(self) -> None:
        self._ready: deque = deque()
        self._scheduled: list[list] = []

    def feed(self, obj: object) -> None:
        self._ready.append(obj)

    def feed_after_drain_cycles(self, obj: object, *, cycles: int) -> None:
        self._scheduled.append([cycles, obj])

    def _pop_ready(self) -> object:
        if self._ready:
            return self._ready.popleft()

        for entry in self._scheduled:
            entry[0] -= 1
        ready_now = [obj for remaining, obj in self._scheduled if remaining <= 0]
        self._scheduled = [entry for entry in self._scheduled if entry[0] > 0]
        self._ready.extend(ready_now)

        raise zmq.ZMQError(zmq.EAGAIN, "Resource temporarily unavailable")

    def recv(self, flags: int = 0) -> bytes:
        return msgpack_encode(wrap_as_pickle(self._pop_ready()))

    def recv_pyobj(self, flags: int = 0) -> object:
        return self._pop_ready()


def _is_control(obj: object) -> bool:
    return isinstance(obj, _ControlMsg)


def _is_start_req(rid: str):
    return lambda obj: isinstance(obj, _StartReq) and obj.rid == rid


class TestScriptedTokenizerRecvProxyRecv(CustomTestCase):

    def test_recv_drains_then_returns_msgpack_bytes_fifo(self):
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        first, second = _ControlMsg("a"), _ControlMsg("b")
        underlying.feed(first)
        underlying.feed(second)

        self.assertEqual(msgpack_decode(proxy.recv()), first)
        self.assertEqual(msgpack_decode(proxy.recv()), second)

    def test_recv_empty_noblock_raises_eagain(self):
        proxy = ScriptedTokenizerRecvProxy(underlying=_FakeUnderlyingSocket())

        with self.assertRaises(zmq.ZMQError) as ctx:
            proxy.recv(zmq.NOBLOCK)
        self.assertEqual(ctx.exception.errno, zmq.EAGAIN)

    def test_recv_empty_blocking_raises_runtime_error(self):
        proxy = ScriptedTokenizerRecvProxy(underlying=_FakeUnderlyingSocket())

        with self.assertRaisesRegex(RuntimeError, "blocking recv is not supported"):
            proxy.recv()

    def test_recv_pyobj_drains_then_pops_fifo(self):
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        first, second = _ControlMsg("a"), _ControlMsg("b")
        underlying.feed(first)
        underlying.feed(second)

        self.assertEqual(proxy.recv_pyobj(), first)
        self.assertEqual(proxy.recv_pyobj(), second)

    def test_recv_pyobj_empty_noblock_raises_eagain(self):
        proxy = ScriptedTokenizerRecvProxy(underlying=_FakeUnderlyingSocket())

        with self.assertRaises(zmq.ZMQError) as ctx:
            proxy.recv_pyobj(zmq.NOBLOCK)
        self.assertEqual(ctx.exception.errno, zmq.EAGAIN)

    def test_recv_pyobj_empty_blocking_raises_runtime_error(self):
        proxy = ScriptedTokenizerRecvProxy(underlying=_FakeUnderlyingSocket())

        with self.assertRaisesRegex(RuntimeError, "blocking recv is not supported"):
            proxy.recv_pyobj()


class TestScriptedTokenizerRecvProxyWaitUntilArrived(CustomTestCase):

    def _proxy_with_stale_control(self):
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        stale = _ControlMsg("stale")
        underlying.feed(stale)
        proxy.wait_until_arrived(_is_control, timeout_s=1.0)
        return proxy, underlying, stale

    def test_wait_until_arrived_returns_on_first_match_when_buffer_empty(self):
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        msg = _ControlMsg("first")
        underlying.feed(msg)

        proxy.wait_until_arrived(_is_control, timeout_s=1.0)

        self.assertEqual(proxy.recv_pyobj(), msg)

    def test_wait_until_arrived_skips_stale_same_type_object(self):
        proxy, _, _ = self._proxy_with_stale_control()

        with self.assertRaises(TimeoutError):
            proxy.wait_until_arrived(_is_control, timeout_s=0.05)

    def test_wait_until_arrived_returns_on_new_object_after_stale(self):
        proxy, underlying, stale = self._proxy_with_stale_control()
        fresh = _ControlMsg("fresh")
        underlying.feed_after_drain_cycles(fresh, cycles=1)

        proxy.wait_until_arrived(_is_control, timeout_s=2.0)

        self.assertEqual(proxy.recv_pyobj(), stale)
        self.assertEqual(proxy.recv_pyobj(), fresh)

    def test_wait_until_arrived_rid_predicate_ignores_stale_other_rid(self):
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        old = _StartReq(rid="old")
        underlying.feed(old)
        proxy.wait_until_arrived(_is_start_req("old"), timeout_s=1.0)

        new = _StartReq(rid="new")
        underlying.feed(new)
        proxy.wait_until_arrived(_is_start_req("new"), timeout_s=1.0)

        self.assertEqual(proxy.recv_pyobj(), old)
        self.assertEqual(proxy.recv_pyobj(), new)

    def test_wait_until_arrived_rid_predicate_skips_stale_same_rid(self):
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        underlying.feed(_StartReq(rid="reused"))
        proxy.wait_until_arrived(_is_start_req("reused"), timeout_s=1.0)

        with self.assertRaises(TimeoutError):
            proxy.wait_until_arrived(_is_start_req("reused"), timeout_s=0.05)

    def test_wait_until_arrived_timeout_message_names_description(self):
        proxy = ScriptedTokenizerRecvProxy(underlying=_FakeUnderlyingSocket())

        with self.assertRaisesRegex(TimeoutError, "FlushCacheReqInput"):
            proxy.wait_until_arrived(
                _is_control, timeout_s=0.02, description="FlushCacheReqInput"
            )


if __name__ == "__main__":
    unittest.main()
