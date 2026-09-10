from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime.background_http_poster import BackgroundHttpPoster
from sglang.test.scripted_runtime.context.http_post import _http_post_and_await_recv_msg
from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
    ScriptedTokenizerRecvProxy,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import unittest


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

    def recv_pyobj(self, flags: int = 0) -> object:
        if self._ready:
            return self._ready.popleft()

        for entry in self._scheduled:
            entry[0] -= 1
        ready_now = [obj for remaining, obj in self._scheduled if remaining <= 0]
        self._scheduled = [entry for entry in self._scheduled if entry[0] > 0]
        self._ready.extend(ready_now)

        raise zmq.ZMQError(zmq.EAGAIN, "Resource temporarily unavailable")


def _is_control(obj: object) -> bool:
    return isinstance(obj, _ControlMsg)


def _is_start_req(rid: str):
    return lambda obj: isinstance(obj, _StartReq) and obj.rid == rid


class TestScriptedTokenizerRecvProxyRecv(CustomTestCase):
    def test_recv_pyobj_drains_then_pops_fifo(self):
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        first, second = _ControlMsg("a"), _ControlMsg("b")
        underlying.feed(first)
        underlying.feed(second)

        self.assertIs(proxy.recv_pyobj(), first)
        self.assertIs(proxy.recv_pyobj(), second)

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
    def test_http_post_failure_reaches_script_wait(self):
        """A rejected POST must report its error instead of a socket-arrival timeout."""
        poster = BackgroundHttpPoster()
        self.addCleanup(poster.close)
        poster.post = AsyncMock(side_effect=RuntimeError("request rejected"))
        ctx = SimpleNamespace(
            scheduler=SimpleNamespace(
                server_args=SimpleNamespace(host="127.0.0.1", port=30000)
            ),
            _http_poster=poster,
            _tokenizer_recv_proxy=ScriptedTokenizerRecvProxy(
                underlying=_FakeUnderlyingSocket()
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "request rejected"):
            _http_post_and_await_recv_msg(
                ctx,
                path="/generate",
                json={"rid": "reused", "stream": True},
                predicate=_is_start_req("reused"),
                description="reused request",
                timeout_s=1.0,
            )

        poster.post.assert_awaited_once_with(
            "http://127.0.0.1:30000/generate", {"rid": "reused", "stream": True}
        )

    def test_wait_until_arrived_propagates_failed_post(self):
        """A failed request must interrupt the wait for its scheduler message."""
        proxy = ScriptedTokenizerRecvProxy(underlying=_FakeUnderlyingSocket())
        future = Future()
        future.set_exception(RuntimeError("Duplicate request ID detected: reused"))

        with self.assertRaisesRegex(RuntimeError, "Duplicate request ID detected"):
            proxy.wait_until_arrived(
                _is_start_req("reused"), timeout_s=0.02, post_future=future
            )

    def test_successful_post_still_requires_socket_arrival(self):
        proxy = ScriptedTokenizerRecvProxy(underlying=_FakeUnderlyingSocket())
        future = Future()
        future.set_result(None)

        with self.assertRaises(TimeoutError):
            proxy.wait_until_arrived(_is_control, timeout_s=0.02, post_future=future)

    def test_socket_arrival_does_not_wait_for_post_completion(self):
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        msg = _StartReq("pending")
        underlying.feed(msg)
        future = Future()

        proxy.wait_until_arrived(
            _is_start_req("pending"), timeout_s=0.02, post_future=future
        )

        self.assertFalse(future.done())
        self.assertIs(proxy.recv_pyobj(), msg)

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

        self.assertIs(proxy.recv_pyobj(), msg)

    def test_wait_until_arrived_skips_stale_same_type_object(self):
        proxy, _, _ = self._proxy_with_stale_control()

        with self.assertRaises(TimeoutError):
            proxy.wait_until_arrived(_is_control, timeout_s=0.05)

    def test_wait_until_arrived_returns_on_new_object_after_stale(self):
        proxy, underlying, stale = self._proxy_with_stale_control()
        fresh = _ControlMsg("fresh")
        underlying.feed_after_drain_cycles(fresh, cycles=1)

        proxy.wait_until_arrived(_is_control, timeout_s=2.0)

        self.assertIs(proxy.recv_pyobj(), stale)
        self.assertIs(proxy.recv_pyobj(), fresh)

    def test_wait_until_arrived_rid_predicate_ignores_stale_other_rid(self):
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        old = _StartReq(rid="old")
        underlying.feed(old)
        proxy.wait_until_arrived(_is_start_req("old"), timeout_s=1.0)

        new = _StartReq(rid="new")
        underlying.feed(new)
        proxy.wait_until_arrived(_is_start_req("new"), timeout_s=1.0)

        self.assertIs(proxy.recv_pyobj(), old)
        self.assertIs(proxy.recv_pyobj(), new)

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
