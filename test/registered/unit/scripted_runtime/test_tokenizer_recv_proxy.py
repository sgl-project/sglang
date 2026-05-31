"""Unit tests for scripted_runtime/tokenizer_recv_proxy — no server, no zmq sockets.

Drives ``ScriptedTokenizerRecvProxy`` against a fake PULL socket so every branch
of the buffering / staleness logic is exercised on CPU. The headline regression
is ``test_wait_until_arrived_skips_stale_same_type_object``: it pins the
"two same-type control verbs back-to-back, no yield between" scenario from
``2026-05-31-tokenizer-recv-proxy-start-len-ut.md`` — the second wait must keep
waiting for its own object instead of matching the first verb's leftover.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
    ScriptedTokenizerRecvProxy,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest


@dataclass
class _ControlMsg:
    """Stands in for a typed control verb (e.g. FlushCacheReqInput) matched by type."""

    tag: str = "flush"


@dataclass
class _StartReq:
    """Stands in for a request injected by start_req, matched by rid."""

    rid: str


class _FakeUnderlyingSocket:
    """zmq PULL stand-in: ``recv_pyobj`` drains a ready queue, raises EAGAIN when empty.

    ``feed`` makes an object immediately drainable; ``feed_after_drain_cycles`` delays
    an object until N drain cycles (EAGAIN raises) have elapsed, modeling an in-flight
    HTTP POST that only lands on the socket a few scheduler loops later.
    """

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
    """recv_pyobj drains the underlying socket, then pops buffered items FIFO."""

    def test_recv_pyobj_drains_then_pops_fifo(self):
        """Two fed objects come back in arrival order across successive recvs."""
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        first, second = _ControlMsg("a"), _ControlMsg("b")
        underlying.feed(first)
        underlying.feed(second)

        self.assertIs(proxy.recv_pyobj(), first)
        self.assertIs(proxy.recv_pyobj(), second)

    def test_recv_pyobj_empty_noblock_raises_eagain(self):
        """An empty buffer under NOBLOCK raises the same EAGAIN a real PULL socket would."""
        proxy = ScriptedTokenizerRecvProxy(underlying=_FakeUnderlyingSocket())

        with self.assertRaises(zmq.ZMQError) as ctx:
            proxy.recv_pyobj(zmq.NOBLOCK)
        self.assertEqual(ctx.exception.errno, zmq.EAGAIN)

    def test_recv_pyobj_empty_blocking_raises_runtime_error(self):
        """Blocking recv on an empty buffer is unsupported and raises RuntimeError."""
        proxy = ScriptedTokenizerRecvProxy(underlying=_FakeUnderlyingSocket())

        with self.assertRaisesRegex(RuntimeError, "blocking recv is not supported"):
            proxy.recv_pyobj()


class TestScriptedTokenizerRecvProxyWaitUntilArrived(CustomTestCase):
    """wait_until_arrived blocks for a *newly* drained match and never pops it."""

    def _proxy_with_stale_control(self):
        """Return (proxy, underlying, stale) with one already-buffered _ControlMsg."""
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        stale = _ControlMsg("stale")
        underlying.feed(stale)
        # First control verb drains `stale` into the buffer without popping it.
        proxy.wait_until_arrived(_is_control, timeout_s=1.0)
        return proxy, underlying, stale

    def test_wait_until_arrived_returns_on_first_match_when_buffer_empty(self):
        """With an empty buffer, the first matching object satisfies the wait and stays buffered."""
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        msg = _ControlMsg("first")
        underlying.feed(msg)

        proxy.wait_until_arrived(_is_control, timeout_s=1.0)

        # Non-destructive: the scheduler still observes it on the next recv.
        self.assertIs(proxy.recv_pyobj(), msg)

    def test_wait_until_arrived_skips_stale_same_type_object(self):
        """A same-type object already in the buffer does NOT satisfy a later wait (start_seq guard)."""
        proxy, _, _ = self._proxy_with_stale_control()

        # Second same-type verb, nothing new on the socket: must time out, not
        # falsely match the leftover from the first verb. Deleting the start_seq
        # snapshot would make this return immediately and fail.
        with self.assertRaises(TimeoutError):
            proxy.wait_until_arrived(_is_control, timeout_s=0.05)

    def test_wait_until_arrived_returns_on_new_object_after_stale(self):
        """A second wait returns only once its own object lands, then both stay buffered FIFO."""
        proxy, underlying, stale = self._proxy_with_stale_control()
        fresh = _ControlMsg("fresh")
        underlying.feed_after_drain_cycles(fresh, cycles=1)

        proxy.wait_until_arrived(_is_control, timeout_s=2.0)

        # Both the stale and the fresh control message survive, in arrival order.
        self.assertIs(proxy.recv_pyobj(), stale)
        self.assertIs(proxy.recv_pyobj(), fresh)

    def test_wait_until_arrived_rid_predicate_ignores_stale_other_rid(self):
        """A start_req rid predicate skips a buffered request with a different rid."""
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
        """Re-using a rid does not match the leftover same-rid request from an earlier verb."""
        underlying = _FakeUnderlyingSocket()
        proxy = ScriptedTokenizerRecvProxy(underlying=underlying)
        underlying.feed(_StartReq(rid="reused"))
        # First start_req drains the stale same-rid request into the buffer.
        proxy.wait_until_arrived(_is_start_req("reused"), timeout_s=1.0)

        # A second start_req re-using that rid must wait for its own object, not
        # the buffered leftover (the start_seq guard protects rid-reuse too).
        with self.assertRaises(TimeoutError):
            proxy.wait_until_arrived(_is_start_req("reused"), timeout_s=0.05)

    def test_wait_until_arrived_timeout_message_names_description(self):
        """The timeout error embeds the caller-supplied description and timeout."""
        proxy = ScriptedTokenizerRecvProxy(underlying=_FakeUnderlyingSocket())

        with self.assertRaisesRegex(TimeoutError, "FlushCacheReqInput"):
            proxy.wait_until_arrived(
                _is_control, timeout_s=0.02, description="FlushCacheReqInput"
            )


if __name__ == "__main__":
    unittest.main()
