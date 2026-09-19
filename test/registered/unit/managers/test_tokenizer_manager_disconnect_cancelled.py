"""
Regression test for TokenizerManager._is_request_disconnected.

Background: TokenizerManager._stream_one_response polls
`await request.is_disconnected()` (Starlette) every
SGLANG_REQUEST_STATE_WAIT_TIMEOUT seconds while waiting on a request. If the
client's TCP connection is torn down while that poll is in flight, uvicorn
cancels the pending ASGI receive, raising asyncio.CancelledError from inside
is_disconnected(). Since Python 3.8, CancelledError is a BaseException, not
an Exception, so it silently bypasses the `except Exception` handlers used
throughout the request-handling stack and propagates uncaught, which is
mistaken for a fatal engine error upstream and brings down the whole engine
process for what should have been a single aborted request (see #39216).

_is_request_disconnected() catches only that specific CancelledError and
distinguishes: cancellation of *our own* task (e.g. graceful shutdown, which
must keep propagating) from cancellation caused by the transport tearing
down the connection (which must be treated as "client disconnected" and
handled by the engine's existing per-request abort path). This test drives
that helper directly via mocks -- covering both the Python 3.11+
Task.cancelling() path and the Python 3.10 gracefully_exit-flag fallback --
plus an end-to-end check that a request whose is_disconnected() poll is hit
by a bare CancelledError does not crash the engine and a subsequent request
still completes normally.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _bare_tokenizer_manager(gracefully_exit: bool = False) -> TokenizerManager:
    """A TokenizerManager stand-in exposing only what
    _is_request_disconnected reads, without running __init__ (which spins up
    the full engine: scheduler processes, tokenizers, zmq sockets, ...)."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.gracefully_exit = gracefully_exit
    return tm


class FakeTaskWithCancelling:
    """Simulates a Python 3.11+ asyncio.Task exposing .cancelling()."""

    def __init__(self, cancelling_count: int):
        self._n = cancelling_count

    def cancelling(self) -> int:
        return self._n


class FakeTaskWithoutCancelling:
    """Simulates a Python 3.10 asyncio.Task (no .cancelling() method)."""


class TestIsRequestDisconnectedHandlesCancelledError(CustomTestCase):
    """Unit-level coverage of the CancelledError-vs-disconnect branching."""

    def test_disconnected_client_no_cancellation(self):
        tm = _bare_tokenizer_manager()
        request = AsyncMock()
        request.is_disconnected.return_value = True
        self.assertTrue(asyncio.run(tm._is_request_disconnected(request)))

    def test_connected_client_no_cancellation(self):
        tm = _bare_tokenizer_manager()
        request = AsyncMock()
        request.is_disconnected.return_value = False
        self.assertFalse(asyncio.run(tm._is_request_disconnected(request)))

    def test_transport_teardown_cancelled_error_is_swallowed(self):
        """The core bug scenario: uvicorn cancels the pending receive when
        the client's socket closes mid-poll. Our own task was not asked to
        cancel (Task.cancelling() == 0), so this must be treated as a normal
        disconnect, not re-raised as an uncaught BaseException."""
        tm = _bare_tokenizer_manager(gracefully_exit=False)
        request = AsyncMock()
        request.is_disconnected.side_effect = asyncio.CancelledError()

        async def run():
            with patch("asyncio.current_task", return_value=FakeTaskWithCancelling(0)):
                return await tm._is_request_disconnected(request)

        self.assertTrue(asyncio.run(run()))

    def test_real_task_cancellation_still_propagates(self):
        """When our own task has a genuine pending cancellation (e.g. server
        shutdown), Task.cancelling() > 0 and the CancelledError must
        propagate -- swallowing it here would risk hanging shutdown."""
        tm = _bare_tokenizer_manager(gracefully_exit=False)
        request = AsyncMock()
        request.is_disconnected.side_effect = asyncio.CancelledError()

        async def run():
            with patch("asyncio.current_task", return_value=FakeTaskWithCancelling(1)):
                await tm._is_request_disconnected(request)

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(run())

    def test_py310_fallback_transport_teardown(self):
        """Without Task.cancelling() (Python 3.10), fall back to the
        engine's own gracefully_exit flag. Not shutting down -> disconnect."""
        tm = _bare_tokenizer_manager(gracefully_exit=False)
        request = AsyncMock()
        request.is_disconnected.side_effect = asyncio.CancelledError()

        async def run():
            with patch(
                "asyncio.current_task", return_value=FakeTaskWithoutCancelling()
            ):
                return await tm._is_request_disconnected(request)

        self.assertTrue(asyncio.run(run()))

    def test_py310_fallback_shutdown_propagates(self):
        """Without Task.cancelling(), a gracefully_exit shutdown in progress
        must still propagate the cancellation."""
        tm = _bare_tokenizer_manager(gracefully_exit=True)
        request = AsyncMock()
        request.is_disconnected.side_effect = asyncio.CancelledError()

        async def run():
            with patch(
                "asyncio.current_task", return_value=FakeTaskWithoutCancelling()
            ):
                await tm._is_request_disconnected(request)

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(run())


class TestStreamOneResponseSurvivesDisconnectCancellation(CustomTestCase):
    """End-to-end (still process-local, no real server/model) check that a
    disconnect-triggered CancelledError aborts only the one request and
    leaves the manager able to serve a following request."""

    def test_one_request_aborted_next_request_unaffected(self):
        tm = _bare_tokenizer_manager(gracefully_exit=False)
        tm.abort_request = MagicMock()

        class FakeObj:
            rid = "req-1"
            background = False

        class FakeState:
            def __init__(self):
                self.event = asyncio.Event()
                self.out_list = []
                self.finished = False

        state = FakeState()
        request = AsyncMock()
        request.is_disconnected.side_effect = asyncio.CancelledError()

        async def drive():
            # The real bug fires while _stream_one_response is polling via
            # asyncio.wait_for(..., timeout=_REQUEST_STATE_WAIT_TIMEOUT)
            # (default 4s) waiting for a long-running request. Shrink that
            # timeout so the test hits the same code path without a real
            # multi-second wait.
            with (
                patch(
                    "sglang.srt.managers.tokenizer_manager._REQUEST_STATE_WAIT_TIMEOUT",
                    0.01,
                ),
                patch("asyncio.current_task", return_value=FakeTaskWithCancelling(0)),
            ):
                gen = TokenizerManager._stream_one_response(
                    tm, obj=FakeObj(), state=state, request=request
                )
                with self.assertRaises(ValueError):
                    await gen.__anext__()

        asyncio.run(drive())

        # The single disconnected request was aborted through the engine's
        # existing per-request cleanup path...
        tm.abort_request.assert_called_once_with("req-1")

        # ...and the manager itself is still alive: nothing here touched
        # process state, so a subsequent, independent request's polling
        # helper still behaves normally.
        next_request = AsyncMock()
        next_request.is_disconnected.return_value = False
        self.assertFalse(asyncio.run(tm._is_request_disconnected(next_request)))


if __name__ == "__main__":
    unittest.main()
