"""Unit tests for scripted_runtime/background_http_poster — no real HTTP server.

Covers the four pieces of ``BackgroundHttpPoster`` that carry real logic: the
background event-loop/thread lifecycle, cross-thread coroutine dispatch, the
done-callback that logs and swallows coroutine failures, and the create-or-reuse
``aiohttp`` session branching. The two ``post_*`` coroutines run against a fake
session so the post call and body-draining are observable without any network.
"""

from __future__ import annotations

import asyncio
import threading
import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime import background_http_poster as bg_poster
from sglang.test.scripted_runtime.background_http_poster import BackgroundHttpPoster
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeResponse:
    """Stand-in aiohttp response: tracks whether the body was drained / read."""

    def __init__(self) -> None:
        self.read_called = False
        self.iter_any_consumed = False
        self._chunks = [b"chunk-1", b"chunk-2"]

    async def read(self) -> bytes:
        self.read_called = True
        return b"".join(self._chunks)

    @property
    def content(self) -> "_FakeResponse":
        return self

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk
        self.iter_any_consumed = True


class _FakePostCM:
    """Async context manager returned by a fake ``session.post(...)``."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class _FakeSession:
    """Records every ``post`` call and hands back one shared fake response."""

    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[str, object]] = []
        self.response = _FakeResponse()

    def post(self, url: str, json: object) -> _FakePostCM:
        self.calls.append((url, json))
        return _FakePostCM(self.response)

    async def close(self) -> None:
        self.closed = True


class TestBackgroundHttpPosterLifecycle(CustomTestCase):
    """__init__ spins up a running loop on a daemon thread; close tears it down."""

    def test_init_starts_running_loop_on_daemon_thread(self):
        """After construction the background loop is running on a live daemon thread."""
        poster = BackgroundHttpPoster()
        self.addCleanup(poster.close)

        self.assertIsNotNone(poster._loop)
        self.assertTrue(poster._loop.is_running())
        self.assertTrue(poster._thread.is_alive())
        self.assertTrue(poster._thread.daemon)

    def test_close_stops_loop_and_joins_thread(self):
        """close() stops the event loop and joins the background thread."""
        poster = BackgroundHttpPoster()

        poster.close()

        self.assertFalse(poster._loop.is_running())
        self.assertFalse(poster._thread.is_alive())

    def test_close_is_safe_when_loop_never_started(self):
        """close() returns early (no raise) when the loop was never created."""
        poster = BackgroundHttpPoster.__new__(BackgroundHttpPoster)
        poster._loop = None
        poster._thread = None
        poster._session = None

        poster.close()  # must not raise


class TestBackgroundHttpPosterSubmitCoro(CustomTestCase):
    """submit_coro dispatches onto the background loop and never lets it crash."""

    def test_submit_coro_runs_on_background_loop_thread(self):
        """The submitted coroutine executes on the named background loop thread."""
        poster = BackgroundHttpPoster()
        self.addCleanup(poster.close)
        done = threading.Event()
        recorded: dict[str, str] = {}

        async def record_thread() -> None:
            recorded["thread_name"] = threading.current_thread().name
            done.set()

        poster.submit_coro(record_thread())

        self.assertTrue(done.wait(timeout=5.0))
        self.assertEqual(recorded["thread_name"], "scripted-runtime-async")

    def test_log_coro_exception_logs_real_failure(self):
        """A coroutine that raises is reported via logger.exception, not propagated."""
        future: Future = Future()
        future.set_exception(RuntimeError("boom"))

        original = bg_poster.logger.exception
        bg_poster.logger.exception = MagicMock()
        try:
            BackgroundHttpPoster._log_coro_exception(future)
            bg_poster.logger.exception.assert_called_once()
        finally:
            bg_poster.logger.exception = original

    def test_log_coro_exception_swallows_cancellation_silently(self):
        """A cancelled coroutine is swallowed without logging."""
        future: Future = Future()
        future.set_exception(asyncio.CancelledError())

        original = bg_poster.logger.exception
        bg_poster.logger.exception = MagicMock()
        try:
            BackgroundHttpPoster._log_coro_exception(future)
            bg_poster.logger.exception.assert_not_called()
        finally:
            bg_poster.logger.exception = original

    def test_log_coro_exception_quiet_on_success(self):
        """A coroutine that completes cleanly logs nothing."""
        future: Future = Future()
        future.set_result(None)

        original = bg_poster.logger.exception
        bg_poster.logger.exception = MagicMock()
        try:
            BackgroundHttpPoster._log_coro_exception(future)
            bg_poster.logger.exception.assert_not_called()
        finally:
            bg_poster.logger.exception = original


class TestBackgroundHttpPosterEnsureSession(CustomTestCase):
    """_ensure_session lazily creates a session and recreates it once closed."""

    def test_ensure_session_creates_reuses_then_recreates_when_closed(self):
        """First call creates a session, second reuses it, a closed one is replaced."""
        poster = BackgroundHttpPoster()
        self.addCleanup(poster.close)
        sessions = [MagicMock(closed=False), MagicMock(closed=False)]
        original = bg_poster.aiohttp.ClientSession
        bg_poster.aiohttp.ClientSession = MagicMock(side_effect=sessions)
        try:
            first = poster._ensure_session()
            self.assertIs(first, sessions[0])

            reused = poster._ensure_session()
            self.assertIs(reused, sessions[0])

            sessions[0].closed = True
            recreated = poster._ensure_session()
            self.assertIs(recreated, sessions[1])
        finally:
            bg_poster.aiohttp.ClientSession = original
            # Drop the mock session so close() doesn't try to await a MagicMock.
            poster._session = None


class TestBackgroundHttpPosterPost(CustomTestCase):
    """post_and_drain / post_no_body POST the json and consume the response body."""

    def _run_on_loop(self, poster: BackgroundHttpPoster, coro) -> None:
        asyncio.run_coroutine_threadsafe(coro, poster._loop).result(timeout=5.0)

    def test_post_and_drain_posts_json_and_drains_body(self):
        """post_and_drain posts the payload and iterates the streamed body to completion."""
        poster = BackgroundHttpPoster()
        self.addCleanup(poster.close)
        session = _FakeSession()
        poster._ensure_session = lambda: session

        self._run_on_loop(poster, poster.post_and_drain("http://h/flush", {"a": 1}))

        self.assertEqual(session.calls, [("http://h/flush", {"a": 1})])
        self.assertTrue(session.response.iter_any_consumed)

    def test_post_no_body_posts_json_and_reads_response(self):
        """post_no_body posts the payload and reads the whole response body."""
        poster = BackgroundHttpPoster()
        self.addCleanup(poster.close)
        session = _FakeSession()
        poster._ensure_session = lambda: session

        self._run_on_loop(poster, poster.post_no_body("http://h/abort", {"b": 2}))

        self.assertEqual(session.calls, [("http://h/abort", {"b": 2})])
        self.assertTrue(session.response.read_called)


if __name__ == "__main__":
    unittest.main()
