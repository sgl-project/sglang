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

    def __init__(self) -> None:
        self.read_called = False

    async def read(self) -> bytes:
        self.read_called = True
        return b"chunk-1chunk-2"


class _FakePostCM:

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class _FakeSession:

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

    def test_init_starts_running_loop_on_daemon_thread(self):
        poster = BackgroundHttpPoster()
        self.addCleanup(poster.close)

        self.assertIsNotNone(poster._loop)
        self.assertTrue(poster._loop.is_running())
        self.assertTrue(poster._thread.is_alive())
        self.assertTrue(poster._thread.daemon)

    def test_close_stops_loop_and_joins_thread(self):
        poster = BackgroundHttpPoster()

        poster.close()

        self.assertFalse(poster._loop.is_running())
        self.assertFalse(poster._thread.is_alive())

    def test_close_is_safe_when_loop_never_started(self):
        poster = BackgroundHttpPoster.__new__(BackgroundHttpPoster)
        poster._loop = None
        poster._thread = None
        poster._session = None

        poster.close()


class TestBackgroundHttpPosterSubmitCoro(CustomTestCase):

    def test_submit_coro_runs_on_background_loop_thread(self):
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

    def test_ensure_session_creates_reuses_then_recreates_when_closed(self):
        poster = BackgroundHttpPoster()
        self.addCleanup(poster.close)
        sessions = [MagicMock(closed=False), MagicMock(closed=False)]
        original = bg_poster.aiohttp.ClientSession
        original_connector = bg_poster.aiohttp.TCPConnector
        bg_poster.aiohttp.ClientSession = MagicMock(side_effect=sessions)
        bg_poster.aiohttp.TCPConnector = MagicMock()
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
            bg_poster.aiohttp.TCPConnector = original_connector
            poster._session = None


class TestBackgroundHttpPosterPost(CustomTestCase):

    def _run_on_loop(self, poster: BackgroundHttpPoster, coro) -> None:
        asyncio.run_coroutine_threadsafe(coro, poster._loop).result(timeout=5.0)

    def test_post_posts_json_and_reads_body(self):
        poster = BackgroundHttpPoster()
        self.addCleanup(poster.close)
        session = _FakeSession()
        poster._ensure_session = lambda: session

        self._run_on_loop(poster, poster.post("http://h/flush", {"a": 1}))

        self.assertEqual(session.calls, [("http://h/flush", {"a": 1})])
        self.assertTrue(session.response.read_called)


if __name__ == "__main__":
    unittest.main()
