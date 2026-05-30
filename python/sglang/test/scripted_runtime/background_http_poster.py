"""BackgroundHttpPoster: one async loop + thread + aiohttp session for posts.

Scripted HTTP posts (``/generate``, control verbs) are fire-and-forget: the
scheduler must never block on the HTTP response. This wraps a single daemon
thread running a forever asyncio event loop plus one shared
:class:`aiohttp.ClientSession`, replacing the old thread-per-request model.
Synchronous callers (the scheduler-side hook / context) schedule post
coroutines on the shared loop via :meth:`submit_coro`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, Optional

import aiohttp

logger = logging.getLogger(__name__)

JOIN_TIMEOUT_S: float = 10.0


class BackgroundHttpPoster:
    """Owns the shared async loop / thread / session for fire-and-forget posts.

    The loop is created inside the thread so its owning thread is unambiguous.
    The session is created lazily on that loop (an :class:`aiohttp.ClientSession`
    binds to the running event loop).
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._start()

    def _start(self) -> None:
        loop_ready = threading.Event()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop_ready.set()
            loop.run_forever()

        self._thread = threading.Thread(
            target=_run, name="scripted-runtime-async", daemon=True
        )
        self._thread.start()
        loop_ready.wait()

    def submit_coro(self, coro: Coroutine) -> None:
        """Schedule ``coro`` on the shared loop fire-and-forget.

        Does not block on the result. A done-callback logs any non-cancellation
        exception so a failed post never fails silently.
        """
        assert self._loop is not None, "background async loop is not started"
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.add_done_callback(self._log_coro_exception)

    @staticmethod
    def _log_coro_exception(future: Future) -> None:
        try:
            future.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("scripted_runtime: background async coroutine failed")

    async def post_and_drain(self, url: str, json: Any) -> None:
        """POST ``json`` to ``url`` (streaming) and discard the response body.

        The scheduler must never block on the HTTP response, so the streamed
        ``/generate`` body is drained and dropped chunk-by-chunk on the loop.
        """
        session = self._ensure_session()
        async with session.post(url, json=json) as resp:
            async for _ in resp.content.iter_any():
                pass

    async def post_no_body(self, url: str, json: Any) -> None:
        """POST ``json`` to ``url`` and read (discard) the full response.

        For non-streaming control endpoints whose response is small; the body
        is read so the connection is released back to the session pool.
        """
        session = self._ensure_session()
        async with session.post(url, json=json) as resp:
            await resp.read()

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Return the shared session, creating it on the running loop.

        Must be called from inside a coroutine running on the loop, since
        :class:`aiohttp.ClientSession` binds to the running event loop.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def close(self) -> None:
        """Best-effort close the shared session and stop the loop / thread."""
        loop = self._loop
        if loop is None:
            return
        try:
            session = self._session
            if session is not None:
                close_future = asyncio.run_coroutine_threadsafe(session.close(), loop)
                close_future.result(timeout=JOIN_TIMEOUT_S)
        except Exception:
            logger.exception("scripted_runtime: failed to close aiohttp session")
        try:
            loop.call_soon_threadsafe(loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=JOIN_TIMEOUT_S)
        except Exception:
            logger.exception("scripted_runtime: failed to stop background async loop")
