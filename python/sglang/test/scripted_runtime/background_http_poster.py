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

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._loop = asyncio.new_event_loop()
        # Control POSTs are fire-and-forget so the engine can keep stepping, so a
        # rejected request would otherwise surface only as an unrelated
        # "nothing arrived on the socket" timeout. Record failures here and let
        # the waiter report the real cause.
        self._failures: list[str] = []
        self._failures_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run_loop, name="scripted-runtime-async", daemon=True
        )
        self._thread.start()

    def take_failures(self) -> list[str]:
        """Failed POSTs recorded since the last call (drains the list)."""
        with self._failures_lock:
            failures = self._failures[:]
            self._failures.clear()
        return failures

    def _record_failure(self, message: str) -> None:
        with self._failures_lock:
            self._failures.append(message)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit_coro(self, coro: Coroutine) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.add_done_callback(self._log_coro_exception)

    def _log_coro_exception(self, future: Future) -> None:
        try:
            future.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("scripted_runtime: background async coroutine failed")
            self._record_failure(f"{type(e).__name__}: {e}")

    async def post(self, url: str, json: Any) -> None:
        session = self._ensure_session()
        async with session.post(url, json=json) as resp:
            body = await resp.read()
            if resp.status >= 400:
                raise RuntimeError(
                    f"POST {url} -> HTTP {resp.status}: "
                    f"{body.decode(errors='replace')[:500]}"
                )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0)
            )
        return self._session

    def close(self) -> None:
        try:
            if self._session is not None:
                future = asyncio.run_coroutine_threadsafe(
                    self._session.close(), self._loop
                )
                future.result(timeout=JOIN_TIMEOUT_S)
        except Exception:
            logger.exception("scripted_runtime: failed to close aiohttp session")
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=JOIN_TIMEOUT_S)
        except Exception:
            logger.exception("scripted_runtime: failed to stop background async loop")
