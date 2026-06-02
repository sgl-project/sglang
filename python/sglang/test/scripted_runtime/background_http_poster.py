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
        self._thread = threading.Thread(
            target=self._run_loop, name="scripted-runtime-async", daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit_coro(self, coro: Coroutine) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.add_done_callback(self._log_coro_exception)

    @staticmethod
    def _log_coro_exception(future: Future) -> None:
        try:
            future.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("scripted_runtime: background async coroutine failed")

    async def post(self, url: str, json: Any) -> None:
        session = self._ensure_session()
        async with session.post(url, json=json) as resp:
            await resp.read()

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
