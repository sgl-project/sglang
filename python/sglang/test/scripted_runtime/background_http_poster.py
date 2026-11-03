from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, Optional

import aiohttp
import orjson

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

    def submit_coro(self, coro: Coroutine) -> Future:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.add_done_callback(self._log_coro_exception)
        return future

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
            if resp.status >= 400:
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=await resp.text(),
                    headers=resp.headers,
                )
            if resp.content_type != "text/event-stream":
                await resp.read()
                return

            # /generate can report validation failures in HTTP 200 SSE events.
            pending = b""
            # Avoid readline's size limit for large logprob events.
            async for chunk in resp.content.iter_any():
                lines = (pending + chunk).split(b"\n")
                pending = lines.pop()
                for line in lines:
                    self._check_stream_error(url=url, line=line)
            if pending:
                self._check_stream_error(url=url, line=pending)

    @staticmethod
    def _check_stream_error(url: str, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            return
        payload = orjson.loads(data)
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(f"POST {url} failed: {payload['error']}")

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
