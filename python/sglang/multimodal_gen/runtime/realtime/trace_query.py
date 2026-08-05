# SPDX-License-Identifier: Apache-2.0

"""CloudWatch-backed query plane for realtime traces."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable


_TRACE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_RETRYABLE_LOGS_ERROR_CODES = {
    "InternalFailure",
    "InternalServiceError",
    "LimitExceededException",
    "ServiceUnavailableException",
    "Throttling",
    "ThrottlingException",
}


@dataclass(slots=True)
class _CacheEntry:
    expires_at: float
    value: dict[str, Any]


class CloudWatchTraceQuery:
    def __init__(
        self,
        logs_client,
        *,
        log_group: str,
        cache_ttl_s: float = 15.0,
        query_timeout_s: float = 4.0,
        max_cache_entries: int = 256,
        max_concurrent_queries: int = 4,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if not log_group:
            raise ValueError("log_group is required")
        if cache_ttl_s <= 0 or query_timeout_s <= 0:
            raise ValueError("trace query timeouts must be positive")
        if max_concurrent_queries <= 0:
            raise ValueError("max_concurrent_queries must be positive")
        self.logs_client = logs_client
        self.log_group = log_group
        self.cache_ttl_s = cache_ttl_s
        self.query_timeout_s = query_timeout_s
        self.max_cache_entries = max(1, max_cache_entries)
        self._clock = clock
        self._wall_clock = wall_clock
        self._cache: OrderedDict[tuple[str, int, int, int], _CacheEntry] = (
            OrderedDict()
        )
        self._inflight: dict[
            tuple[str, int, int, int], asyncio.Task[dict[str, Any]]
        ] = {}
        self._lock = asyncio.Lock()
        self._query_semaphore = asyncio.Semaphore(max_concurrent_queries)

    async def query(
        self,
        trace_id: str,
        *,
        after: int = 0,
        limit: int = 220,
        window_s: int = 300,
    ) -> dict[str, Any]:
        if not _TRACE_ID.fullmatch(trace_id):
            raise ValueError("invalid trace_id")
        after = max(0, int(after))
        limit = min(500, max(1, int(limit)))
        window_s = min(3600, max(30, int(window_s)))
        key = (trace_id, after, limit, window_s)
        async with self._lock:
            now = self._clock()
            cached = self._cache.get(key)
            if cached is not None and cached.expires_at > now:
                self._cache.move_to_end(key)
                return cached.value
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(
                    self._query_and_cache(
                        key,
                        trace_id,
                        after=after,
                        limit=limit,
                        window_s=window_s,
                    )
                )
                self._inflight[key] = task

        return await asyncio.shield(task)

    async def _query_and_cache(
        self,
        key: tuple[str, int, int, int],
        trace_id: str,
        *,
        after: int,
        limit: int,
        window_s: int,
    ) -> dict[str, Any]:
        task = asyncio.current_task()
        try:
            async with self._query_semaphore:
                value = await asyncio.to_thread(
                    self._query_sync,
                    trace_id,
                    after=after,
                    limit=limit,
                    window_s=window_s,
                )

            async with self._lock:
                self._cache[key] = _CacheEntry(
                    self._clock() + self.cache_ttl_s,
                    value,
                )
                self._cache.move_to_end(key)
                while len(self._cache) > self.max_cache_entries:
                    self._cache.popitem(last=False)
            return value
        finally:
            async with self._lock:
                if self._inflight.get(key) is task:
                    self._inflight.pop(key, None)

    def _query_sync(
        self,
        trace_id: str,
        *,
        after: int,
        limit: int,
        window_s: int,
    ) -> dict[str, Any]:
        end_time = int(self._wall_clock())
        query_string = (
            "fields @timestamp, @message, @ptr "
            f"| filter @message like /{trace_id}/ "
            "| sort @timestamp desc "
            f"| limit {limit}"
        )
        started = self._call_logs_api(
            self.logs_client.start_query,
            logGroupName=self.log_group,
            startTime=end_time - window_s,
            endTime=end_time,
            queryString=query_string,
            limit=limit,
        )
        query_id = started["queryId"]
        deadline = time.monotonic() + self.query_timeout_s
        while True:
            result = self._call_logs_api(
                self.logs_client.get_query_results, queryId=query_id
            )
            status = result.get("status")
            if status == "Complete":
                break
            if status in {"Failed", "Cancelled", "Timeout", "Unknown"}:
                raise RuntimeError(f"CloudWatch trace query ended with {status}")
            if time.monotonic() >= deadline:
                stopper = getattr(self.logs_client, "stop_query", None)
                if stopper is not None:
                    stopper(queryId=query_id)
                raise TimeoutError("CloudWatch trace query timed out")
            time.sleep(0.05)

        events = []
        for row in result.get("results", []):
            fields = {entry.get("field"): entry.get("value") for entry in row}
            event = self._parse_message(fields.get("@message", ""))
            if event is None or event.get("trace_id") != trace_id:
                continue
            cursor = int(
                event.get("trace_seq")
                or event.get("server_epoch_ms")
                or event.get("client_epoch_ms")
                or 0
            )
            event["trace_seq"] = cursor
            if cursor <= after:
                continue
            events.append(event)
        events.sort(key=lambda event: int(event.get("trace_seq") or 0))
        next_cursor = max(
            [after, *(int(event.get("trace_seq") or 0) for event in events)]
        )
        return {
            "trace_id": trace_id,
            "events": events[-limit:],
            "next_cursor": next_cursor,
            "window_s": window_s,
        }

    @staticmethod
    def _call_logs_api(method, **kwargs):
        for attempt in range(3):
            try:
                return method(**kwargs)
            except Exception as exc:
                response = getattr(exc, "response", {})
                code = response.get("Error", {}).get("Code")
                if code not in _RETRYABLE_LOGS_ERROR_CODES or attempt == 2:
                    raise
                time.sleep(0.05 * (2**attempt))
        raise RuntimeError("unreachable CloudWatch retry state")

    @staticmethod
    def _parse_message(message: str) -> dict[str, Any] | None:
        value: dict[str, Any] | None = None
        for _ in range(3):
            start = message.find("{")
            if start < 0:
                return value
            try:
                parsed = json.loads(message[start:])
            except json.JSONDecodeError:
                return value
            if not isinstance(parsed, dict):
                return value
            value = parsed
            if parsed.get("trace_id") is not None:
                return parsed

            nested = parsed.get("log")
            if not isinstance(nested, str):
                return parsed
            message = nested
        return value
