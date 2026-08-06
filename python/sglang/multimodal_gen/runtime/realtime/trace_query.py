# SPDX-License-Identifier: Apache-2.0

"""CloudWatch-backed query plane for realtime traces."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
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
_CLOUDWATCH_QUERY_LIMIT = 10_000
_STAGES = (
    ("browser", "Browser"),
    ("gateway", "Gateway"),
    ("api", "Realtime API"),
    ("scheduler", "Scheduler"),
    ("vae_encode", "VAE Encode"),
    ("denoise", "Denoising"),
    ("vae_decode", "VAE Decode"),
    ("transport", "Transport"),
    ("frontend", "Frontend"),
)


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
        self._cache: OrderedDict[tuple[str, int], _CacheEntry] = OrderedDict()
        self._last_good: dict[tuple[str, int], dict[str, Any]] = {}
        self._inflight: dict[tuple[str, int], asyncio.Task[dict[str, Any]]] = {}
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
        key = (trace_id, window_s)
        async with self._lock:
            now = self._clock()
            cached = self._cache.get(key)
            if cached is not None and cached.expires_at > now:
                self._cache.move_to_end(key)
                base = cached.value
                task = None
            else:
                base = None
                task = self._inflight.get(key)
                if task is None:
                    task = asyncio.create_task(
                        self._query_and_cache(
                            key,
                            trace_id,
                            window_s=window_s,
                        )
                    )
                    self._inflight[key] = task

        if base is None:
            assert task is not None
            base = await asyncio.shield(task)
        return self._project(base, after=after, limit=limit)

    async def _query_and_cache(
        self,
        key: tuple[str, int],
        trace_id: str,
        *,
        window_s: int,
    ) -> dict[str, Any]:
        task = asyncio.current_task()
        try:
            try:
                async with self._query_semaphore:
                    value = await asyncio.to_thread(
                        self._query_sync,
                        trace_id,
                        window_s=window_s,
                    )
            except Exception:
                async with self._lock:
                    previous = self._last_good.get(key)
                if previous is None:
                    raise
                value = self._stale_copy(previous, reason="query_failed")

            if not self._has_stage_samples(value):
                async with self._lock:
                    previous = self._last_good.get(key)
                if previous is not None:
                    value = self._stale_copy(previous, reason="no_results")

            async with self._lock:
                if not value.get("stale") and self._has_stage_samples(value):
                    self._last_good[key] = value
                self._cache[key] = _CacheEntry(
                    self._clock() + self.cache_ttl_s,
                    value,
                )
                self._cache.move_to_end(key)
                while len(self._cache) > self.max_cache_entries:
                    evicted_key, _ = self._cache.popitem(last=False)
                    self._last_good.pop(evicted_key, None)
            return value
        finally:
            async with self._lock:
                if self._inflight.get(key) is task:
                    self._inflight.pop(key, None)

    def _query_sync(
        self,
        trace_id: str,
        *,
        window_s: int,
    ) -> dict[str, Any]:
        end_time = int(self._wall_clock())
        # ``_TRACE_ID`` already rejects regex operators; only ``.`` remains
        # special in the accepted alphabet and must be treated literally.
        escaped_trace_id = trace_id.replace(".", r"\.")
        query_string = (
            "fields @timestamp, @message, @ptr "
            f"| filter @message like /{escaped_trace_id}/ "
            "| sort @timestamp desc "
            f"| limit {_CLOUDWATCH_QUERY_LIMIT}"
        )
        started = self._call_logs_api(
            self.logs_client.start_query,
            logGroupName=self.log_group,
            startTime=end_time - window_s,
            endTime=end_time,
            queryString=query_string,
            limit=_CLOUDWATCH_QUERY_LIMIT,
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
            events.append(event)
        events.sort(key=lambda event: int(event.get("trace_seq") or 0))
        return {
            "trace_id": trace_id,
            "events": events[-1_000:],
            "stages": self._aggregate_stages(events),
            "stale": False,
            "observed_at": self._format_observed_at(end_time),
            "window": {
                "seconds": window_s,
                "start_epoch_ms": (end_time - window_s) * 1_000,
                "end_epoch_ms": end_time * 1_000,
            },
        }

    @staticmethod
    def _project(
        base: dict[str, Any], *, after: int, limit: int
    ) -> dict[str, Any]:
        result = {key: value for key, value in base.items() if key != "events"}
        all_events = base.get("events", [])
        events = [
            event
            for event in all_events
            if int(event.get("trace_seq") or 0) > after
        ]
        result["events"] = events[-limit:]
        result["next_cursor"] = max(
            [after, *(int(event.get("trace_seq") or 0) for event in all_events)]
        )
        result["window_s"] = int(base.get("window", {}).get("seconds") or 300)
        return result

    @staticmethod
    def _stale_copy(value: dict[str, Any], *, reason: str) -> dict[str, Any]:
        stale = copy.deepcopy(value)
        stale["stale"] = True
        stale["stale_reason"] = reason
        stale["events"] = []
        return stale

    @staticmethod
    def _has_stage_samples(value: dict[str, Any]) -> bool:
        return any(
            int(stage.get("count") or 0) > 0
            for stage in value.get("stages", [])
        )

    @staticmethod
    def _format_observed_at(epoch_s: int) -> str:
        return (
            datetime.fromtimestamp(epoch_s, tz=UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    @classmethod
    def _aggregate_stages(cls, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        samples: dict[str, dict[str, tuple[int, float]]] = {
            stage_id: {} for stage_id, _ in _STAGES
        }
        pending_browser_start_ms: float | None = None

        def put(
            stage_id: str,
            event: dict[str, Any],
            value: Any,
            *,
            priority: int = 1,
            sample_key: str | None = None,
        ) -> None:
            number = cls._number(value)
            if number is None or number < 0:
                return
            key = sample_key or cls._sample_key(event)
            previous = samples[stage_id].get(key)
            if previous is None or priority >= previous[0]:
                samples[stage_id][key] = (priority, number)

        for event in events:
            event_name = str(event.get("event") or event.get("name") or "")
            if event_name == "client.generate_clicked":
                pending_browser_start_ms = cls._event_epoch_ms(event)
                continue
            if event_name == "client.ws_open" and pending_browser_start_ms is not None:
                opened_ms = cls._event_epoch_ms(event)
                if opened_ms is not None:
                    put(
                        "browser",
                        event,
                        opened_ms - pending_browser_start_ms,
                        sample_key=(
                            f"browser:{event.get('trace_seq', len(samples['browser']))}"
                        ),
                    )
                pending_browser_start_ms = None
                continue
            if event_name == "gateway.coordinator_admit_complete":
                put("gateway", event, event.get("coordinator_admit_ms"))
                continue
            if event_name == "server.chunk_complete":
                put("api", event, event.get("request_prepare_ms"), priority=2)
                put("scheduler", event, event.get("scheduler_forward_ms"), priority=2)
                transport_ms = cls._sum_numbers(
                    event.get("raw_payload_build_ms"), event.get("ws_write_ms")
                )
                put("transport", event, transport_ms, priority=2)
                continue
            if event_name == "server.scheduler_forward_done":
                put("scheduler", event, cls._duration(event))
                continue
            if event_name == "server.vae_encode_complete":
                put("vae_encode", event, cls._duration(event), priority=2)
                continue
            if event_name == "server.model_denoise_complete":
                put("denoise", event, cls._duration(event), priority=2)
                continue
            if event_name == "server.vae_decode_complete":
                put("vae_decode", event, cls._duration(event), priority=3)
                continue
            if event_name == "server.remote_vae_complete":
                put("vae_decode", event, event.get("vae_decode_ms"), priority=2)
                continue
            if event_name == "server.frame_transfer_complete":
                put("transport", event, cls._duration(event), priority=1)
                continue
            if event_name == "client.chunk_first_rendered":
                put(
                    "frontend",
                    event,
                    cls._sum_numbers(
                        event.get("decode_ms"), event.get("display_lag_ms")
                    ),
                    priority=2,
                )
                continue
            if event_name == "server.pipeline_stage_complete":
                stage_name = str(
                    event.get("stage") or event.get("component") or ""
                ).lower()
                if "denois" in stage_name:
                    put("denoise", event, cls._duration(event))
                elif "vae" in stage_name and "encod" in stage_name:
                    put("vae_encode", event, cls._duration(event))
                elif "vae" in stage_name and "decod" in stage_name:
                    put("vae_decode", event, cls._duration(event))

        stage_results = []
        for stage_id, title in _STAGES:
            values = [sample[1] for sample in samples[stage_id].values()]
            stage_results.append(cls._stage_summary(stage_id, title, values))
        return stage_results

    @staticmethod
    def _sample_key(event: dict[str, Any]) -> str:
        chunk_index = event.get("chunk_index")
        event_id = event.get("event_id")
        if chunk_index is not None:
            return f"chunk:{chunk_index}:event:{event_id}"
        return f"seq:{event.get('trace_seq', event.get('server_epoch_ms', id(event)))}"

    @staticmethod
    def _event_epoch_ms(event: dict[str, Any]) -> float | None:
        for field in ("client_epoch_ms", "server_epoch_ms"):
            value = CloudWatchTraceQuery._number(event.get(field))
            if value is not None:
                return value
        return None

    @staticmethod
    def _duration(event: dict[str, Any]) -> float | None:
        return CloudWatchTraceQuery._number(
            event.get("cuda_ms")
            if event.get("cuda_ms") is not None
            else event.get("duration_ms")
        )

    @staticmethod
    def _sum_numbers(*values: Any) -> float | None:
        numbers = [CloudWatchTraceQuery._number(value) for value in values]
        present = [number for number in numbers if number is not None]
        return sum(present) if present else None

    @staticmethod
    def _number(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _stage_summary(
        cls, stage_id: str, title: str, values: list[float]
    ) -> dict[str, Any]:
        ordered = sorted(values)
        count = len(ordered)
        if not count:
            return {
                "id": stage_id,
                "title": title,
                "count": 0,
                "avg_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "max_ms": None,
            }

        def percentile(ratio: float) -> float:
            index = max(0, math.ceil(count * ratio) - 1)
            return ordered[index]

        return {
            "id": stage_id,
            "title": title,
            "count": count,
            "avg_ms": cls._rounded(sum(ordered) / count),
            "p50_ms": cls._rounded(percentile(0.5)),
            "p95_ms": cls._rounded(percentile(0.95)),
            "max_ms": cls._rounded(ordered[-1]),
        }

    @staticmethod
    def _rounded(value: float) -> int | float:
        rounded = round(value, 3)
        return int(rounded) if rounded.is_integer() else rounded

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
