"""Replay an InferCast trace against a real SGLang server."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from infercast_trace import TraceRequest, file_identity, load_trace_requests


@dataclass
class RequestMeasurement:
    request_index: int
    success: bool
    scheduled_timestamp_ms: float
    actual_start_timestamp_ms: float
    input_length: int
    requested_output_length: int
    output_length: int
    cached_tokens: int
    cached_tokens_details: dict[str, Any] | None
    latency_ms: float
    ttft_ms: float
    itl_ms: list[float]
    metadata: dict[str, Any]
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--warmup-trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    parser.add_argument(
        "--flush-cache", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--coalesce-same-timestamp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "dispatch requests with the same trace timestamp through one native "
            "SGLang batch request"
        ),
    )
    parser.add_argument("--server-identity", type=Path)
    return parser.parse_args()


async def read_server_identity(
    session: aiohttp.ClientSession, base_url: str
) -> dict[str, Any] | None:
    try:
        async with session.get(f"{base_url}/get_server_info") as response:
            if response.status != 200:
                return None
            return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
        return None


async def flush_cache(
    session: aiohttp.ClientSession,
    base_url: str,
    *,
    timeout_seconds: float = 30.0,
    retry_interval_seconds: float = 0.05,
) -> dict[str, Any]:
    """Flush after SGLang finishes retiring the preceding streamed request.

    The HTTP stream may close just before the scheduler removes the request from
    its running set.  During that bounded tail SGLang rejects ``/flush_cache``
    with HTTP 400 instead of waiting.  Retrying preserves the required cache
    isolation; accepting the 400 or continuing without a successful flush
    would silently contaminate the next repetition.
    """
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    while True:
        attempts += 1
        async with session.post(f"{base_url}/flush_cache") as response:
            body = await response.text()
            status = response.status
        if status == 200:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {"response": body}
            if attempts > 1:
                if isinstance(payload, dict):
                    payload = {**payload, "_client_flush_attempts": attempts}
                else:
                    payload = {
                        "response": payload,
                        "_client_flush_attempts": attempts,
                    }
            return payload
        if status != 400 or time.monotonic() >= deadline:
            raise RuntimeError(
                f"flush_cache failed after {attempts} attempt(s) ({status}): {body}"
            )
        await asyncio.sleep(retry_interval_seconds)


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    request: TraceRequest,
    request_index: int,
    phase_start: float,
) -> RequestMeasurement:
    return (
        await send_request_group(
            session,
            base_url,
            [(request_index, request)],
            phase_start,
        )
    )[0]


async def send_request_group(
    session: aiohttp.ClientSession,
    base_url: str,
    indexed_requests: list[tuple[int, TraceRequest]],
    phase_start: float,
) -> list[RequestMeasurement]:
    if not indexed_requests:
        raise ValueError("request group must not be empty")
    timestamps = {request.timestamp_ms for _, request in indexed_requests}
    if len(timestamps) != 1:
        raise ValueError("request group must share one trace timestamp")

    target_start = phase_start + indexed_requests[0][1].timestamp_ms / 1000.0
    delay = target_start - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)

    start = time.perf_counter()
    progress = [
        {
            "first_token_time": None,
            "last_token_time": None,
            "itl_ms": [],
            "output_length": 0,
            "cached_tokens": 0,
            "cached_tokens_details": None,
        }
        for _ in indexed_requests
    ]
    batched = len(indexed_requests) > 1
    requests = [request for _, request in indexed_requests]
    payload = {
        "input_ids": (
            [request.token_ids for request in requests]
            if batched
            else requests[0].token_ids
        ),
        "sampling_params": [
            {
                "temperature": 0.0,
                "max_new_tokens": request.output_length,
                "ignore_eos": True,
            }
            for request in requests
        ]
        if batched
        else {
            "temperature": 0.0,
            "max_new_tokens": requests[0].output_length,
            "ignore_eos": True,
        },
        "stream": True,
    }

    error = ""
    try:
        async with session.post(f"{base_url}/generate", json=payload) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"generate failed ({response.status}): {body}")
            async for raw_line in response.content:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(b"data: "):
                    line = line[6:]
                if line == b"[DONE]":
                    continue
                event = json.loads(line)
                event_index = event.get("index", 0 if not batched else None)
                if not isinstance(event_index, int) or not (
                    0 <= event_index < len(progress)
                ):
                    raise ValueError("batched stream event has no valid request index")
                state = progress[event_index]
                meta = event.get("meta_info") or {}
                state["cached_tokens"] = int(
                    meta.get("cached_tokens") or state["cached_tokens"]
                )
                state["cached_tokens_details"] = (
                    meta.get("cached_tokens_details") or state["cached_tokens_details"]
                )
                completion_tokens = int(meta.get("completion_tokens") or 0)
                if completion_tokens <= state["output_length"]:
                    continue
                now = time.perf_counter()
                if state["first_token_time"] is None:
                    state["first_token_time"] = now
                elif state["last_token_time"] is not None:
                    new_tokens = completion_tokens - state["output_length"]
                    gap_ms = (now - state["last_token_time"]) * 1000.0
                    state["itl_ms"].extend([gap_ms / new_tokens] * new_tokens)
                state["last_token_time"] = now
                state["output_length"] = completion_tokens
    except (
        aiohttp.ClientError,
        asyncio.TimeoutError,
        json.JSONDecodeError,
        RuntimeError,
        ValueError,
    ) as exception:
        error = repr(exception)

    end = time.perf_counter()
    return [
        RequestMeasurement(
            request_index=request_index,
            success=not error,
            scheduled_timestamp_ms=request.timestamp_ms,
            actual_start_timestamp_ms=(start - phase_start) * 1000.0,
            input_length=request.input_length,
            requested_output_length=request.output_length,
            output_length=int(state["output_length"]),
            cached_tokens=int(state["cached_tokens"]),
            cached_tokens_details=state["cached_tokens_details"],
            latency_ms=(end - start) * 1000.0,
            ttft_ms=(
                (state["first_token_time"] - start) * 1000.0
                if state["first_token_time"] is not None
                else 0.0
            ),
            itl_ms=state["itl_ms"],
            metadata=request.metadata,
            error=error,
        )
        for (request_index, request), state in zip(
            indexed_requests, progress, strict=True
        )
    ]


def group_same_timestamp_requests(
    requests: list[TraceRequest],
) -> list[list[tuple[int, TraceRequest]]]:
    groups: dict[float, list[tuple[int, TraceRequest]]] = {}
    for index, request in enumerate(requests):
        groups.setdefault(request.timestamp_ms, []).append((index, request))
    return [groups[timestamp] for timestamp in sorted(groups)]


async def replay_phase(
    session: aiohttp.ClientSession,
    base_url: str,
    requests: list[TraceRequest],
    *,
    coalesce_same_timestamp: bool = False,
) -> list[RequestMeasurement]:
    phase_start = time.perf_counter()
    if coalesce_same_timestamp:
        grouped = await asyncio.gather(
            *(
                send_request_group(session, base_url, group, phase_start)
                for group in group_same_timestamp_requests(requests)
            )
        )
        return sorted(
            (measurement for group in grouped for measurement in group),
            key=lambda measurement: measurement.request_index,
        )
    tasks = [
        asyncio.create_task(
            send_request(session, base_url, request, index, phase_start)
        )
        for index, request in enumerate(requests)
    ]
    return await asyncio.gather(*tasks)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[rank]


def summarize(requests: list[RequestMeasurement]) -> dict[str, Any]:
    successful = [request for request in requests if request.success]
    latencies = [request.latency_ms for request in successful]
    ttfts = [request.ttft_ms for request in successful]
    itls = [value for request in successful for value in request.itl_ms]
    return {
        "request_count": len(requests),
        "success_count": len(successful),
        "failure_count": len(requests) - len(successful),
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "ttft_ms": {
            "mean": sum(ttfts) / len(ttfts) if ttfts else None,
            "p50": percentile(ttfts, 0.5),
            "p95": percentile(ttfts, 0.95),
            "max": max(ttfts) if ttfts else None,
        },
        "itl_ms": {
            "mean": sum(itls) / len(itls) if itls else None,
            "p50": percentile(itls, 0.5),
            "p95": percentile(itls, 0.95),
            "max": max(itls) if itls else None,
        },
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    base_url = args.base_url.rstrip("/")
    target_requests = load_trace_requests(args.trace.resolve())
    warmup_requests = (
        load_trace_requests(args.warmup_trace.resolve())
        if args.warmup_trace is not None
        else None
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    repetitions = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        server_reported_identity = await read_server_identity(session, base_url)
        for repetition in range(args.repetitions):
            flush_result = (
                await flush_cache(session, base_url) if args.flush_cache else None
            )
            warmup = (
                await replay_phase(
                    session,
                    base_url,
                    warmup_requests,
                    coalesce_same_timestamp=args.coalesce_same_timestamp,
                )
                if warmup_requests is not None
                else None
            )
            target = await replay_phase(
                session,
                base_url,
                target_requests,
                coalesce_same_timestamp=args.coalesce_same_timestamp,
            )
            repetitions.append(
                {
                    "repetition": repetition,
                    "flush_cache": flush_result,
                    "warmup": (
                        {
                            "summary": summarize(warmup),
                            "requests": [asdict(request) for request in warmup],
                        }
                        if warmup is not None
                        else None
                    ),
                    "target": {
                        "summary": summarize(target),
                        "requests": [asdict(request) for request in target],
                    },
                }
            )

    supplied_identity = None
    if args.server_identity is not None:
        supplied_identity = json.loads(args.server_identity.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "infercast_real_open_loop_v1",
        "coalesce_same_timestamp": args.coalesce_same_timestamp,
        "client": {"hostname": platform.node(), "platform": platform.platform()},
        "base_url": base_url,
        "trace": file_identity(args.trace.resolve()),
        "warmup_trace": (
            file_identity(args.warmup_trace.resolve())
            if args.warmup_trace is not None
            else None
        ),
        "repetitions": repetitions,
        "server_reported_identity": server_reported_identity,
        "server_supplied_identity": supplied_identity,
    }


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    failures = sum(
        repetition["target"]["summary"]["failure_count"]
        for repetition in result["repetitions"]
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
