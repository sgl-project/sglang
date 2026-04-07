"""Lightweight WebSocket microbenchmark for local gateway iteration.

This is intentionally separate from the existing genai-bench HTTP controls.
It measures the WebSocket Responses path directly on a small local model and
writes a JSON summary for repeatable local regression checks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# Allow CI and local CI-equivalent runs to swap this test model without editing code.
# Default remains the original 1B model for parity with the plan, but this can be
# overridden to any model id from MODEL_SPECS for environments without access.
_WS_BENCHMARK_MODEL = os.environ.get(
    "SGLANG_WS_BENCHMARK_MODEL", "llama-1b"
)

# WebSocket benchmark coverage is scoped to grpc worker backends in the current
# router configuration (responses WS is not supported over HTTP-backend worker links).
_WS_BENCH_BACKENDS = ["grpc"]


def _gateway_ws_url(base_url: str) -> str:
    if base_url.startswith("https://"):
        return f"wss://{base_url.removeprefix('https://')}/v1/responses"
    return f"ws://{base_url.removeprefix('http://')}/v1/responses"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _benchmark_request_body(model: str) -> dict:
    return {
        "model": model,
        "input": "Reply with the single word: hello",
        "temperature": 0,
        "max_output_tokens": 16,
        "store": False,
    }


def _ws_request(**request_fields) -> dict:
    return {"type": "response.create", **request_fields}


def _chain_turn_input(turn_index: int) -> str:
    return (
        f"Continuation turn {turn_index}. "
        "Reply with the single word: hello."
    )


def _tool_output_chain_turn_input(turn_index: int) -> list[dict]:
    return [
        {
            "type": "function_call_output",
            "call_id": f"call_chain_{turn_index}",
            "output": json.dumps(
                {
                    "step": turn_index,
                    "status": "ok",
                    "summary": f"tool result {turn_index}",
                    "artifacts": [f"chunk_{turn_index}_{index}" for index in range(3)],
                }
            ),
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        f"Tool step {turn_index} is complete. "
                        "Reply with the single word: hello."
                    ),
                }
            ],
        },
    ]


def _response_output_text_from_response(response: dict) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content_part in item.get("content", []):
            if not isinstance(content_part, dict):
                continue
            text = content_part.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _usage_token_totals(response: dict) -> tuple[int, int]:
    usage = response.get("usage", {})
    input_tokens = int(
        usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    )
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return input_tokens, output_tokens


def _http_stream_events(base_url: str, path: str, payload: dict, timeout_secs: float):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        response = urllib.request.urlopen(request, timeout=timeout_secs)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(
            f"HTTP SSE benchmark request failed status={exc.code} body={body}"
        ) from exc

    with response:
        event_lines: list[str] = []

        def parse_event(lines: list[str]) -> dict | None:
            data_lines = [
                line.removeprefix("data:").lstrip()
                for line in lines
                if line.startswith("data:")
            ]
            if not data_lines:
                return None
            payload_text = "\n".join(data_lines)
            if payload_text == "[DONE]":
                return None
            return json.loads(payload_text)

        while True:
            raw_line = response.readline()
            if not raw_line:
                if event_lines:
                    event = parse_event(event_lines)
                    if event is not None:
                        yield event
                break

            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if not event_lines:
                    continue
                event = parse_event(event_lines)
                event_lines = []
                if event is not None:
                    yield event
                continue

            if line.startswith(":"):
                continue

            event_lines.append(line)


def _collect_http_response_metrics(
    base_url: str, payload: dict, timeout_secs: float = 90
) -> tuple[dict, dict[str, float | int]]:
    request_payload = {**payload, "stream": True}
    request_payload_bytes = len(json.dumps(request_payload).encode("utf-8"))
    request_started_at = time.perf_counter()

    first_event_ms: float | None = None
    first_content_ms: float | None = None
    completed_ms: float | None = None
    response_payload_bytes = 0
    event_count = 0
    output_text_delta_count = 0

    for event in _http_stream_events(base_url, "/v1/responses", request_payload, timeout_secs):
        now = time.perf_counter()
        event_type = event.get("type")
        event_count += 1
        response_payload_bytes += len(json.dumps(event).encode("utf-8"))

        if first_event_ms is None:
            first_event_ms = (now - request_started_at) * 1000

        if (
            first_content_ms is None
            and event_type == "response.output_text.delta"
            and isinstance(event.get("delta"), str)
            and event["delta"]
        ):
            first_content_ms = (now - request_started_at) * 1000
            output_text_delta_count += 1
        elif (
            event_type == "response.output_text.delta"
            and isinstance(event.get("delta"), str)
            and event["delta"]
        ):
            output_text_delta_count += 1

        if event_type in {"error", "response.failed", "response.incomplete"}:
            raise AssertionError(
                f"HTTP transcript benchmark terminated with {event_type}: {event}"
            )

        if event_type == "response.completed":
            completed_ms = (now - request_started_at) * 1000
            response = event["response"]
            if first_content_ms is None:
                first_content_ms = completed_ms
            input_tokens, output_tokens = _usage_token_totals(response)
            return response, {
                "request_payload_bytes": request_payload_bytes,
                "response_payload_bytes": response_payload_bytes,
                "request_to_first_event_ms": first_event_ms or 0.0,
                "request_to_first_content_ms": first_content_ms or 0.0,
                "request_to_completed_ms": completed_ms or 0.0,
                "first_event_to_first_content_ms": max(
                    (first_content_ms or 0.0) - (first_event_ms or 0.0), 0.0
                ),
                "first_content_to_completed_ms": max(
                    (completed_ms or 0.0) - (first_content_ms or 0.0), 0.0
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "event_count": event_count,
                "output_text_delta_count": output_text_delta_count,
            }

    raise AssertionError("HTTP transcript benchmark ended without response.completed")


async def _collect_ws_response_metrics(
    websocket, request: dict, timeout_secs: float = 90
) -> tuple[dict, dict[str, float | int]]:
    payload = _ws_request(**request)
    request_payload_bytes = len(json.dumps(payload).encode("utf-8"))
    await websocket.send(json.dumps(payload))
    request_started_at = time.perf_counter()

    first_event_ms: float | None = None
    first_content_ms: float | None = None
    completed_ms: float | None = None
    response_payload_bytes = 0
    event_count = 0
    output_text_delta_count = 0

    while True:
        message = await asyncio.wait_for(websocket.recv(), timeout=timeout_secs)
        now = time.perf_counter()
        event = json.loads(message)
        event_type = event.get("type")
        event_count += 1
        response_payload_bytes += len(message.encode("utf-8"))

        if first_event_ms is None:
            first_event_ms = (now - request_started_at) * 1000

        if (
            first_content_ms is None
            and event_type == "response.output_text.delta"
            and isinstance(event.get("delta"), str)
            and event["delta"]
        ):
            first_content_ms = (now - request_started_at) * 1000
            output_text_delta_count += 1
        elif (
            event_type == "response.output_text.delta"
            and isinstance(event.get("delta"), str)
            and event["delta"]
        ):
            output_text_delta_count += 1

        if event_type == "error":
            raise AssertionError(f"Unexpected websocket transcript benchmark error: {event}")

        if event_type == "response.completed":
            completed_ms = (now - request_started_at) * 1000
            response = event["response"]
            if first_content_ms is None:
                first_content_ms = completed_ms
            input_tokens, output_tokens = _usage_token_totals(response)
            return response, {
                "request_payload_bytes": request_payload_bytes,
                "response_payload_bytes": response_payload_bytes,
                "request_to_first_event_ms": first_event_ms or 0.0,
                "request_to_first_content_ms": first_content_ms or 0.0,
                "request_to_completed_ms": completed_ms or 0.0,
                "first_event_to_first_content_ms": max(
                    (first_content_ms or 0.0) - (first_event_ms or 0.0), 0.0
                ),
                "first_content_to_completed_ms": max(
                    (completed_ms or 0.0) - (first_content_ms or 0.0), 0.0
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "event_count": event_count,
                "output_text_delta_count": output_text_delta_count,
            }


def _summarize_samples(samples: list[dict[str, float | int]]) -> dict[str, float | int]:
    def values(key: str) -> list[float]:
        return [float(sample[key]) for sample in samples]

    summary = {
        "samples": len(samples),
        "request_to_first_event_ms_p50": _percentile(
            values("request_to_first_event_ms"), 0.50
        ),
        "request_to_first_event_ms_p95": _percentile(
            values("request_to_first_event_ms"), 0.95
        ),
        "request_to_first_content_ms_p50": _percentile(
            values("request_to_first_content_ms"), 0.50
        ),
        "request_to_first_content_ms_p95": _percentile(
            values("request_to_first_content_ms"), 0.95
        ),
        "request_to_completed_ms_p50": _percentile(
            values("request_to_completed_ms"), 0.50
        ),
        "request_to_completed_ms_p95": _percentile(
            values("request_to_completed_ms"), 0.95
        ),
        "output_tokens_per_second_mean": statistics.fmean(
            values("output_tokens_per_second")
        ),
    }

    if "connect_ms" in samples[0]:
        summary["connect_ms_p50"] = _percentile(values("connect_ms"), 0.50)
        summary["connect_ms_p95"] = _percentile(values("connect_ms"), 0.95)

    return summary


def _summarize_chain_samples(
    samples: list[dict[str, float | int | list[float] | list[dict[str, float]]]],
) -> dict[str, float | int]:
    def values(key: str) -> list[float]:
        return [float(sample[key]) for sample in samples]

    summary: dict[str, float | int] = {"samples": len(samples)}

    for key in (
        "total_chain_ms",
        "first_turn_completed_ms",
        "continuation_only_total_ms",
    ):
        summary[f"{key}_mean"] = statistics.fmean(values(key))
        summary[f"{key}_p50"] = _percentile(values(key), 0.50)
        summary[f"{key}_p95"] = _percentile(values(key), 0.95)

    if "connect_ms" in samples[0]:
        summary["connect_ms_mean"] = statistics.fmean(values("connect_ms"))
        summary["connect_ms_p50"] = _percentile(values("connect_ms"), 0.50)
        summary["connect_ms_p95"] = _percentile(values("connect_ms"), 0.95)

    return summary


async def _run_single_ws_sample(ws_url: str, model: str) -> dict[str, float | int]:
    import websockets

    request = _ws_request(**_benchmark_request_body(model))

    connect_started_at = time.perf_counter()
    async with websockets.connect(ws_url, open_timeout=30, close_timeout=5) as websocket:
        connected_at = time.perf_counter()
        await websocket.send(json.dumps(request))
        request_sent_at = time.perf_counter()

        first_event_ms: float | None = None
        first_content_ms: float | None = None
        completed_ms: float | None = None
        output_tokens = 0

        while True:
            payload = await asyncio.wait_for(websocket.recv(), timeout=90)
            event = json.loads(payload)
            now = time.perf_counter()

            if first_event_ms is None:
                first_event_ms = (now - request_sent_at) * 1000

            if (
                first_content_ms is None
                and event.get("type") == "response.output_text.delta"
                and isinstance(event.get("delta"), str)
                and event["delta"]
            ):
                first_content_ms = (now - request_sent_at) * 1000

            if event.get("type") == "error":
                raise AssertionError(f"Unexpected websocket benchmark error: {event}")

            if event.get("type") == "response.completed":
                completed_ms = (now - request_sent_at) * 1000
                usage = event.get("response", {}).get("usage", {})
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                if first_content_ms is None:
                    first_content_ms = completed_ms
                break

    total_connect_ms = (connected_at - connect_started_at) * 1000
    tokens_per_second = 0.0
    if output_tokens > 0 and completed_ms and completed_ms > 0:
        tokens_per_second = output_tokens / (completed_ms / 1000)

    return {
        "connect_ms": total_connect_ms,
        "request_to_first_event_ms": first_event_ms or 0.0,
        "request_to_first_content_ms": first_content_ms or 0.0,
        "request_to_completed_ms": completed_ms or 0.0,
        "output_tokens": output_tokens,
        "output_tokens_per_second": tokens_per_second,
    }


def _run_single_http_sample(
    base_url: str, model: str, timeout_secs: float = 90
) -> dict[str, float | int]:
    request_started_at = time.perf_counter()

    first_event_ms: float | None = None
    first_content_ms: float | None = None
    completed_ms: float | None = None
    output_tokens = 0

    for event in _http_stream_events(
        base_url,
        "/v1/responses",
        {
            **_benchmark_request_body(model),
            "stream": True,
        },
        timeout_secs,
    ):
        now = time.perf_counter()
        event_type = event.get("type")

        if first_event_ms is None:
            first_event_ms = (now - request_started_at) * 1000

        if (
            first_content_ms is None
            and event_type == "response.output_text.delta"
            and isinstance(event.get("delta"), str)
            and event["delta"]
        ):
            first_content_ms = (now - request_started_at) * 1000

        if event_type in {"error", "response.failed", "response.incomplete"}:
            raise AssertionError(
                f"HTTP transport benchmark terminated with {event_type}: {event}"
            )

        if event_type == "response.completed":
            completed_ms = (now - request_started_at) * 1000
            usage = event.get("response", {}).get("usage", {})
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            if first_content_ms is None:
                first_content_ms = completed_ms
            break

    tokens_per_second = 0.0
    if output_tokens > 0 and completed_ms and completed_ms > 0:
        tokens_per_second = output_tokens / (completed_ms / 1000)

    return {
        "request_to_first_event_ms": first_event_ms or 0.0,
        "request_to_first_content_ms": first_content_ms or 0.0,
        "request_to_completed_ms": completed_ms or 0.0,
        "output_tokens": output_tokens,
        "output_tokens_per_second": tokens_per_second,
    }


async def _run_concurrency_profile(
    ws_url: str,
    model: str,
    concurrency_levels: list[int],
    samples_per_concurrency: int,
) -> dict:
    results = []
    for concurrency in concurrency_levels:
        samples: list[dict[str, float | int]] = []
        for _ in range(samples_per_concurrency):
            batch = await asyncio.gather(
                *[_run_single_ws_sample(ws_url, model) for _ in range(concurrency)]
            )
            samples.extend(batch)

        summary = _summarize_samples(samples)
        logger.info("WS microbench concurrency=%s summary=%s", concurrency, summary)
        results.append(
            {
                "concurrency": concurrency,
                "samples": samples,
                "summary": summary,
            }
        )

    return {
        "profile": {
            "concurrency_levels": concurrency_levels,
            "samples_per_concurrency": samples_per_concurrency,
        },
        "results": results,
    }


async def _run_ws_sample_batch(
    ws_url: str, model: str, samples: int
) -> list[dict[str, float | int]]:
    return await asyncio.gather(
        *[_run_single_ws_sample(ws_url, model) for _ in range(samples)]
    )


async def _collect_ws_terminal_event(websocket, request: dict) -> tuple[dict, float]:
    return await _collect_ws_terminal_event_with_timeout(websocket, request, timeout_secs=90)


async def _collect_ws_terminal_event_with_timeout(
    websocket, request: dict, *, timeout_secs: float
) -> tuple[dict, float]:
    await websocket.send(json.dumps(request))
    request_started_at = time.perf_counter()

    while True:
        payload = await asyncio.wait_for(websocket.recv(), timeout=timeout_secs)
        event = json.loads(payload)
        now = time.perf_counter()

        if event.get("type") == "error":
            raise AssertionError(f"Unexpected websocket chain benchmark error: {event}")

        if event.get("type") == "response.completed":
            return event, (now - request_started_at) * 1000


async def _run_ws_continuation_chain_sample(
    ws_url: str, model: str, turns: int
) -> dict[str, float | int | list[float]]:
    import websockets

    connect_started_at = time.perf_counter()
    async with websockets.connect(ws_url, open_timeout=30, close_timeout=5) as websocket:
        connected_at = time.perf_counter()

        per_turn_ms: list[float] = []
        response, first_turn_ms = await _collect_ws_terminal_event(
            websocket,
            _ws_request(
                model=model,
                input=_chain_turn_input(1),
                temperature=0,
                max_output_tokens=16,
                store=True,
            ),
        )
        per_turn_ms.append(first_turn_ms)
        previous_response_id = response["response"]["id"]

        for turn_index in range(2, turns + 1):
            response, completed_ms = await _collect_ws_terminal_event(
                websocket,
                _ws_request(
                    model=model,
                    input=_chain_turn_input(turn_index),
                    temperature=0,
                    max_output_tokens=16,
                    store=True,
                    previous_response_id=previous_response_id,
                ),
            )
            per_turn_ms.append(completed_ms)
            previous_response_id = response["response"]["id"]

    continuation_turns = per_turn_ms[1:]
    continuation_mean_ms = (
        statistics.fmean(continuation_turns) if continuation_turns else 0.0
    )
    total_chain_ms = sum(per_turn_ms)

    return {
        "connect_ms": (connected_at - connect_started_at) * 1000,
        "turns": turns,
        "total_chain_ms": total_chain_ms,
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
    }


async def _run_ws_tool_output_chain_sample(
    ws_url: str, model: str, tool_turns: int
) -> dict[str, float | int | list[float]]:
    import websockets

    connect_started_at = time.perf_counter()
    async with websockets.connect(ws_url, open_timeout=30, close_timeout=5) as websocket:
        connected_at = time.perf_counter()

        per_turn_ms: list[float] = []
        response, seed_turn_ms = await _collect_ws_terminal_event(
            websocket,
            _ws_request(
                model=model,
                input="Seed the tool-output continuation chain. Reply with hello.",
                temperature=0,
                max_output_tokens=16,
                store=True,
            ),
        )
        per_turn_ms.append(seed_turn_ms)
        previous_response_id = response["response"]["id"]

        for turn_index in range(1, tool_turns + 1):
            response, completed_ms = await _collect_ws_terminal_event(
                websocket,
                _ws_request(
                    model=model,
                    input=_tool_output_chain_turn_input(turn_index),
                    temperature=0,
                    max_output_tokens=16,
                    store=True,
                    previous_response_id=previous_response_id,
                ),
            )
            per_turn_ms.append(completed_ms)
            previous_response_id = response["response"]["id"]

    continuation_turns = per_turn_ms[1:]
    continuation_mean_ms = (
        statistics.fmean(continuation_turns) if continuation_turns else 0.0
    )
    total_chain_ms = sum(per_turn_ms)

    return {
        "connect_ms": (connected_at - connect_started_at) * 1000,
        "turns": tool_turns + 1,
        "tool_turns": tool_turns,
        "total_chain_ms": total_chain_ms,
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
    }


def _collect_http_completed_ms(
    base_url: str, payload: dict, timeout_secs: float = 90
) -> tuple[dict, float]:
    request_started_at = time.perf_counter()
    event_types: list[str] = []

    for event in _http_stream_events(base_url, "/v1/responses", payload, timeout_secs):
        event_type = str(event.get("type"))
        event_types.append(event_type)
        if event_type in {"error", "response.failed", "response.incomplete"}:
            raise AssertionError(
                f"HTTP continuation benchmark terminated with {event_type}: {event_types}"
            )
        if event_type == "response.completed":
            return event["response"], (time.perf_counter() - request_started_at) * 1000

    raise AssertionError(
        "HTTP continuation benchmark ended without response.completed; "
        f"events={event_types}"
    )


def _run_http_continuation_chain_sample(
    base_url: str, model: str, turns: int, timeout_secs: float = 90
) -> dict[str, float | int | list[float]]:
    per_turn_ms: list[float] = []

    response, completed_ms = _collect_http_completed_ms(
        base_url,
        {
            "model": model,
            "input": _chain_turn_input(1),
            "temperature": 0,
            "max_output_tokens": 16,
            "store": True,
            "stream": True,
        },
        timeout_secs,
    )
    per_turn_ms.append(completed_ms)
    previous_response_id = response["id"]

    for turn_index in range(2, turns + 1):
        response, completed_ms = _collect_http_completed_ms(
            base_url,
            {
                "model": model,
                "input": _chain_turn_input(turn_index),
                "previous_response_id": previous_response_id,
                "temperature": 0,
                "max_output_tokens": 16,
                "store": True,
                "stream": True,
            },
            timeout_secs,
        )
        per_turn_ms.append(completed_ms)
        previous_response_id = response["id"]

    continuation_turns = per_turn_ms[1:]
    continuation_mean_ms = (
        statistics.fmean(continuation_turns) if continuation_turns else 0.0
    )
    total_chain_ms = sum(per_turn_ms)

    return {
        "turns": turns,
        "total_chain_ms": total_chain_ms,
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
    }


def _run_http_tool_output_chain_sample(
    base_url: str, model: str, tool_turns: int, timeout_secs: float = 90
) -> dict[str, float | int | list[float]]:
    per_turn_ms: list[float] = []

    response, completed_ms = _collect_http_completed_ms(
        base_url,
        {
            "model": model,
            "input": "Seed the tool-output continuation chain. Reply with hello.",
            "temperature": 0,
            "max_output_tokens": 16,
            "store": True,
            "stream": True,
        },
        timeout_secs,
    )
    per_turn_ms.append(completed_ms)
    previous_response_id = response["id"]

    for turn_index in range(1, tool_turns + 1):
        response, completed_ms = _collect_http_completed_ms(
            base_url,
            {
                "model": model,
                "input": _tool_output_chain_turn_input(turn_index),
                "previous_response_id": previous_response_id,
                "temperature": 0,
                "max_output_tokens": 16,
                "store": True,
                "stream": True,
            },
            timeout_secs,
        )
        per_turn_ms.append(completed_ms)
        previous_response_id = response["id"]

    continuation_turns = per_turn_ms[1:]
    continuation_mean_ms = (
        statistics.fmean(continuation_turns) if continuation_turns else 0.0
    )
    total_chain_ms = sum(per_turn_ms)

    return {
        "turns": tool_turns + 1,
        "tool_turns": tool_turns,
        "total_chain_ms": total_chain_ms,
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
    }


def _maybe_write_summary(label: str, payload: dict) -> None:
    """Write JSON summary to disk only when SGLANG_WS_BENCH_OUTPUT_DIR is set."""
    output_dir = os.environ.get("SGLANG_WS_BENCH_OUTPUT_DIR")
    if not output_dir:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = label.lower().replace(" ", "_").replace("-", "_")
    out_path = out_dir / f"{slug}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Summary written to %s", out_path)


def _print_transport_table(
    label: str,
    model: str,
    http_summary: dict,
    ws_summary: dict,
    ratios: dict,
) -> None:
    """Print a formatted HTTP-vs-WS comparison table to stdout."""
    rows = [
        ("First Event (p50 ms)", "request_to_first_event_ms_p50"),
        ("First Content (p50 ms)", "request_to_first_content_ms_p50"),
        ("Completed (p50 ms)", "request_to_completed_ms_p50"),
        ("Output tok/s (mean)", "output_tokens_per_second_mean"),
    ]
    hdr = f"\n{'=' * 70}\n  {label}  |  model: {model}\n{'=' * 70}"
    print(hdr)
    print(f"  {'Metric':<28} {'HTTP SSE':>12} {'WebSocket':>12} {'WS/HTTP':>10}")
    print(f"  {'-' * 62}")
    for name, key in rows:
        h = http_summary.get(key, 0)
        w = ws_summary.get(key, 0)
        r = w / h if h > 0 else 0.0
        print(f"  {name:<28} {h:>12.2f} {w:>12.2f} {r:>10.3f}")
    print(f"  {'-' * 62}")
    print(f"  Samples: {int(http_summary.get('samples', 0))}")
    print(f"{'=' * 70}\n")


def _print_chain_table(
    label: str,
    model: str,
    http_summary: dict,
    ws_summary: dict,
    ratios: dict,
) -> None:
    """Print a formatted chain comparison table to stdout."""
    rows = [
        ("Total Chain (p50 ms)", "total_chain_ms_p50"),
        ("First Turn (p50 ms)", "first_turn_completed_ms_p50"),
        ("Continuation Only (p50 ms)", "continuation_only_total_ms_p50"),
    ]
    hdr = f"\n{'=' * 70}\n  {label}  |  model: {model}\n{'=' * 70}"
    print(hdr)
    print(f"  {'Metric':<28} {'HTTP SSE':>12} {'WebSocket':>12} {'WS/HTTP':>10}")
    print(f"  {'-' * 62}")
    for name, key in rows:
        h = http_summary.get(key, 0)
        w = ws_summary.get(key, 0)
        r = w / h if h > 0 else 0.0
        print(f"  {name:<28} {h:>12.2f} {w:>12.2f} {r:>10.3f}")
    print(f"  {'-' * 62}")
    delta_key = "ws_total_chain_delta_pct"
    if delta_key in ratios:
        print(f"  WS total chain delta: {ratios[delta_key]:+.2f}%")
    print(f"  Samples: {int(http_summary.get('samples', 0))}")
    print(f"{'=' * 70}\n")


def _worker_transport_for_backend(backend_name: str) -> str:
    if backend_name == "http":
        return "http"
    if backend_name == "grpc":
        return "grpc"
    if backend_name == "pd":
        return "http"
    raise ValueError(f"Unsupported benchmark backend: {backend_name}")


def _router_topology_for_backend(backend_name: str) -> str:
    if backend_name == "http":
        return "regular_http_worker"
    if backend_name == "grpc":
        return "regular_grpc_worker"
    if backend_name == "pd":
        return "pd_http_workers"
    raise ValueError(f"Unsupported benchmark backend: {backend_name}")


def _topology_overlay_for_backend(backend_name: str) -> str:
    if backend_name == "pd":
        return "pd"
    return "none"


def _benchmark_context(
    *,
    benchmark_family: str,
    run_class: str,
    backend_name: str,
    model: str,
    store_mode: str,
    workload_kind: str,
) -> dict[str, str]:
    return {
        "benchmark_family": benchmark_family,
        "run_class": run_class,
        "worker_transport": _worker_transport_for_backend(backend_name),
        "router_topology": _router_topology_for_backend(backend_name),
        "model_id": model,
        "topology_overlay": _topology_overlay_for_backend(backend_name),
        "store_mode": store_mode,
        "workload_kind": workload_kind,
    }


def _benchmark_contract(*, client_transport: str, **context: str) -> dict[str, str]:
    return {
        **context,
        "client_transport": client_transport,
    }


def _transport_result(
    *,
    context: dict[str, str],
    client_transport: str,
    samples: list[dict],
    summary: dict,
) -> dict:
    return {
        "transport": client_transport,
        "benchmark_contract": _benchmark_contract(
            client_transport=client_transport,
            **context,
        ),
        "samples": samples,
        "summary": summary,
    }


def _transport_ratios(http_summary: dict, ws_summary: dict) -> dict[str, float]:
    def ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    return {
        "ws_over_http_first_event_p50": ratio(
            float(ws_summary["request_to_first_event_ms_p50"]),
            float(http_summary["request_to_first_event_ms_p50"]),
        ),
        "ws_over_http_first_content_p50": ratio(
            float(ws_summary["request_to_first_content_ms_p50"]),
            float(http_summary["request_to_first_content_ms_p50"]),
        ),
        "ws_over_http_completed_p50": ratio(
            float(ws_summary["request_to_completed_ms_p50"]),
            float(http_summary["request_to_completed_ms_p50"]),
        ),
        "ws_over_http_output_tps_mean": ratio(
            float(ws_summary["output_tokens_per_second_mean"]),
            float(http_summary["output_tokens_per_second_mean"]),
        ),
    }


def _chain_transport_ratios(http_summary: dict, ws_summary: dict) -> dict[str, float]:
    def ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    total_ratio = ratio(
        float(ws_summary["total_chain_ms_p50"]), float(http_summary["total_chain_ms_p50"])
    )
    continuation_ratio = ratio(
        float(ws_summary["continuation_only_total_ms_p50"]),
        float(http_summary["continuation_only_total_ms_p50"]),
    )

    return {
        "ws_over_http_total_chain": total_ratio,
        "ws_over_http_continuation_only": continuation_ratio,
        "ws_vs_http_total_chain_delta_pct": (1.0 - total_ratio) * 100.0,
        "ws_vs_http_continuation_only_delta_pct": (1.0 - continuation_ratio)
            * 100.0,
    }


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Benchmark timing is only meaningful sequentially.")
@pytest.mark.model(_WS_BENCHMARK_MODEL)
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", _WS_BENCH_BACKENDS, indirect=True)
class TestWsMicrobench:
    """WebSocket benchmark for the Responses route on a single small model."""

    def test_ws_microbench(self, setup_backend):
        backend_name, model, _, gateway = setup_backend

        concurrency_levels = [
            int(value)
            for value in os.environ.get("SGLANG_WS_BENCH_CONCURRENCY", "1,2,4").split(",")
            if value.strip()
        ]
        samples_per_concurrency = int(
            os.environ.get("SGLANG_WS_BENCH_SAMPLES_PER_CONCURRENCY", "2")
        )
        benchmark_context = _benchmark_context(
            benchmark_family="transport_qos",
            run_class="ws_microbench_profile",
            backend_name=backend_name,
            model=model,
            store_mode="store_false",
            workload_kind="single_turn_text",
        )

        payload = asyncio.run(
            _run_concurrency_profile(
                _gateway_ws_url(gateway.base_url),
                model,
                concurrency_levels,
                samples_per_concurrency,
            )
        )
        payload["benchmark_contract"] = _benchmark_contract(
            client_transport="websocket",
            **benchmark_context,
        )
        payload["worker_backend"] = backend_name
        payload["router_url"] = gateway.base_url
        payload["model"] = model
        _maybe_write_summary("ws_microbench", payload)
        print(f"\n{'=' * 50}")
        print(f"  WS Microbenchmark  |  model: {model}")
        print(f"{'=' * 50}")
        for profile in payload.get("profiles", []):
            c = profile.get("concurrency", "?")
            s = profile.get("summary", {})
            print(
                f"  concurrency={c}  "
                f"first_event_p50={s.get('request_to_first_event_ms_p50', 0):.1f}ms  "
                f"completed_p50={s.get('request_to_completed_ms_p50', 0):.1f}ms"
            )
        print(f"{'=' * 50}\n")

        for result in payload["results"]:
            summary = result["summary"]
            assert summary["samples"] > 0
            assert summary["request_to_first_event_ms_p50"] > 0
            assert summary["request_to_first_content_ms_p50"] > 0
            assert summary["request_to_completed_ms_p50"] > 0


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Benchmark timing is only meaningful sequentially.")
@pytest.mark.model(_WS_BENCHMARK_MODEL)
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", _WS_BENCH_BACKENDS, indirect=True)
class TestResponsesTransportCompare:
    """Small-model transport comparison for HTTP SSE vs WebSocket Responses."""

    def test_http_vs_ws_transport_compare(self, setup_backend):
        backend_name, model, client, gateway = setup_backend

        samples = int(os.environ.get("SGLANG_HTTP_WS_COMPARE_SAMPLES", "2"))
        benchmark_context = _benchmark_context(
            benchmark_family="transport_qos",
            run_class="http_vs_ws_transport_compare",
            backend_name=backend_name,
            model=model,
            store_mode="store_false",
            workload_kind="single_turn_text",
        )

        timeout_secs = float(
            os.environ.get("SGLANG_HTTP_WS_COMPARE_TIMEOUT_SECS", "90")
        )
        http_samples = [
            _run_single_http_sample(gateway.base_url, model, timeout_secs)
            for _ in range(samples)
        ]
        ws_samples = asyncio.run(
            _run_ws_sample_batch(_gateway_ws_url(gateway.base_url), model, samples)
        )

        http_summary = _summarize_samples(http_samples)
        ws_summary = _summarize_samples(ws_samples)

        payload = {
            "benchmark_context": benchmark_context,
            "worker_backend": backend_name,
            "router_url": gateway.base_url,
            "model": model,
            "samples_per_transport": samples,
            "http": _transport_result(
                context=benchmark_context,
                client_transport="http_sse",
                samples=http_samples,
                summary=http_summary,
            ),
            "websocket": _transport_result(
                context=benchmark_context,
                client_transport="websocket",
                samples=ws_samples,
                summary=ws_summary,
            ),
            "ratios": _transport_ratios(http_summary, ws_summary),
        }

        _maybe_write_summary("http_vs_ws_transport_compare", payload)
        _print_transport_table(
            "HTTP vs WS Transport Compare",
            model,
            http_summary,
            ws_summary,
            payload["ratios"],
        )

        assert http_summary["samples"] > 0
        assert ws_summary["samples"] > 0
        assert http_summary["request_to_first_event_ms_p50"] > 0
        assert ws_summary["request_to_first_event_ms_p50"] > 0
        assert http_summary["request_to_completed_ms_p50"] > 0
        assert ws_summary["request_to_completed_ms_p50"] > 0


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Benchmark timing is only meaningful sequentially.")
@pytest.mark.model(_WS_BENCHMARK_MODEL)
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", _WS_BENCH_BACKENDS, indirect=True)
class TestResponsesContinuationChainCompare:
    """Long-chain continuation comparison for HTTP vs persistent WS."""

    def test_http_vs_ws_continuation_chain_compare(self, setup_backend):
        backend_name, model, client, gateway = setup_backend

        turns = int(os.environ.get("SGLANG_HTTP_WS_CHAIN_TURNS", "20"))
        samples = int(os.environ.get("SGLANG_HTTP_WS_CHAIN_SAMPLES", "1"))
        benchmark_context = _benchmark_context(
            benchmark_family="continuation_qos",
            run_class="http_vs_ws_continuation_compare",
            backend_name=backend_name,
            model=model,
            store_mode="store_true",
            workload_kind="incremental_text_continuation",
        )

        timeout_secs = float(
            os.environ.get("SGLANG_HTTP_WS_CHAIN_TIMEOUT_SECS", "90")
        )
        http_samples = [
            _run_http_continuation_chain_sample(
                gateway.base_url, model, turns, timeout_secs
            )
            for _ in range(samples)
        ]
        ws_samples = [
            asyncio.run(
                _run_ws_continuation_chain_sample(
                    _gateway_ws_url(gateway.base_url), model, turns
                )
            )
            for _ in range(samples)
        ]
        http_summary = _summarize_chain_samples(http_samples)
        ws_summary = _summarize_chain_samples(ws_samples)

        payload = {
            "benchmark_context": benchmark_context,
            "worker_backend": backend_name,
            "router_url": gateway.base_url,
            "model": model,
            "turns": turns,
            "samples": samples,
            "http": _transport_result(
                context=benchmark_context,
                client_transport="http_sse",
                samples=http_samples,
                summary=http_summary,
            ),
            "websocket": _transport_result(
                context=benchmark_context,
                client_transport="websocket",
                samples=ws_samples,
                summary=ws_summary,
            ),
            "ratios": _chain_transport_ratios(http_summary, ws_summary),
        }

        _maybe_write_summary("continuation_chain_compare", payload)
        _print_chain_table(
            "Continuation Chain Compare",
            model,
            http_summary,
            ws_summary,
            payload["ratios"],
        )

        assert http_samples[0]["turns"] == turns
        assert ws_samples[0]["turns"] == turns
        assert float(http_samples[0]["total_chain_ms"]) > 0
        assert float(ws_samples[0]["total_chain_ms"]) > 0


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Benchmark timing is only meaningful sequentially.")
@pytest.mark.model(_WS_BENCHMARK_MODEL)
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", _WS_BENCH_BACKENDS, indirect=True)
class TestResponsesToolOutputChainCompare:
    """Tool-output-heavy continuation comparison for HTTP vs persistent WS."""

    def test_http_vs_ws_tool_output_chain_compare(self, setup_backend):
        backend_name, model, client, gateway = setup_backend

        tool_turns = int(os.environ.get("SGLANG_HTTP_WS_TOOL_CHAIN_TURNS", "20"))
        samples = int(os.environ.get("SGLANG_HTTP_WS_TOOL_CHAIN_SAMPLES", "1"))
        benchmark_context = _benchmark_context(
            benchmark_family="continuation_qos",
            run_class="http_vs_ws_tool_output_compare",
            backend_name=backend_name,
            model=model,
            store_mode="store_true",
            workload_kind="incremental_tool_output_continuation",
        )

        timeout_secs = float(
            os.environ.get("SGLANG_HTTP_WS_TOOL_CHAIN_TIMEOUT_SECS", "90")
        )
        http_samples = [
            _run_http_tool_output_chain_sample(
                gateway.base_url, model, tool_turns, timeout_secs
            )
            for _ in range(samples)
        ]
        ws_samples = [
            asyncio.run(
                _run_ws_tool_output_chain_sample(
                    _gateway_ws_url(gateway.base_url), model, tool_turns
                )
            )
            for _ in range(samples)
        ]
        http_summary = _summarize_chain_samples(http_samples)
        ws_summary = _summarize_chain_samples(ws_samples)

        payload = {
            "benchmark_context": benchmark_context,
            "worker_backend": backend_name,
            "router_url": gateway.base_url,
            "model": model,
            "tool_turns": tool_turns,
            "samples": samples,
            "http": _transport_result(
                context=benchmark_context,
                client_transport="http_sse",
                samples=http_samples,
                summary=http_summary,
            ),
            "websocket": _transport_result(
                context=benchmark_context,
                client_transport="websocket",
                samples=ws_samples,
                summary=ws_summary,
            ),
            "ratios": _chain_transport_ratios(http_summary, ws_summary),
        }

        _maybe_write_summary("tool_output_chain_compare", payload)
        _print_chain_table(
            "Tool-Output Chain Compare",
            model,
            http_summary,
            ws_summary,
            payload["ratios"],
        )

        assert http_samples[0]["tool_turns"] == tool_turns
        assert ws_samples[0]["tool_turns"] == tool_turns
        assert float(http_samples[0]["total_chain_ms"]) > 0
        assert float(ws_samples[0]["total_chain_ms"]) > 0
