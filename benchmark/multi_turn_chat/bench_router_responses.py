"""Multi-turn Responses API benchmark for sgl-model-gateway.

Measures per-turn latency over HTTP SSE and WebSocket transports against a
running gateway, reusing the existing multi-turn workload generator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from . import data_gen
except ImportError:
    import data_gen


DEFAULT_API_KEY = "not-used"


@dataclass(frozen=True)
class ChainRuntimeConfig:
    base_url: str
    model: str
    api_key: str
    client_transport: str
    chain_mode: str
    store: bool
    request_timeout_secs: float
    capture_event_trace: bool
    event_trace_limit: int


@dataclass
class ResponseTimingTracker:
    request_started_at: float
    capture_event_trace: bool
    event_trace_limit: int
    first_event_ms: float | None = None
    first_content_ms: float | None = None
    completed_ms: float | None = None
    event_count: int = 0
    output_text_delta_count: int = 0
    event_trace: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, event: dict[str, Any], observed_at: float) -> None:
        elapsed_ms = (observed_at - self.request_started_at) * 1000
        event_type = _event_type(event)
        if self.first_event_ms is None:
            self.first_event_ms = elapsed_ms

        delta_chars = 0
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                self.output_text_delta_count += 1
                delta_chars = len(delta)
                if self.first_content_ms is None:
                    self.first_content_ms = elapsed_ms

        if event_type == "response.completed":
            self.completed_ms = elapsed_ms

        self.event_count += 1
        if self.capture_event_trace and len(self.event_trace) < self.event_trace_limit:
            self.event_trace.append(
                {
                    "event_type": event_type,
                    "t_ms": elapsed_ms,
                    "delta_chars": delta_chars,
                }
            )

    def metrics(self) -> dict[str, Any]:
        completed_ms = self.completed_ms or 0.0
        first_event_ms = self.first_event_ms or 0.0
        first_content_ms = self.first_content_ms or completed_ms
        return {
            "request_to_first_event_ms": first_event_ms,
            "request_to_first_content_ms": first_content_ms,
            "request_to_completed_ms": completed_ms,
            "first_event_to_first_content_ms": max(first_content_ms - first_event_ms, 0.0),
            "first_content_to_completed_ms": max(completed_ms - first_content_ms, 0.0),
            "event_count": self.event_count,
            "output_text_delta_count": self.output_text_delta_count,
            "event_trace": self.event_trace if self.capture_event_trace else [],
        }


def _normalize_base_url(base_url: str) -> str:
    if "://" not in base_url:
        return f"http://{base_url.rstrip('/')}"
    return base_url.rstrip("/")


def _gateway_ws_url(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    if normalized.startswith("https://"):
        return f"wss://{normalized.removeprefix('https://')}/v1/responses"
    return f"ws://{normalized.removeprefix('http://')}/v1/responses"


def _store_mode_label(store: bool) -> str:
    return "store_true" if store else "store_false"


def _topology_overlay(router_topology: str) -> str:
    if router_topology == "pd_http_workers":
        return "pd"
    return "none"


def _benchmark_contract(args: argparse.Namespace, client_transport: str) -> dict[str, str]:
    return {
        "benchmark_family": "long_context_multiturn_qos",
        "run_class": "router_multiturn_adapter",
        "client_transport": client_transport,
        "worker_transport": args.worker_transport,
        "router_topology": args.router_topology,
        "model_id": args.model,
        "topology_overlay": _topology_overlay(args.router_topology),
        "store_mode": _store_mode_label(args.store_mode == "true"),
        "workload_kind": f"multi_turn_chat_{args.chain_mode}",
    }


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


def _user_message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def _assistant_message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _prepare_turn_input(
    chain_mode: str,
    conversation: list[dict[str, Any]],
    prompt: str,
) -> list[dict[str, Any]] | str:
    if chain_mode == "previous_response_id":
        return prompt
    return [*conversation, _user_message(prompt)]


def _response_output_text_from_http_response(response: Any) -> str:
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


def _response_output_text_from_ws_response(response: dict[str, Any]) -> str:
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


def _event_type(event: dict[str, Any]) -> str:
    event_type = event.get("type")
    if isinstance(event_type, str) and event_type:
        return event_type
    sse_event_name = event.get("_sse_event_name")
    if isinstance(sse_event_name, str) and sse_event_name:
        return sse_event_name
    return "unknown"


def _parse_sse_event(lines: list[str]) -> dict[str, Any] | None:
    data_lines = [
        line.removeprefix("data:").lstrip() for line in lines if line.startswith("data:")
    ]
    if not data_lines:
        return None
    payload_text = "\n".join(data_lines)
    if payload_text == "[DONE]":
        return None

    event = json.loads(payload_text)
    event_name = next(
        (
            line.removeprefix("event:").lstrip()
            for line in lines
            if line.startswith("event:")
        ),
        None,
    )
    if event_name and isinstance(event, dict) and "_sse_event_name" not in event:
        event["_sse_event_name"] = event_name
    return event


def _http_stream_events(
    base_url: str,
    request: dict[str, Any],
    timeout_secs: float,
):
    payload = {**request, "stream": True}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{_normalize_base_url(base_url)}/v1/responses",
        data=data,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        response = urllib.request.urlopen(req, timeout=timeout_secs)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP response chain request failed status={exc.code} body={body}"
        ) from exc

    with response:
        event_lines: list[str] = []

        while True:
            raw_line = response.readline()
            if not raw_line:
                if event_lines:
                    event = _parse_sse_event(event_lines)
                    if event is not None:
                        yield event
                break

            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if not event_lines:
                    continue
                event = _parse_sse_event(event_lines)
                event_lines = []
                if event is not None:
                    yield event
                continue

            if line.startswith(":"):
                continue

            event_lines.append(line)


def _collect_http_response(
    config: ChainRuntimeConfig,
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_started_at = time.perf_counter()
    tracker = ResponseTimingTracker(
        request_started_at=request_started_at,
        capture_event_trace=config.capture_event_trace,
        event_trace_limit=config.event_trace_limit,
    )

    for event in _http_stream_events(
        config.base_url,
        request,
        config.request_timeout_secs,
    ):
        now = time.perf_counter()
        tracker.observe(event, now)
        event_type = _event_type(event)

        if event_type in {"error", "response.failed", "response.incomplete"}:
            raise RuntimeError(f"HTTP response chain failed with event={event_type}")

        if event_type == "response.completed":
            return event["response"], tracker.metrics()

    raise RuntimeError("HTTP response stream ended without response.completed")


async def _collect_ws_response(
    websocket: Any,
    config: ChainRuntimeConfig,
    request: dict[str, Any],
    timeout_secs: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_started_at = time.perf_counter()
    await websocket.send(json.dumps({"type": "response.create", **request}))
    tracker = ResponseTimingTracker(
        request_started_at=request_started_at,
        capture_event_trace=config.capture_event_trace,
        event_trace_limit=config.event_trace_limit,
    )

    while True:
        payload = await asyncio.wait_for(websocket.recv(), timeout=timeout_secs)
        event = json.loads(payload)
        now = time.perf_counter()
        tracker.observe(event, now)

        if event.get("type") == "error":
            raise RuntimeError(f"WebSocket response chain failed with event={event}")

        if event.get("type") == "response.completed":
            return event["response"], tracker.metrics()


def _append_full_replay_turn(
    conversation: list[dict[str, Any]],
    prompt: str,
    assistant_text: str,
) -> None:
    conversation.append(_user_message(prompt))
    conversation.append(_assistant_message(assistant_text))


def _run_http_chain(
    config: ChainRuntimeConfig,
    qas: list[dict[str, int | str]],
) -> dict[str, Any]:
    conversation: list[dict[str, Any]] = []
    previous_response_id: str | None = None
    per_turn: list[dict[str, Any]] = []
    total_request_payload_bytes = 0
    chain_started_at = time.perf_counter()

    for turn_index, qa in enumerate(qas, start=1):
        turn_input = _prepare_turn_input(
            config.chain_mode,
            conversation,
            str(qa["prompt"]),
        )
        request = {
            "model": config.model,
            "input": turn_input,
            "temperature": 0,
            "max_output_tokens": int(qa["new_tokens"]),
            "store": config.store,
        }
        if config.chain_mode == "previous_response_id" and previous_response_id is not None:
            request["previous_response_id"] = previous_response_id

        request_payload_bytes = len(json.dumps({**request, "stream": True}).encode("utf-8"))
        response, metrics = _collect_http_response(config, request)
        assistant_text = _response_output_text_from_http_response(response)
        if config.chain_mode == "full_replay":
            _append_full_replay_turn(conversation, str(qa["prompt"]), assistant_text)
        previous_response_id = response.get("id")
        total_request_payload_bytes += request_payload_bytes

        per_turn.append(
            {
                "turn_index": turn_index,
                "prompt_len": len(str(qa["prompt"])),
                "max_output_tokens": int(qa["new_tokens"]),
                "response_id": previous_response_id,
                "request_payload_bytes": request_payload_bytes,
                "output_text": assistant_text,
                **metrics,
            }
        )

    return {
        "turn_count": len(qas),
        "total_chain_ms": (time.perf_counter() - chain_started_at) * 1000,
        "connect_ms": 0.0,
        "total_request_payload_bytes": total_request_payload_bytes,
        "per_turn": per_turn,
    }


async def _run_ws_chain_async(
    config: ChainRuntimeConfig,
    qas: list[dict[str, int | str]],
) -> dict[str, Any]:
    import websockets

    conversation: list[dict[str, Any]] = []
    previous_response_id: str | None = None
    per_turn: list[dict[str, Any]] = []
    total_request_payload_bytes = 0
    chain_started_at = time.perf_counter()
    connect_started_at = time.perf_counter()

    async with websockets.connect(
        _gateway_ws_url(config.base_url),
        open_timeout=30,
        close_timeout=5,
    ) as websocket:
        connect_ms = (time.perf_counter() - connect_started_at) * 1000
        for turn_index, qa in enumerate(qas, start=1):
            turn_input = _prepare_turn_input(
                config.chain_mode,
                conversation,
                str(qa["prompt"]),
            )
            request = {
                "model": config.model,
                "input": turn_input,
                "temperature": 0,
                "max_output_tokens": int(qa["new_tokens"]),
                "store": config.store,
            }
            if (
                config.chain_mode == "previous_response_id"
                and previous_response_id is not None
            ):
                request["previous_response_id"] = previous_response_id

            request_payload_bytes = len(
                json.dumps({"type": "response.create", **request}).encode("utf-8")
            )
            response, metrics = await _collect_ws_response(
                websocket,
                config,
                request,
                config.request_timeout_secs,
            )
            assistant_text = _response_output_text_from_ws_response(response)
            if config.chain_mode == "full_replay":
                _append_full_replay_turn(conversation, str(qa["prompt"]), assistant_text)
            previous_response_id = response.get("id")
            total_request_payload_bytes += request_payload_bytes

            per_turn.append(
                {
                    "turn_index": turn_index,
                    "prompt_len": len(str(qa["prompt"])),
                    "max_output_tokens": int(qa["new_tokens"]),
                    "response_id": previous_response_id,
                    "request_payload_bytes": request_payload_bytes,
                    "output_text": assistant_text,
                    **metrics,
                }
            )

    return {
        "turn_count": len(qas),
        "total_chain_ms": (time.perf_counter() - chain_started_at) * 1000,
        "connect_ms": connect_ms,
        "total_request_payload_bytes": total_request_payload_bytes,
        "per_turn": per_turn,
    }


def _run_ws_chain(
    config: ChainRuntimeConfig,
    qas: list[dict[str, int | str]],
) -> dict[str, Any]:
    return asyncio.run(_run_ws_chain_async(config, qas))


def _summarize_transport(
    *,
    args: argparse.Namespace,
    client_transport: str,
    chain_results: list[dict[str, Any]],
    wall_clock_ms: float,
) -> dict[str, Any]:
    chain_totals = [float(result["total_chain_ms"]) for result in chain_results]
    turn_totals = [
        float(turn["request_to_completed_ms"])
        for result in chain_results
        for turn in result["per_turn"]
    ]
    first_events = [
        float(turn["request_to_first_event_ms"])
        for result in chain_results
        for turn in result["per_turn"]
    ]
    first_content = [
        float(turn["request_to_first_content_ms"])
        for result in chain_results
        for turn in result["per_turn"]
    ]
    first_event_to_first_content = [
        float(turn["first_event_to_first_content_ms"])
        for result in chain_results
        for turn in result["per_turn"]
    ]
    first_content_to_completed = [
        float(turn["first_content_to_completed_ms"])
        for result in chain_results
        for turn in result["per_turn"]
    ]
    event_counts = [
        float(turn["event_count"]) for result in chain_results for turn in result["per_turn"]
    ]
    output_text_delta_counts = [
        float(turn["output_text_delta_count"])
        for result in chain_results
        for turn in result["per_turn"]
    ]
    connect_ms = [float(result.get("connect_ms", 0.0)) for result in chain_results]
    request_payload_totals = [
        float(result["total_request_payload_bytes"]) for result in chain_results
    ]

    return {
        "benchmark_contract": _benchmark_contract(args, client_transport),
        "chain_results": chain_results,
        "summary": {
            "chains": len(chain_results),
            "turns": sum(int(result["turn_count"]) for result in chain_results),
            "wall_clock_ms": wall_clock_ms,
            "chain_ms_mean": statistics.fmean(chain_totals) if chain_totals else 0.0,
            "chain_ms_p50": _percentile(chain_totals, 0.50),
            "chain_ms_p95": _percentile(chain_totals, 0.95),
            "turn_ms_mean": statistics.fmean(turn_totals) if turn_totals else 0.0,
            "turn_ms_p50": _percentile(turn_totals, 0.50),
            "turn_ms_p95": _percentile(turn_totals, 0.95),
            "first_event_ms_p50": _percentile(first_events, 0.50),
            "first_content_ms_p50": _percentile(first_content, 0.50),
            "first_event_to_first_content_ms_p50": _percentile(
                first_event_to_first_content, 0.50
            ),
            "first_content_to_completed_ms_p50": _percentile(
                first_content_to_completed, 0.50
            ),
            "event_count_mean": statistics.fmean(event_counts) if event_counts else 0.0,
            "output_text_delta_count_mean": (
                statistics.fmean(output_text_delta_counts)
                if output_text_delta_counts
                else 0.0
            ),
            "connect_ms_mean": statistics.fmean(connect_ms) if connect_ms else 0.0,
            "request_payload_bytes_total_mean": (
                statistics.fmean(request_payload_totals)
                if request_payload_totals
                else 0.0
            ),
            "request_payload_bytes_total_p50": _percentile(
                request_payload_totals, 0.50
            ),
            "request_payload_bytes_total_p95": _percentile(
                request_payload_totals, 0.95
            ),
        },
    }


def _run_transport(
    *,
    args: argparse.Namespace,
    client_transport: str,
    workloads: list[dict[str, Any]],
) -> dict[str, Any]:
    config = ChainRuntimeConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        client_transport=client_transport,
        chain_mode=args.chain_mode,
        store=args.store_mode == "true",
        request_timeout_secs=args.request_timeout_secs,
        capture_event_trace=args.capture_event_trace,
        event_trace_limit=args.event_trace_limit,
    )

    chain_runner = _run_http_chain if client_transport == "http_sse" else _run_ws_chain
    started_at = time.perf_counter()
    chain_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(chain_runner, config, workload["qas"])
            for workload in workloads
        ]
        for future in as_completed(futures):
            chain_results.append(future.result())

    wall_clock_ms = (time.perf_counter() - started_at) * 1000
    return _summarize_transport(
        args=args,
        client_transport=client_transport,
        chain_results=chain_results,
        wall_clock_ms=wall_clock_ms,
    )


def _transport_targets(client_transport: str) -> list[str]:
    if client_transport == "both":
        return ["http_sse", "websocket"]
    return [client_transport]


def _build_workloads(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_gen.random.seed(args.seed)
    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            args.tokenizer, trust_remote_code=args.trust_remote_code
        )
    except ModuleNotFoundError:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    return data_gen.gen_arguments(args, tokenizer)


def _write_summary(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def _append_result_line(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fout:
        fout.write(json.dumps(payload) + "\n")


def _validate_args(args: argparse.Namespace) -> None:
    if args.parallel < 1:
        raise ValueError("--parallel must be >= 1")
    if getattr(args, "event_trace_limit", 1) < 1:
        raise ValueError("--event-trace-limit must be >= 1")
    if args.client_transport in {"http_sse", "both"}:
        if args.chain_mode == "previous_response_id" and args.store_mode == "false":
            raise ValueError(
                "HTTP SSE with --chain-mode previous_response_id requires --store-mode true"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the router Responses path using the existing multi-turn-chat "
            "workload generator."
        )
    )
    parser.add_argument("--base-url", type=str, required=True, help="Router base URL.")
    parser.add_argument("--model", type=str, required=True, help="Responses model id.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer id.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="OpenAI-compatible API key value.",
    )
    parser.add_argument(
        "--client-transport",
        type=str,
        default="both",
        choices=["http_sse", "websocket", "both"],
        help="Client-to-router transport to benchmark.",
    )
    parser.add_argument(
        "--chain-mode",
        type=str,
        default="full_replay",
        choices=["full_replay", "previous_response_id"],
        help="Conversation chaining strategy.",
    )
    parser.add_argument(
        "--store-mode",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to set store=true or store=false on Responses requests.",
    )
    parser.add_argument(
        "--worker-transport",
        type=str,
        default="unknown",
        choices=["unknown", "http", "grpc"],
        help="Metadata label for the router-to-worker transport.",
    )
    parser.add_argument(
        "--router-topology",
        type=str,
        default="unknown",
        choices=["unknown", "regular_http_worker", "regular_grpc_worker", "pd_http_workers"],
        help="Metadata label for the router topology.",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--num-qa", type=int, default=20)
    parser.add_argument("--min-len-q", type=int, default=256)
    parser.add_argument("--max-len-q", type=int, default=512)
    parser.add_argument("--min-len-a", type=int, default=4)
    parser.add_argument("--max-len-a", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout-secs", type=float, default=180)
    parser.add_argument(
        "--capture-event-trace",
        action="store_true",
        help="Capture a bounded per-turn event trace in the summary output.",
    )
    parser.add_argument(
        "--event-trace-limit",
        type=int,
        default=12,
        help="Maximum number of event trace entries to record per turn.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument(
        "--result-file",
        type=str,
        default=None,
        help="JSONL file for compact run summaries (omit to skip file output).",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="JSON file for full run details (omit to skip file output).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.base_url = _normalize_base_url(args.base_url)

    if args.long:
        args.min_len_a = 256
        args.max_len_a = 512
        args.num_qa = 20

    _validate_args(args)
    workloads = _build_workloads(args)

    transport_results = {}
    for client_transport in _transport_targets(args.client_transport):
        transport_results[client_transport] = _run_transport(
            args=args,
            client_transport=client_transport,
            workloads=workloads,
        )

    payload = {
        "task": "multi_turn_chat_router_responses",
        "router_url": args.base_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "chain_mode": args.chain_mode,
        "store_mode": _store_mode_label(args.store_mode == "true"),
        "parallel": args.parallel,
        "num_requests": args.num_qa,
        "num_turns": args.turns,
        "request_length_range": [args.min_len_q, args.max_len_q],
        "output_length_range": [args.min_len_a, args.max_len_a],
        "results": transport_results,
    }

    _write_summary(args.summary_file, payload)
    _append_result_line(
        args.result_file,
        {
            "task": payload["task"],
            "router_url": args.base_url,
            "model": args.model,
            "chain_mode": args.chain_mode,
            "store_mode": payload["store_mode"],
            "parallel": args.parallel,
            "num_requests": args.num_qa,
            "num_turns": args.turns,
            "results": {
                transport: result["summary"]
                for transport, result in transport_results.items()
            },
        },
    )

    print(json.dumps(payload["results"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
