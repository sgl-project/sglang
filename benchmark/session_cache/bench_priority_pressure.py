#!/usr/bin/env python3

"""Measure session-cache demotion under single-engine KV pressure."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class GenerateResult:
    phase: str
    role: str
    session_id: str
    round: int
    input_tokens: int
    cached_tokens: int
    latency_ms: float
    success: bool
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare protected and demoted radix-native sessions under device-KV "
            "pressure. Run each arm against a fresh server."
        )
    )
    parser.add_argument("--arm", choices=("protected", "demoted"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--high-sessions", type=int, default=6)
    parser.add_argument("--low-sessions", type=int, default=8)
    parser.add_argument("--prime-tokens", type=int, default=6144)
    parser.add_argument("--growth-tokens", type=int, default=2048)
    parser.add_argument("--pressure-rounds", type=int, default=4)
    parser.add_argument("--pressure-concurrency", type=int, default=8)
    parser.add_argument(
        "--token-base",
        type=int,
        default=1000,
        help="First token ID used for synthetic prompts; override for small vocabularies.",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def request_json(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float,
) -> Any:
    response = requests.request(method, url, json=body, timeout=timeout)
    if not response.ok:
        raise RuntimeError(
            f"{method} {url} failed ({response.status_code}): {response.text}"
        )
    if not response.content:
        return None
    content_type = response.headers.get("content-type", "")
    return response.json() if "json" in content_type else response.text


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def make_prompt(token_id: int, length: int) -> list[int]:
    return [token_id] * length


def generate(
    base_url: str,
    session_id: str,
    input_ids: list[int],
    *,
    phase: str,
    role: str,
    round_index: int,
    timeout: float,
) -> GenerateResult:
    started = time.perf_counter()
    try:
        response = request_json(
            "POST",
            base_url + "/generate",
            body={
                "input_ids": input_ids,
                "session_id": session_id,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
            },
            timeout=timeout,
        )
        meta = response["meta_info"]
        return GenerateResult(
            phase=phase,
            role=role,
            session_id=session_id,
            round=round_index,
            input_tokens=len(input_ids),
            cached_tokens=int(meta.get("cached_tokens", 0)),
            latency_ms=(time.perf_counter() - started) * 1000,
            success=True,
        )
    except Exception as error:
        return GenerateResult(
            phase=phase,
            role=role,
            session_id=session_id,
            round=round_index,
            input_tokens=len(input_ids),
            cached_tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000,
            success=False,
            error=str(error),
        )


def run_batch(
    base_url: str,
    sessions: list[tuple[str, int]],
    *,
    input_tokens: int,
    phase: str,
    role: str,
    round_index: int,
    concurrency: int,
    timeout: float,
) -> list[GenerateResult]:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(concurrency, len(sessions))
    ) as executor:
        futures = [
            executor.submit(
                generate,
                base_url,
                session_id,
                make_prompt(token_id, input_tokens),
                phase=phase,
                role=role,
                round_index=round_index,
                timeout=timeout,
            )
            for session_id, token_id in sessions
        ]
        return [future.result() for future in futures]


def summarize(events: list[GenerateResult], prime_tokens: int) -> dict[str, Any]:
    high_probes = [event for event in events if event.phase == "probe_high"]
    low_pressure = [event for event in events if event.phase == "pressure_low"]
    successful_high = [event for event in high_probes if event.success]
    successful_low = [event for event in low_pressure if event.success]
    high_cached = [event.cached_tokens for event in successful_high]
    high_latencies = [event.latency_ms for event in successful_high]
    low_latencies = [event.latency_ms for event in successful_low]
    return {
        "all_requests_succeeded": all(event.success for event in events),
        "failed_requests": sum(not event.success for event in events),
        "low_pressure_completed": len(successful_low),
        "low_pressure_total": len(low_pressure),
        "high_probe_completed": len(successful_high),
        "high_probe_total": len(high_probes),
        "high_cached_tokens_mean": (
            statistics.fmean(high_cached) if high_cached else None
        ),
        "high_cached_fraction_mean": (
            statistics.fmean(value / prime_tokens for value in high_cached)
            if high_cached
            else None
        ),
        "high_cached_fraction_min": (
            min(value / prime_tokens for value in high_cached) if high_cached else None
        ),
        "high_probe_latency_ms_p50": percentile(high_latencies, 0.50),
        "high_probe_latency_ms_p95": percentile(high_latencies, 0.95),
        "low_pressure_latency_ms_p50": percentile(low_latencies, 0.50),
        "low_pressure_latency_ms_p95": percentile(low_latencies, 0.95),
    }


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    if (
        min(
            args.high_sessions,
            args.low_sessions,
            args.prime_tokens,
            args.growth_tokens,
            args.pressure_rounds,
            args.pressure_concurrency,
        )
        <= 0
    ):
        raise ValueError(
            "session counts, token counts, rounds, and concurrency must be positive"
        )

    server_info = request_json("GET", base_url + "/server_info", timeout=args.timeout)
    if not server_info.get("enable_session_radix_cache"):
        raise RuntimeError("server must use --enable-session-radix-cache")

    max_total_tokens = int(server_info["max_total_num_tokens"])
    initial_tokens = (args.high_sessions + args.low_sessions) * args.prime_tokens
    final_tokens = args.high_sessions * args.prime_tokens + args.low_sessions * (
        args.prime_tokens + args.pressure_rounds * args.growth_tokens
    )
    if initial_tokens >= max_total_tokens:
        raise ValueError(
            f"initial working set {initial_tokens} must be below device KV capacity "
            f"{max_total_tokens}"
        )
    if final_tokens <= max_total_tokens:
        raise ValueError(
            f"final working set {final_tokens} must exceed device KV capacity "
            f"{max_total_tokens}"
        )

    request_json("POST", base_url + "/flush_cache", timeout=args.timeout)
    run_id = uuid.uuid4().hex[:12]
    low_sessions = [
        (f"priority-pressure-{run_id}-low-{index}", args.token_base + index)
        for index in range(args.low_sessions)
    ]
    high_sessions = [
        (
            f"priority-pressure-{run_id}-high-{index}",
            args.token_base + args.low_sessions + index,
        )
        for index in range(args.high_sessions)
    ]
    opened: list[str] = []
    events: list[GenerateResult] = []
    controls: list[dict[str, Any]] = []

    try:
        capacity = args.prime_tokens + args.pressure_rounds * args.growth_tokens + 1024
        for session_id, _ in low_sessions + high_sessions:
            returned_id = request_json(
                "POST",
                base_url + "/open_session",
                body={"session_id": session_id, "capacity_of_str_len": capacity},
                timeout=args.timeout,
            )
            if returned_id != session_id:
                raise RuntimeError(
                    f"open_session returned {returned_id!r} for {session_id!r}"
                )
            opened.append(session_id)

        events.extend(
            run_batch(
                base_url,
                low_sessions,
                input_tokens=args.prime_tokens,
                phase="prime_low",
                role="low",
                round_index=0,
                concurrency=args.pressure_concurrency,
                timeout=args.timeout,
            )
        )
        if not all(event.success for event in events):
            raise RuntimeError("low-session priming failed")

        for session_id, _ in low_sessions:
            result = request_json(
                "POST",
                base_url + "/set_session_cache_priority",
                body={
                    "session_id": session_id,
                    "cache_priority": (
                        "evictable" if args.arm == "demoted" else "protected"
                    ),
                },
                timeout=args.timeout,
            )
            controls.append({"session_id": session_id, "response": result})
            targeted = [item for item in result if item["status"] != "not_targeted"]
            if not targeted or not all(item["success"] for item in targeted):
                raise RuntimeError(f"priority update failed for {session_id}: {result}")

        events.extend(
            run_batch(
                base_url,
                high_sessions,
                input_tokens=args.prime_tokens,
                phase="prime_high",
                role="high",
                round_index=0,
                concurrency=args.pressure_concurrency,
                timeout=args.timeout,
            )
        )
        if not all(event.success for event in events):
            raise RuntimeError("session priming failed")

        for round_index in range(1, args.pressure_rounds + 1):
            events.extend(
                run_batch(
                    base_url,
                    low_sessions,
                    input_tokens=args.prime_tokens + round_index * args.growth_tokens,
                    phase="pressure_low",
                    role="low",
                    round_index=round_index,
                    concurrency=args.pressure_concurrency,
                    timeout=args.timeout,
                )
            )

        events.extend(
            run_batch(
                base_url,
                high_sessions,
                input_tokens=args.prime_tokens,
                phase="probe_high",
                role="high",
                round_index=args.pressure_rounds + 1,
                concurrency=args.pressure_concurrency,
                timeout=args.timeout,
            )
        )
    finally:
        for session_id in opened:
            try:
                request_json(
                    "POST",
                    base_url + "/close_session",
                    body={"session_id": session_id},
                    timeout=args.timeout,
                )
            except Exception as error:
                controls.append({"session_id": session_id, "close_error": str(error)})

    output = {
        "run_id": run_id,
        "arm": args.arm,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "server": {
            "version": server_info.get("version"),
            "model_path": server_info.get("model_path"),
            "max_total_num_tokens": max_total_tokens,
            "radix_eviction_policy": server_info.get("radix_eviction_policy"),
            "enable_session_radix_cache": server_info.get("enable_session_radix_cache"),
        },
        "working_set": {
            "initial_tokens": initial_tokens,
            "final_tokens": final_tokens,
            "capacity_tokens": max_total_tokens,
            "pressure_tokens": final_tokens - max_total_tokens,
        },
        "controls": controls,
        "events": [asdict(event) for event in events],
        "summary": summarize(events, args.prime_tokens),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    if not output["summary"]["all_requests_succeeded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
