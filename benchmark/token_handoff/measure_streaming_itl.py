#!/usr/bin/env python3
"""Measure first-token and inter-chunk latency from an OpenAI-compatible stream."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass


@dataclass
class RequestResult:
    request_index: int
    status: str
    latency_ms: float
    first_token_ms: float | None
    chunk_count: int
    chunk_times_ms: list[float]
    inter_chunk_ms: list[float]
    error: str | None = None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_request(
    *,
    request_index: int,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    chunk_times: list[float] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                choices = event.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text")
                if text is None:
                    text = (choices[0].get("delta") or {}).get("content")
                if not text:
                    continue
                chunk_times.append((time.perf_counter() - started) * 1000.0)
        total_ms = (time.perf_counter() - started) * 1000.0
        return RequestResult(
            request_index=request_index,
            status="ok",
            latency_ms=total_ms,
            first_token_ms=chunk_times[0] if chunk_times else None,
            chunk_count=len(chunk_times),
            chunk_times_ms=chunk_times,
            inter_chunk_ms=[
                current - previous
                for previous, current in zip(chunk_times, chunk_times[1:])
            ],
        )
    except Exception as exc:
        return RequestResult(
            request_index=request_index,
            status="error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            first_token_ms=None,
            chunk_count=0,
            chunk_times_ms=[],
            inter_chunk_ms=[],
            error=repr(exc),
        )


def summarize(results: list[RequestResult]) -> dict:
    successful = [result for result in results if result.status == "ok"]
    first_tokens = [
        result.first_token_ms
        for result in successful
        if result.first_token_ms is not None
    ]
    all_itl = [value for result in successful for value in result.inter_chunk_ms]
    first_gaps = [
        result.inter_chunk_ms[0] for result in successful if result.inter_chunk_ms
    ]
    return {
        "requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "ttft_ms": {
            "median": statistics.median(first_tokens) if first_tokens else None,
            "p95": percentile(first_tokens, 0.95),
            "p99": percentile(first_tokens, 0.99),
        },
        "first_to_second_chunk_ms": {
            "median": statistics.median(first_gaps) if first_gaps else None,
            "p95": percentile(first_gaps, 0.95),
            "p99": percentile(first_gaps, 0.99),
        },
        "all_inter_chunk_ms": {
            "median": statistics.median(all_itl) if all_itl else None,
            "p95": percentile(all_itl, 0.95),
            "p99": percentile(all_itl, 0.99),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-repeat", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = "benchmark " * args.prompt_repeat
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                run_request,
                request_index=index,
                url=args.url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            for index in range(args.requests)
        ]
        results = [future.result() for future in futures]

    report = {
        "config": vars(args),
        "summary": summarize(results),
        "results": [asdict(result) for result in results],
    }
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(output)
            file.write("\n")


if __name__ == "__main__":
    main()
