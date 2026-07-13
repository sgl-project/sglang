#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import statistics
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def build_requests(n: int, max_tokens: int) -> list[dict]:
    base = (
        "We are validating distributed inference failover. "
        "Please produce a concise technical checklist with operational details. "
    )
    requests = []
    for i in range(n):
        category = ["short", "medium", "prefill_heavy", "tool_context", "long_context"][
            (i * 7 + 2) % 5
        ]
        repeat = {
            "short": 4,
            "medium": 10,
            "prefill_heavy": 26,
            "tool_context": 18,
            "long_context": 22,
        }[category]
        prompt = (
            f"Request {i + 1}. Category={category}. "
            + base * repeat
            + "End with numbered bullets and avoid long prose."
        )
        requests.append(
            {
                "request_id": f"trace-{i + 1:03d}",
                "category": category,
                "prompt_chars": len(prompt),
                "max_tokens": max_tokens,
                "prompt": prompt,
            }
        )
    return requests


def parse_sse_line(raw: bytes):
    line = raw.decode("utf-8", errors="replace").strip()
    if not line or line.startswith(":") or not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return "DONE"
    return json.loads(data)


def run_one(router_url: str, model: str, req: dict, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": req["prompt"]}],
        "max_tokens": req["max_tokens"],
        "temperature": 0,
        "stream": True,
    }
    started_wall = time.time()
    started = time.monotonic()
    first = None
    last = None
    chunks = 0
    intervals = []
    finish_reason = None
    content_chars = 0
    error = None
    try:
        request = urllib.request.Request(
            router_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                parsed = parse_sse_line(raw)
                if parsed is None:
                    continue
                if parsed == "DONE":
                    break
                choice = (parsed.get("choices") or [{}])[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                text = delta.get("content") or ""
                if not text:
                    continue
                now = time.monotonic()
                if first is None:
                    first = now
                elif last is not None:
                    intervals.append(now - last)
                last = now
                chunks += 1
                content_chars += len(text)
    except Exception as exc:  # noqa: BLE001 - record all client failures
        error = {
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=4),
        }
    ended = time.monotonic()
    ttft = (first - started) if first is not None else None
    avg_tpot = statistics.fmean(intervals) if intervals else None
    p95_tpot = percentile(intervals, 95) if intervals else None
    return {
        "request_id": req["request_id"],
        "category": req["category"],
        "prompt_chars": req["prompt_chars"],
        "max_tokens": req["max_tokens"],
        "start_epoch": started_wall,
        "elapsed_s": ended - started,
        "ttft_s": ttft,
        "avg_tpot_s": avg_tpot,
        "p95_tpot_s": p95_tpot,
        "chunks": chunks,
        "content_chars": content_chars,
        "finish_reason": finish_reason,
        "error": json.dumps(error, sort_keys=True) if error else "",
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[dict], ttft_slo: float, tpot_slo: float) -> dict:
    ok = [r for r in results if not r["error"] and r["ttft_s"] is not None]
    total = len(results)
    ttft_good = sum(1 for r in ok if r["ttft_s"] <= ttft_slo)
    tpot_good = sum(
        1
        for r in ok
        if r["avg_tpot_s"] is not None and r["avg_tpot_s"] <= tpot_slo
    )
    combined_good = sum(
        1
        for r in ok
        if r["ttft_s"] <= ttft_slo
        and r["avg_tpot_s"] is not None
        and r["avg_tpot_s"] <= tpot_slo
    )
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    tpots = [r["avg_tpot_s"] for r in ok if r["avg_tpot_s"] is not None]
    return {
        "requests": total,
        "completed": len(ok),
        "errors": total - len(ok),
        "ttft_slo_seconds": ttft_slo,
        "tpot_slo_seconds": tpot_slo,
        "ttft_attainment": ttft_good / total if total else 0,
        "tpot_attainment": tpot_good / total if total else 0,
        "combined_attainment": combined_good / total if total else 0,
        "avg_ttft_s": statistics.fmean(ttfts) if ttfts else None,
        "p95_ttft_s": percentile(ttfts, 95),
        "avg_tpot_s": statistics.fmean(tpots) if tpots else None,
        "p95_tpot_s": percentile(tpots, 95),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--ttft-slo", type=float, default=8.0)
    parser.add_argument("--tpot-slo", type=float, default=0.02)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace = build_requests(args.requests, args.max_tokens)
    write_csv(
        out / "trace_requests.csv",
        [{k: v for k, v in r.items() if k != "prompt"} for r in trace],
        ["request_id", "category", "prompt_chars", "max_tokens"],
    )
    (out / "trace_requests.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8"
    )

    results = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, args.router_url, args.model, r, args.timeout) for r in trace]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: r["request_id"])
    fields = [
        "request_id",
        "category",
        "prompt_chars",
        "max_tokens",
        "start_epoch",
        "elapsed_s",
        "ttft_s",
        "avg_tpot_s",
        "p95_tpot_s",
        "chunks",
        "content_chars",
        "finish_reason",
        "error",
    ]
    write_csv(out / "results.csv", results, fields)
    summary = summarize(results, args.ttft_slo, args.tpot_slo)
    summary["wall_seconds"] = time.monotonic() - start
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
