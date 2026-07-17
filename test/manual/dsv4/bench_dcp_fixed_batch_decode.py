#!/usr/bin/env python3
"""Benchmark DCP decode with one pre-tokenized, fixed-size shared-prefix batch."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prefix-len", type=int, default=128000)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-output-len", type=int, default=8)
    parser.add_argument("--token-start", type=int, default=100)
    parser.add_argument("--token-cycle", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900)
    return parser.parse_args()


def make_payload(input_ids: list[list[int]], output_len: int) -> dict[str, Any]:
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
            "stream_interval": 1,
        },
        "stream": True,
    }


def encode_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def post_json(base_url: str, payload: dict[str, Any], timeout: float) -> None:
    response = requests.post(
        f"{base_url}/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(result["error"])


def run_batch(
    *,
    base_url: str,
    body: bytes,
    batch_size: int,
    output_len: int,
    timeout: float,
) -> dict[str, float | int]:
    start = time.perf_counter()
    last_ttft = 0.0
    last_completion_tokens = 0

    with requests.post(
        f"{base_url}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line or not raw_line.startswith(b"data:"):
                continue
            if raw_line == b"data: [DONE]":
                break

            data = json.loads(raw_line[5:].strip())
            if data.get("error"):
                raise RuntimeError(data["error"])
            meta_info = data["meta_info"]
            completion_tokens = int(meta_info["completion_tokens"])
            last_completion_tokens = max(last_completion_tokens, completion_tokens)
            if completion_tokens == 1:
                # A batched request can emit one first-token event per request.
                last_ttft = time.perf_counter() - start

    latency = time.perf_counter() - start
    if last_ttft <= 0:
        raise RuntimeError("stream did not report a first token")
    if last_completion_tokens != output_len:
        raise RuntimeError(
            f"completion_tokens={last_completion_tokens}, expected={output_len}"
        )

    decode_seconds = latency - last_ttft
    decode_intervals = max(output_len - 1, 1)
    return {
        "latency_s": latency,
        "last_ttft_s": last_ttft,
        "decode_seconds": decode_seconds,
        "step_tpot_ms": 1000 * decode_seconds / decode_intervals,
        "output_throughput": batch_size * decode_intervals / decode_seconds,
        "completion_tokens": last_completion_tokens,
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    for name in (
        "batch_size",
        "prefix_len",
        "output_len",
        "repeats",
        "warmup_output_len",
        "token_cycle",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.prefix_len + args.output_len > 131072:
        raise ValueError("prefix_len + output_len must not exceed 131072")
    if args.result_file.exists():
        raise FileExistsError(args.result_file)

    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    prefix = [
        args.token_start + (position % args.token_cycle)
        for position in range(args.prefix_len)
    ]
    batch_input_ids = [prefix] * args.batch_size

    flush_response = requests.post(
        f"{args.base_url}/flush_cache", timeout=args.timeout
    )
    flush_response.raise_for_status()
    post_json(
        args.base_url,
        {
            "input_ids": [prefix],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
            "stream": False,
        },
        args.timeout,
    )

    warmup_body = encode_payload(
        make_payload(batch_input_ids, args.warmup_output_len)
    )
    measured_body = encode_payload(make_payload(batch_input_ids, args.output_len))
    print(
        f"fixed batch: B={args.batch_size} prefix={args.prefix_len} "
        f"output={args.output_len} body={len(measured_body) / (1024**2):.1f} MiB",
        flush=True,
    )

    run_batch(
        base_url=args.base_url,
        body=warmup_body,
        batch_size=args.batch_size,
        output_len=args.warmup_output_len,
        timeout=args.timeout,
    )

    rows = []
    for repeat in range(1, args.repeats + 1):
        metrics = run_batch(
            base_url=args.base_url,
            body=measured_body,
            batch_size=args.batch_size,
            output_len=args.output_len,
            timeout=args.timeout,
        )
        row = {
            "record_type": "run",
            "variant": args.variant,
            "batch_size": args.batch_size,
            "prefix_len": args.prefix_len,
            "output_len": args.output_len,
            "repeat": repeat,
            **metrics,
        }
        rows.append(row)
        append_jsonl(args.result_file, row)
        print(
            f"repeat={repeat} step_tpot={metrics['step_tpot_ms']:.3f} ms "
            f"output_tok_s={metrics['output_throughput']:.2f}",
            flush=True,
        )

    summary = {
        "record_type": "summary",
        "variant": args.variant,
        "batch_size": args.batch_size,
        "prefix_len": args.prefix_len,
        "output_len": args.output_len,
        "repeats": args.repeats,
        "median_step_tpot_ms": statistics.median(
            row["step_tpot_ms"] for row in rows
        ),
        "median_output_throughput": statistics.median(
            row["output_throughput"] for row in rows
        ),
        "min_step_tpot_ms": min(row["step_tpot_ms"] for row in rows),
        "max_step_tpot_ms": max(row["step_tpot_ms"] for row in rows),
    }
    append_jsonl(args.result_file, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
