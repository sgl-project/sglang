"""Measure cold-prefix first-text and completion latency through /generate.

Run against a server started with SGLANG_ENABLE_DLLM_KV_ONLY_PREFILL=0 or 1.
Input JSONL records contain input_ids and optional uid, dataset, and input_sha.
Use the same manifest, ordering, and batch size for both server configurations.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import requests


def measure(session, base_url, records, max_new_tokens, timeout):
    response = session.post(base_url + "/flush_cache?timeout=30", timeout=timeout)
    response.raise_for_status()
    first_text = [None] * len(records)
    completed = [None] * len(records)
    outputs = [None] * len(records)
    start = time.perf_counter()
    with session.post(
        base_url + "/generate",
        json={
            "input_ids": [record["input_ids"] for record in records],
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": False,
            },
        },
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1024):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            result = json.loads(line[6:])
            elapsed = (time.perf_counter() - start) * 1000
            index = result.get("index", 0)
            finish_reason = result.get("meta_info", {}).get("finish_reason")
            if finish_reason and finish_reason.get("type") == "abort":
                raise RuntimeError(f"Generation aborted: {finish_reason}")
            if result.get("text", "").strip() and first_text[index] is None:
                first_text[index] = elapsed
            if finish_reason:
                completed[index] = elapsed
                outputs[index] = result
    if any(result is None for result in outputs):
        raise RuntimeError("The stream ended before every request completed")
    rows = []
    for index, (record, result) in enumerate(zip(records, outputs, strict=True)):
        ids = result["output_ids"]
        rows.append(
            {
                "uid": record.get("uid"),
                "dataset": record.get("dataset"),
                "input_tokens": len(record["input_ids"]),
                "input_sha256": hashlib.sha256(
                    json.dumps(record["input_ids"]).encode()
                ).hexdigest(),
                "output_tokens": len(ids),
                "output_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "first_text_ms": first_text[index],
                "e2e_ms": completed[index],
                "finish_reason": result["meta_info"]["finish_reason"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1, choices=(1, 4))
    parser.add_argument("--input-length", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=360)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.input_length is not None:
        records = [
            record
            for record in records
            if args.input_length <= len(record["input_ids"]) <= args.input_length + 31
        ]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        parser.error("No inputs match this configuration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    measurements = []
    with (
        requests.Session() as session,
        args.output.open("w", encoding="utf-8") as output,
    ):
        for _ in range(args.warmup):
            measure(
                session, args.base_url, records[: args.batch_size], 32, args.timeout
            )
        for offset in range(0, len(records), args.batch_size):
            batch = records[offset : offset + args.batch_size]
            rows = measure(
                session, args.base_url, batch, args.max_new_tokens, args.timeout
            )
            measurements.extend(rows)
            for row in rows:
                output.write(json.dumps(row) + "\n")
            output.flush()
    first_text = [
        row["first_text_ms"] for row in measurements if row["first_text_ms"] is not None
    ]
    print(
        json.dumps(
            {
                "requests": len(measurements),
                "first_text_requests": len(first_text),
                "mean_first_text_ms": statistics.mean(first_text)
                if first_text
                else None,
                "mean_e2e_ms": statistics.mean(row["e2e_ms"] for row in measurements),
                "truncated": sum(
                    row["finish_reason"]["type"] == "length" for row in measurements
                ),
                "input_file_sha256": hashlib.sha256(
                    args.input_jsonl.read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
