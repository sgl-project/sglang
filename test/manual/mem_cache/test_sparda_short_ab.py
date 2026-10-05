#!/usr/bin/env python3
"""Small, reproducible SGLang A/B timing harness for the SparDA path.

This is deliberately a manual experiment.  It uses synthetic, fixed token IDs
and greedy decoding so that the two server configurations can be compared
without depending on a tokenizer or on model text quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def _request_stream(
    base_url: str,
    input_ids: list[int],
    max_new_tokens: int,
    request_id: str,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
        },
        "stream": True,
        "rid": request_id,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    first_event = None
    last_event: dict[str, Any] = {}
    output_ids: list[int] = []
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip()
            if not data or data == b"[DONE]":
                continue
            event = json.loads(data)
            now = time.perf_counter()
            if first_event is None:
                first_event = now
            last_event = event
            if isinstance(event.get("output_ids"), list):
                output_ids = [int(token_id) for token_id in event["output_ids"]]

    finished = time.perf_counter()
    meta = last_event.get("meta_info") or {}
    completion_tokens = int(meta.get("completion_tokens") or len(output_ids))
    ttft = (first_event or finished) - started
    total = finished - started
    tpot = (total - ttft) / max(completion_tokens, 1)
    output_digest = hashlib.sha256(
        json.dumps(output_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "request_id": request_id,
        "prompt_tokens": len(input_ids),
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "total_s": total,
        "tpot_s": tpot,
        "output_ids": output_ids,
        "output_sha256": output_digest,
        "server_meta": meta,
    }


def _flush_cache(base_url: str) -> None:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/flush_cache",
        data=b"",
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()


def _fixed_input(token_count: int, offset: int = 0) -> list[int]:
    # Keep IDs comfortably inside the vocabulary range of MiniCPM4.1 while
    # making every position recognizable in an offline experiment.
    return [
        ((position * 17 + 11 + offset) % 10000) + 10 for position in range(token_count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--warmup-prompt-tokens", type=int)
    parser.add_argument("--warmup-offset", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--flush-before-timed", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_ids = _fixed_input(args.prompt_tokens)
    warmup_ids = input_ids
    if args.warmup_prompt_tokens is not None:
        warmup_ids = _fixed_input(args.warmup_prompt_tokens, args.warmup_offset)
    runs = []
    for index in range(args.warmup):
        runs.append(
            {
                "kind": "warmup",
                "result": _request_stream(
                    args.base_url,
                    warmup_ids,
                    args.max_new_tokens,
                    f"{args.label}-warmup-{index}",
                ),
            }
        )
    for index in range(args.repeats):
        if args.flush_before_timed:
            _flush_cache(args.base_url)
        runs.append(
            {
                "kind": "timed",
                "result": _request_stream(
                    args.base_url,
                    input_ids,
                    args.max_new_tokens,
                    f"{args.label}-timed-{index}",
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "label": args.label,
                "prompt_tokens": args.prompt_tokens,
                "max_new_tokens": args.max_new_tokens,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "runs": runs,
            },
            indent=2,
        )
        + "\n"
    )

    for run in runs:
        result = run["result"]
        print(
            run["kind"],
            result["ttft_s"],
            result["tpot_s"],
            result["total_s"],
            result["output_sha256"],
        )


if __name__ == "__main__":
    main()
