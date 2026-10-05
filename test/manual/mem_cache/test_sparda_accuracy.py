#!/usr/bin/env python3
"""Record deterministic MiniCPM greedy outputs for a sparse/reference run.

The probes use fixed token IDs so the same requests can be replayed against
two server configurations.  The resulting output IDs and token log-probs are
data for a paired correctness check; they are not a task-quality benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def _fixed_input(token_count: int, offset: int) -> list[int]:
    return [
        ((position * 17 + 11 + offset) % 10000) + 10 for position in range(token_count)
    ]


def _request(
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
        "return_logprob": True,
        "top_logprobs_num": 1,
        "logprob_start_len": 0,
        "stream": False,
        "rid": request_id,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    if "error" in result:
        raise RuntimeError(result["error"])

    meta = result.get("meta_info") or {}
    output_ids = [int(token_id) for token_id in result.get("output_ids", [])]
    output_logprobs = meta.get("output_token_logprobs") or []
    normalized_logprobs = []
    for item in output_logprobs:
        if isinstance(item, (list, tuple)) and item:
            normalized_logprobs.append(float(item[0]))
        elif isinstance(item, dict) and "logprob" in item:
            normalized_logprobs.append(float(item["logprob"]))

    return {
        "prompt_tokens": len(input_ids),
        "completion_tokens": len(output_ids),
        "elapsed_s": elapsed,
        "output_ids": output_ids,
        "output_sha256": hashlib.sha256(
            json.dumps(output_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "output_logprobs": normalized_logprobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--prompt-tokens", type=int, nargs="+", default=[1024, 4096, 8192]
    )
    parser.add_argument("--offset", type=int, nargs="+", default=[0, 5000])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = [
        (prompt_tokens, offset)
        for prompt_tokens in args.prompt_tokens
        for offset in args.offset
    ]
    runs = []
    for warmup_index in range(args.warmup):
        prompt_tokens, offset = cases[0]
        runs.append(
            {
                "kind": "warmup",
                "result": _request(
                    args.base_url,
                    _fixed_input(prompt_tokens, offset),
                    args.max_new_tokens,
                    f"{args.label}-warmup-{warmup_index}",
                ),
            }
        )
    for case_index, (prompt_tokens, offset) in enumerate(cases):
        runs.append(
            {
                "kind": "probe",
                "prompt_tokens": prompt_tokens,
                "offset": offset,
                "result": _request(
                    args.base_url,
                    _fixed_input(prompt_tokens, offset),
                    args.max_new_tokens,
                    f"{args.label}-probe-{case_index}",
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "label": args.label,
                "probe_type": "fixed-token greedy generation",
                "max_new_tokens": args.max_new_tokens,
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
            result["prompt_tokens"],
            result["completion_tokens"],
            result["output_sha256"],
        )


if __name__ == "__main__":
    main()
