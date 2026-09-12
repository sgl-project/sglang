#!/usr/bin/env python3
"""Run a small long-context needle test against one SGLang server.

The task labels are known strings embedded in synthetic context.  The same
prompts can be sent to a dense reference and to SparDA; this measures both
answer accuracy and whether the sparse selector preserves the task output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


def _prompt(target_tokens: int, code: str, case_index: int) -> str:
    header = (
        "You are answering a retrieval test. Read every record carefully. "
        "One record contains a secret four-digit code. "
        "At the end, output only that four-digit code.\n\n"
    )
    suffix = (
        "\nQuestion: What four-digit code appears in the record marked "
        f"NEEDLE-{case_index:02d}?\nAnswer:"
    )
    filler = (
        "Record {index:05d}: This ordinary record contains no secret code; "
        "its reference value is {value:04d}.\n"
    )
    target_chars = max(256, target_tokens * 4)
    records = []
    index = 0
    while len(header) + len("".join(records)) + len(suffix) < target_chars:
        if index == (target_tokens // 8):
            records.append(
                f"Record {index:05d}: NEEDLE-{case_index:02d} stores secret "
                f"code {code}.\n"
            )
        else:
            records.append(filler.format(index=index, value=(index * 37) % 10000))
        index += 1
    return header + "".join(records) + suffix


def _request(
    base_url: str,
    prompt: str,
    expected: str,
    max_new_tokens: int,
    request_id: str,
) -> dict[str, Any]:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": True,
        "top_logprobs_num": 1,
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

    text = str(result.get("text", ""))
    meta = result.get("meta_info") or {}
    output_ids = [int(token_id) for token_id in result.get("output_ids", [])]
    logprobs = []
    for item in meta.get("output_token_logprobs") or []:
        if isinstance(item, (list, tuple)) and item:
            logprobs.append(float(item[0]))
        elif isinstance(item, dict) and "logprob" in item:
            logprobs.append(float(item["logprob"]))
    normalized = re.sub(r"[^0-9]", "", text)
    return {
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
        "completion_tokens": len(output_ids),
        "elapsed_s": elapsed,
        "expected_code": expected,
        "output_text": text,
        "output_ids": output_ids,
        "output_sha256": hashlib.sha256(
            json.dumps(output_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "output_logprobs": logprobs,
        "exact_code_match": normalized == expected,
        "code_present": expected in normalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--prompt-tokens", type=int, nargs="+", default=[1024, 4096, 8192]
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    probes = []
    for case_index, prompt_tokens in enumerate(args.prompt_tokens):
        code = f"{7319 + case_index * 137:04d}"
        probes.append(
            (prompt_tokens, case_index, code, _prompt(prompt_tokens, code, case_index))
        )

    runs = []
    for warmup_index in range(args.warmup):
        prompt_tokens, case_index, code, prompt = probes[0]
        runs.append(
            {
                "kind": "warmup",
                "result": _request(
                    args.base_url,
                    prompt,
                    code,
                    args.max_new_tokens,
                    f"{args.label}-warmup-{warmup_index}",
                ),
            }
        )
    for case_index, (prompt_tokens, _, code, prompt) in enumerate(probes):
        runs.append(
            {
                "kind": "probe",
                "requested_prompt_tokens": prompt_tokens,
                "case_index": case_index,
                "result": _request(
                    args.base_url,
                    prompt,
                    code,
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
                "task": "synthetic long-context needle retrieval",
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
            result["exact_code_match"],
            result["code_present"],
            result["elapsed_s"],
        )


if __name__ == "__main__":
    main()
