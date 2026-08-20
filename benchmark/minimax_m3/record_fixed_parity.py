#!/usr/bin/env python3
"""Record deterministic short and long-context MiniMax-M3 responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def long_prompt(approx_tokens: int) -> tuple[str, str]:
    marker = f"MSA-{approx_tokens}-C7F29A"
    sentence = "Archive row 314159 contains ordinary calibration material. "
    target_chars = approx_tokens * 4
    filler = (sentence * (target_chars // len(sentence) + 1))[:target_chars]
    insert_at = len(filler) * 2 // 3
    context = filler[:insert_at] + f"\nSECRET CODE: {marker}\n" + filler[insert_at:]
    prompt = (
        "Read the archive and return only the exact SECRET CODE, with no explanation.\n"
        f"<archive>\n{context}\n</archive>"
    )
    return prompt, marker


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def response_record(body: dict) -> tuple[str, str, str]:
    message = body["choices"][0]["message"]
    reasoning = str(message.get("reasoning_content") or "").strip()
    content = str(message.get("content") or "").strip()
    canonical = json.dumps(
        {"reasoning_content": reasoning, "content": content},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return reasoning, content, canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-tokens", type=int, nargs="+", default=[32768, 65536])
    args = parser.parse_args()

    cases = [
        (
            "short",
            "Return only the string MSA-SHORT-4B19. Do not add punctuation.",
            "MSA-SHORT-4B19",
        )
    ]
    cases.extend(
        (f"long_{tokens}", *long_prompt(tokens)) for tokens in args.long_tokens
    )
    records = []
    for name, prompt, expected in cases:
        body = post_json(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 512,
            },
        )
        reasoning, content, canonical = response_record(body)
        records.append(
            {
                "name": name,
                "expected": expected,
                "exact_expected": content == expected,
                "reasoning_content": reasoning,
                "content": content,
                "response_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "usage": body.get("usage", {}),
            }
        )
    payload = {"label": args.label, "model": args.model, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all(record["exact_expected"] for record in records):
        raise SystemExit("one or more fixed-answer probes failed")


if __name__ == "__main__":
    main()
