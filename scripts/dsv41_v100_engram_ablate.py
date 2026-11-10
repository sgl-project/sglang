#!/usr/bin/env python3
"""Engram on-vs-zero traces against a live Flash server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS = [
    (
        "323",
        "What is 17*19? Reply with only the integer.",
        32,
    ),
    (
        "copy",
        "Repeat exactly, with no extra words:\nThe quick brown fox jumps over the lazy dog.",
        32,
    ),
    (
        "code",
        "Write a Python function is_palindrome(s) that returns True iff s equals "
        "s[::-1]. Output only the function.",
        96,
    ),
    (
        "fact",
        "In one short sentence: what particle mediates the electromagnetic force?",
        32,
    ),
]


def req(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    r = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(
    base: str, content: str, max_tokens: int, timeout: int
) -> dict[str, Any]:
    return req(
        base.rstrip("/") + "/v1/chat/completions",
        {
            "model": "default",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": 5,
        },
        timeout=timeout,
    )


def generate_logprob(base: str, text: str, timeout: int) -> dict[str, Any]:
    return req(
        base.rstrip("/") + "/generate",
        {
            "text": text,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": 8,
        },
        timeout=timeout,
    )


def summarize_chat(out: dict[str, Any]) -> dict[str, Any]:
    msg = out["choices"][0]["message"]
    text = msg.get("content") or ""
    lp = out["choices"][0].get("logprobs") or {}
    content_lp = lp.get("content") or []
    first = content_lp[0] if content_lp else None
    return {
        "text": text,
        "n_tokens": len(content_lp),
        "first_token": None
        if first is None
        else {
            "token": first.get("token"),
            "logprob": first.get("logprob"),
            "top": (first.get("top_logprobs") or [])[:5],
        },
    }


def summarize_generate(out: dict[str, Any]) -> dict[str, Any]:
    meta = out.get("meta_info") or {}
    return {
        "text": out.get("text"),
        "output_ids": out.get("output_ids"),
        "input_token_logprobs": meta.get("input_token_logprobs"),
        "output_token_logprobs": meta.get("output_token_logprobs"),
        "output_top_logprobs": meta.get("output_top_logprobs"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:11435")
    p.add_argument("--tag", required=True, help="on or zero")
    p.add_argument("--out-dir", default="/tmp/dsv41-engram-ablate")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"tag": args.tag, "prompts": {}}
    for name, prompt, max_tokens in PROMPTS:
        print(f"=== {args.tag} chat {name} ===", flush=True)
        t0 = time.time()
        chat_out = chat(args.base, prompt, max_tokens, args.timeout)
        dt = time.time() - t0
        chat_sum = summarize_chat(chat_out)
        print(f"elapsed={dt:.1f}s text={chat_sum['text']!r}", flush=True)

        print(f"=== {args.tag} prefill-logprob {name} ===", flush=True)
        t1 = time.time()
        gen_out = generate_logprob(args.base, prompt, args.timeout)
        dt1 = time.time() - t1
        gen_sum = summarize_generate(gen_out)
        print(f"elapsed={dt1:.1f}s next={gen_sum['text']!r}", flush=True)
        results["prompts"][name] = {
            "prompt": prompt,
            "chat_s": dt,
            "chat": chat_sum,
            "prefill_logprob_s": dt1,
            "prefill": gen_sum,
        }

    path = out_dir / f"{args.tag}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
