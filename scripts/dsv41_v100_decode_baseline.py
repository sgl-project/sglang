#!/usr/bin/env python3
"""np=1 decode tok/s + TTFT against a live Flash server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def req(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    r = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(base: str, content: str, max_tokens: int, timeout: int) -> dict[str, Any]:
    return req(
        base.rstrip("/") + "/v1/chat/completions",
        {
            "model": "default",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )


def generate(
    base: str,
    *,
    text: str | None = None,
    input_ids: list[int] | None = None,
    max_new: int,
    timeout: int,
    ignore_eos: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sampling_params": {
            "max_new_tokens": max_new,
            "temperature": 0,
            "ignore_eos": ignore_eos,
        }
    }
    if input_ids is not None:
        payload["input_ids"] = input_ids
    else:
        payload["text"] = text
    return req(base.rstrip("/") + "/generate", payload, timeout=timeout)


def meta_slice(out: dict[str, Any]) -> dict[str, Any]:
    meta = dict(out.get("meta_info") or {})
    keep = (
        "prompt_tokens",
        "completion_tokens",
        "e2e_latency",
        "prefill_latency",
        "decode_latency",
        "decode_throughput",
        "time_to_first_token",
        "ttft",
    )
    return {k: meta[k] for k in keep if k in meta}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:11435")
    p.add_argument("--out", default="/tmp/dsv41-decode-baseline.json")
    p.add_argument("--decode-new", type=int, default=32)
    p.add_argument(
        "--prefills",
        default="8k",
        help="comma list: 8k,32k,250k",
    )
    p.add_argument("--timeout", type=int, default=86400)
    args = p.parse_args()
    prefills = {
        "8k": 8192,
        "32k": 32768,
        "250k": 250000,
    }
    wanted = [s.strip() for s in args.prefills.split(",") if s.strip()]

    results: dict[str, Any] = {"steps": {}}

    print("=== greedy 17*19 ===", flush=True)
    t0 = time.time()
    out = chat(
        args.base,
        "What is 17*19? Reply with only the integer.",
        max_tokens=32,
        timeout=args.timeout,
    )
    text = out["choices"][0]["message"].get("content") or ""
    dt = time.time() - t0
    usage = out.get("usage") or {}
    print(f"elapsed={dt:.1f}s text={text!r} usage={usage}", flush=True)
    results["steps"]["323"] = {
        "elapsed_s": dt,
        "text": text,
        "usage": usage,
        "ok": "323" in text.replace(",", "").replace(" ", ""),
    }

    print(f"=== decode max_new={args.decode_new} (ignore_eos) ===", flush=True)
    t0 = time.time()
    gout = generate(
        args.base,
        text="Say the word ping and then keep counting from 1.",
        max_new=args.decode_new,
        timeout=args.timeout,
        ignore_eos=True,
    )
    dt = time.time() - t0
    ms = meta_slice(gout)
    n = int(ms.get("completion_tokens") or args.decode_new)
    wall_tok_s = (n / dt) if dt > 0 else None
    print(f"elapsed={dt:.1f}s wall_tok_s={wall_tok_s} meta={ms}", flush=True)
    results["steps"]["decode"] = {
        "elapsed_s": dt,
        "wall_tok_s": wall_tok_s,
        "meta": ms,
        "text": (gout.get("text") or "")[:200],
    }

    for name in wanted:
        ntok = prefills[name]
        print(f"=== TTFT {name} ({ntok} tokens, max_new=1) ===", flush=True)
        t0 = time.time()
        pout = generate(
            args.base,
            input_ids=[1] * ntok,
            max_new=1,
            timeout=args.timeout,
            ignore_eos=True,
        )
        dt = time.time() - t0
        ms = meta_slice(pout)
        print(f"elapsed={dt:.1f}s meta={ms}", flush=True)
        results["steps"][f"prefill_{name}"] = {"elapsed_s": dt, "meta": ms}

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}", flush=True)
    return 0 if results["steps"]["323"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
