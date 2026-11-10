#!/usr/bin/env python3
"""Warmup, arm CUDA_PROFILER on TARGET_VERIFY, run coding-1.

nsys must already be wrapping the serve process with
--capture-range=cudaProfilerApi. This script only triggers Start/Stop via
/start_profile (3 verify forwards). Stacks stay off.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from typing import Any


def req(url: str, payload: dict[str, Any] | None = None, timeout: int = 1800) -> Any:
    if payload is None:
        r = urllib.request.Request(url)
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else {}
    data = json.dumps(payload).encode()
    r = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


def chat(base: str, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    return req(
        base.rstrip("/") + "/v1/chat/completions",
        {
            "model": "default",
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "return_spec_tokens_details": True,
        },
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:11435")
    p.add_argument("--num-verify", type=int, default=3)
    args = p.parse_args()
    base = args.base.rstrip("/")

    print("=== greedy 323 warmup ===", flush=True)
    t0 = time.time()
    out = chat(base, [{"role": "user", "content": "17*19="}], 8)
    text = ((out.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    print(
        json.dumps(
            {
                "step": "warmup-323",
                "elapsed_s": round(time.time() - t0, 2),
                "text": text[:80],
                "ok": "323" in text.replace(" ", ""),
            }
        ),
        flush=True,
    )

    print("=== arm CUDA_PROFILER verify x%d ===" % args.num_verify, flush=True)
    arm = req(
        base + "/start_profile",
        {
            "num_steps": args.num_verify,
            "activities": ["CUDA_PROFILER"],
            "profile_by_stage": True,
            "profile_stages": ["verify"],
            "with_stack": False,
            "record_shapes": False,
        },
        timeout=60,
    )
    print(json.dumps({"step": "start_profile", "resp": arm}), flush=True)

    print("=== coding-1 write (merge_sorted) ===", flush=True)
    t0 = time.time()
    out = chat(
        base,
        [
            {
                "role": "user",
                "content": (
                    "Write a Python function merge_sorted(a, b) that merges two "
                    "already-sorted lists of ints into one sorted list. Include a "
                    "one-line docstring. Output only the function, no markdown."
                ),
            }
        ],
        256,
    )
    dt = time.time() - t0
    usage = out.get("usage") or {}
    n = int(usage.get("completion_tokens") or 0)
    choice0 = (out.get("choices") or [{}])[0]
    msg = choice0.get("message") or {}
    sglext = out.get("sglext") or choice0.get("sglext") or {}
    spec = sglext.get("spec_tokens_details") or {}
    if isinstance(spec, list):
        spec = spec[0] if spec else {}
    row = {
        "step": "coding-1-write",
        "elapsed_s": round(dt, 2),
        "completion_tokens": n,
        "wall_tok_s": None if dt <= 0 or not n else round(n / dt, 3),
        "ms_per_verify": None,
        "spec": spec,
        "text": (msg.get("content") or "")[:400],
    }
    vct = spec.get("spec_verify_ct") if isinstance(spec, dict) else None
    if vct and n:
        row["alpha"] = round(n / float(vct), 3)
        row["ms_per_verify"] = round(dt * 1000.0 / float(vct), 1)
    hist = None
    if isinstance(spec, dict):
        hist = spec.get("spec_correct_drafts_histogram")
    row["correct_drafts_histogram"] = hist
    if hist:
        # histogram[i] = verifies that accepted i drafts (bonus is extra).
        # Cluster at i=0 => reject at first draft / bonus boundary (Q5).
        total = sum(int(x) for x in hist)
        row["q5_frac_zero_draft"] = (
            None if not total else round(int(hist[0]) / total, 3)
        )
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
