#!/usr/bin/env python3
"""Smokes against a live DeepSeek-V4.1-Flash server."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def req(url: str, payload: dict[str, Any] | None = None, timeout: int = 3600) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    r = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        body = resp.read()
        if not body:
            return {"status": resp.status}
        return json.loads(body)


def wait_ready(base: str, timeout_s: int) -> None:
    health = base.rstrip("/") + "/health"
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(health, timeout=5)
            print(f"READY after {time.time() - t0:.0f}s", flush=True)
            return
        except Exception as e:
            last = e
            time.sleep(5)
    raise SystemExit(f"server not ready after {timeout_s}s: {last}")


def chat(base: str, messages: list[dict], extra: dict | None = None, timeout: int = 3600) -> dict:
    payload = {
        "model": "default",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 64,
        **(extra or {}),
    }
    return req(base.rstrip("/") + "/v1/chat/completions", payload, timeout=timeout)


def generate_ids(base: str, n_tokens: int, timeout: int) -> dict:
    # pad with a cheap repeated id; BOS=0 is fine for a fill-memory prefill
    payload = {
        "input_ids": [1] * n_tokens,
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0,
            "ignore_eos": True,
        },
    }
    return req(base.rstrip("/") + "/generate", payload, timeout=timeout)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:11435")
    p.add_argument("--wait", type=int, default=0, help="seconds to wait for /health first")
    p.add_argument(
        "--steps",
        default="323,tools,effort,prefill8k,prefill32k,prefill250k",
    )
    args = p.parse_args()
    if args.wait:
        wait_ready(args.base, args.wait)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    failed = 0

    if "323" in steps:
        print("=== greedy 17*19 ===", flush=True)
        t0 = time.time()
        out = chat(
            args.base,
            [{"role": "user", "content": "What is 17*19? Reply with only the integer."}],
            extra={"max_tokens": 32},
        )
        text = out["choices"][0]["message"].get("content") or ""
        print(f"elapsed={time.time()-t0:.1f}s text={text!r}", flush=True)
        if "323" in text.replace(",", "").replace(" ", ""):
            print("PASS 17*19", flush=True)
        else:
            print("FAIL 17*19 expected 323", flush=True)
            failed += 1

    if "tools" in steps:
        print("=== tool call ===", flush=True)
        t0 = time.time()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        out = chat(
            args.base,
            [{"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}],
            extra={"tools": tools, "max_tokens": 256, "tool_choice": "auto"},
        )
        msg = out["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        print(f"elapsed={time.time()-t0:.1f}s tool_calls={calls!r} content={msg.get('content')!r}", flush=True)
        names = [c.get("function", {}).get("name") for c in calls]
        if "get_weather" in names:
            print("PASS tool-call", flush=True)
        else:
            print("FAIL tool-call", flush=True)
            failed += 1

    if "effort" in steps:
        print("=== reasoning_effort=0.25 (budget 25) ===", flush=True)
        t0 = time.time()
        out = chat(
            args.base,
            [{"role": "user", "content": "Say the word ping and stop."}],
            # The API takes a tier or a float in [0, 0.99]; 0.25 is budget 25.
            extra={"max_tokens": 128, "reasoning_effort": 0.25},
        )
        msg = out["choices"][0]["message"]
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        print(
            f"elapsed={time.time()-t0:.1f}s content={msg.get('content')!r} reasoning={reasoning!r}",
            flush=True,
        )
        print("PASS reasoning_effort round-trip (request accepted)", flush=True)

    for name, ntok, timeout in (
        ("prefill8k", 8192, 1800),
        ("prefill32k", 32768, 7200),
        ("prefill250k", 250000, 86400),
    ):
        if name not in steps:
            continue
        print(f"=== {name} ({ntok} tokens, max_new=1) ===", flush=True)
        t0 = time.time()
        try:
            out = generate_ids(args.base, ntok, timeout=timeout)
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
            failed += 1
            continue
        dt = time.time() - t0
        print(f"elapsed={dt:.1f}s keys={list(out)[:8]} meta={ {k: out.get(k) for k in ('meta_info','text') if k in out} }", flush=True)
        print(f"PASS {name}", flush=True)

    print(f"SMOKE failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
