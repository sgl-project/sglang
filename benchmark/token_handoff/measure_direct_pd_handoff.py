#!/usr/bin/env python3
"""Exercise token handoff without requiring a rebuilt PD router.

The script submits the same OpenAI request to Prefill and Decode concurrently,
using one explicit bootstrap room.  It then models the experimental router's
stream ownership rule: all Prefill events precede all Decode events, and each
side's ``[DONE]`` sentinel is excluded from the merged payload.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import secrets
import time
import urllib.request


def consume_stream(url: str, payload: dict, started: float, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events = []
    done_count = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done_count += 1
                continue
            if not data:
                continue
            event = json.loads(data)
            choices = event.get("choices") or []
            choice = choices[0] if choices else {}
            text = choice.get("text")
            if text is None:
                text = (choice.get("delta") or {}).get("content", "")
            events.append(
                {
                    "at_ms": (time.perf_counter() - started) * 1000.0,
                    "text": text or "",
                    "finish_reason": choice.get("finish_reason"),
                }
            )
    return {"events": events, "done_count": done_count}


def consume_nonstream(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    choices = body.get("choices") or []
    return {
        "text": choices[0].get("text", "") if choices else "",
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "usage": body.get("usage"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--bootstrap-host", required=True)
    parser.add_argument("--bootstrap-port", type=int, required=True)
    parser.add_argument("--standalone-url")
    parser.add_argument("--prompt-repeat", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output")
    parser.add_argument("--require-match", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    room = secrets.randbelow((1 << 63) - 1) + 1
    payload = {
        "model": "dummy",
        "prompt": "A B C D E F G H " * args.prompt_repeat,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "bootstrap_host": args.bootstrap_host,
        "bootstrap_port": args.bootstrap_port,
        "bootstrap_room": room,
    }

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        prefill_future = executor.submit(
            consume_stream,
            args.prefill_url,
            payload,
            started,
            args.timeout,
        )
        decode_future = executor.submit(
            consume_stream,
            args.decode_url,
            payload,
            started,
            args.timeout,
        )
        prefill = prefill_future.result()
        decode = decode_future.result()

    merged = prefill["events"] + decode["events"]
    merged_text = "".join(event["text"] for event in merged)
    result = {
        "bootstrap_room": room,
        "prefill_event_count": len(prefill["events"]),
        "decode_event_count": len(decode["events"]),
        "prefill_done_count": prefill["done_count"],
        "decode_done_count": decode["done_count"],
        "prefill_texts": [event["text"] for event in prefill["events"]],
        "decode_texts": [event["text"] for event in decode["events"]],
        "merged_text": merged_text,
        "merged_event_times_ms": [event["at_ms"] for event in merged],
        "prefill_event_times_ms": [
            event["at_ms"] for event in prefill["events"]
        ],
        "decode_event_times_ms": [event["at_ms"] for event in decode["events"]],
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }
    if args.standalone_url:
        standalone_payload = dict(payload)
        standalone_payload["stream"] = False
        for key in ("bootstrap_host", "bootstrap_port", "bootstrap_room"):
            standalone_payload.pop(key)
        standalone = consume_nonstream(
            args.standalone_url,
            standalone_payload,
            args.timeout,
        )
        result["standalone"] = standalone
        result["text_matches_standalone"] = merged_text == standalone["text"]

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")
    if args.require_match and not result.get("text_matches_standalone", False):
        raise SystemExit("handoff output does not match standalone output")


if __name__ == "__main__":
    main()
