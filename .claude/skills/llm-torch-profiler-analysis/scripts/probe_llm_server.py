#!/usr/bin/env python3
"""Run a small correctness and latency probe against an LLM server."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request

from profile_common import extract_openai_chat_text

DEFAULT_PROMPTS = [
    "用一句中文介绍上海。",
    "What is 2+2? Answer briefly.",
    "Write one short haiku about GPUs.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send a few short requests to an LLM server and record latency plus "
            "sample outputs."
        )
    )
    parser.add_argument(
        "--framework",
        required=True,
        choices=("sglang", "vllm", "trtllm"),
        help="Serving framework.",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Server base URL, for example http://127.0.0.1:30000.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model id. Auto-discovered for vLLM and TensorRT-LLM when omitted.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=6,
        help="How many probe requests to send.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=48,
        help="Generation length for each request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Optional prompt override. Repeat to add more prompts.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def get_json(url: str, timeout: float) -> Dict[str, Any]:
    req = request.Request(url=url, method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def discover_openai_model(base_url: str, timeout: float) -> str:
    payload = get_json(base_url.rstrip("/") + "/v1/models", timeout=timeout)
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"No models returned by {base_url.rstrip('/')}/v1/models")
    first = data[0]
    if isinstance(first, dict) and first.get("id"):
        return str(first["id"])
    raise RuntimeError(f"Malformed /v1/models payload from {base_url.rstrip('/')}")


def p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def sglang_request(base_url: str, prompt: str, max_tokens: int, timeout: float) -> str:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_tokens,
        },
        "stream": False,
    }
    body = post_json(base_url.rstrip("/") + "/generate", payload, timeout=timeout)
    return str(body.get("text", ""))


def openai_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> Dict[str, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = post_json(
        base_url.rstrip("/") + "/v1/chat/completions",
        payload,
        timeout=timeout,
    )
    text, source = extract_openai_chat_text(body)
    return {"text": text, "source": source}


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    model = args.model
    if args.framework in {"vllm", "trtllm"} and not model:
        model = discover_openai_model(args.url, timeout=args.timeout)

    latencies: List[float] = []
    samples: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for request_idx in range(args.requests):
        prompt = prompts[request_idx % len(prompts)]
        start = time.time()
        try:
            if args.framework == "sglang":
                text = sglang_request(
                    args.url,
                    prompt,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                source = "generate.text"
            else:
                assert model is not None
                result = openai_request(
                    args.url,
                    model,
                    prompt,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                text = result["text"]
                source = result["source"]
            elapsed = time.time() - start
            latencies.append(elapsed)
            samples.append(
                {
                    "prompt": prompt,
                    "latency_s": round(elapsed, 3),
                    "content": text[:240],
                    "source": source,
                    "non_empty": bool(text.strip()),
                }
            )
        except Exception as exc:  # pragma: no cover - runtime probe path
            errors.append({"prompt": prompt, "error": repr(exc)})

    return {
        "framework": args.framework,
        "url": args.url,
        "model": model,
        "requests": args.requests,
        "success": len(samples),
        "errors": len(errors),
        "all_non_empty": (
            all(sample["non_empty"] for sample in samples) if samples else False
        ),
        "avg_latency_s": round(statistics.mean(latencies), 3) if latencies else None,
        "p95_latency_s": round(p95(latencies), 3) if latencies else None,
        "samples": samples[:3],
        "error_samples": errors[:3],
    }


def main() -> int:
    args = parse_args()
    summary = run_probe(args)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
