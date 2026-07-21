#!/usr/bin/env python3
"""DSv4 DCP numerical-equivalence regression script.

Usage:
    1. Launch a baseline sglang server with --dcp-size 1 on :30000.
    2. Launch a candidate sglang server with --dcp-size 2 (and
       SGLANG_DSV4_ENABLE_DCP=1) on :30001.
    3. Run:
         python scripts/playground/dcp_equivalence_check.py \\
             --baseline-url http://127.0.0.1:30000 \\
             --candidate-url http://127.0.0.1:30001 \\
             --model-path /path/to/dsv4 \\
             --num-prompts 8 --max-tokens 64

The script issues the same prompt set to both endpoints with
``temperature=0, top_p=1, top_k=1`` and compares:
  1. exact decoded text
  2. completion token id sequence
  3. per-token chosen-token logprob (max abs delta)

Exits non-zero on any divergence so it can be wired into CI.

This script intentionally has no torch / sglang internal dependency; it only
needs ``requests`` so it can run from any Python 3.8+ environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

DEFAULT_PROMPTS: List[str] = [
    # Short / boilerplate
    "Hello, world!",
    # English Q&A
    "Question: What is the capital of France?\nAnswer:",
    # Code completion
    "def fibonacci(n):\n    if n < 2:\n        return n\n    ",
    # Chinese
    "请用一句话总结相对论的核心思想：",
    # Long context (~256 tokens)
    "The following is a verbose technical specification of a hypothetical "
    "machine learning system. Please continue writing in the same style. "
    "The system shall implement decode context parallelism over a sharded "
    "key-value cache. " * 8 + "\n\nContinuing from the above:\n",
    # Multi-turn chat-style
    "User: Explain quantum entanglement in one paragraph.\nAssistant:",
    # Math
    "Compute step by step: 17 * 23 + 41.",
    # Edge: empty-ish (single token prompt)
    "1, 2, 3,",
]


@dataclass
class TestCase:
    name: str
    endpoint: str
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None


@dataclass
class CompletionResult:
    text: str
    token_ids: List[int]
    chosen_logprobs: List[float]  # logprob of each chosen token
    raw: Dict[str, Any]

    def fingerprint(self) -> str:
        # Stable repr for diffing.
        return json.dumps(
            {"token_ids": self.token_ids, "text": self.text, "raw": self.raw},
            ensure_ascii=False,
            sort_keys=True,
        )


def stable_token_id(token: str) -> int:
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")


def build_test_cases(num_prompts: int) -> List[TestCase]:
    cases = [
        TestCase(name=f"completion_{idx}", endpoint="completion", prompt=prompt)
        for idx, prompt in enumerate(DEFAULT_PROMPTS[:num_prompts])
    ]
    cases.extend(
        [
            TestCase(
                name="completion_stream",
                endpoint="completion",
                prompt="Stream a short deterministic checklist about KV cache correctness.",
                stream=True,
            ),
            TestCase(
                name="chat_reasoning_cn",
                endpoint="chat",
                messages=[
                    {
                        "role": "user",
                        "content": "请用三句话解释在线 c128 压缩和 DCP 等价性验证的关系。",
                    }
                ],
            ),
            TestCase(
                name="chat_multi_turn",
                endpoint="chat",
                messages=[
                    {"role": "user", "content": "Remember the word radix."},
                    {"role": "assistant", "content": "I will remember the word radix."},
                    {"role": "user", "content": "What word did I ask you to remember?"},
                ],
            ),
            TestCase(
                name="chat_stream",
                endpoint="chat",
                messages=[
                    {
                        "role": "user",
                        "content": "List two invariants for decode context parallel KV sharding.",
                    }
                ],
                stream=True,
            ),
            TestCase(
                name="chat_tool_call",
                endpoint="chat",
                messages=[
                    {
                        "role": "user",
                        "content": "Use the tool to report weather for Beijing in celsius.",
                    }
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                    "unit": {"type": "string", "enum": ["celsius"]},
                                },
                                "required": ["city", "unit"],
                            },
                        },
                    }
                ],
            ),
        ]
    )
    return cases


def iter_sse_json(resp: requests.Response):
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        yield json.loads(payload)


def call_completion(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    stream: bool = False,
) -> CompletionResult:
    """Call the OpenAI-compatible /v1/completions endpoint."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        # sglang accepts top_k via extra body; ignore if unsupported
        "top_k": 1,
        "logprobs": 1,
        "echo": False,
        "stream": stream,
        "seed": 42,
    }
    resp = requests.post(
        f"{url.rstrip('/')}/v1/completions",
        json=payload,
        timeout=timeout,
        stream=stream,
    )
    resp.raise_for_status()
    if stream:
        chunks = list(iter_sse_json(resp))
        text = "".join(chunk["choices"][0].get("text", "") for chunk in chunks)
        return CompletionResult(text=text, token_ids=[], chosen_logprobs=[], raw={})

    data = resp.json()
    choice = data["choices"][0]
    text = choice["text"]
    lp = choice.get("logprobs") or {}
    token_logprobs = lp.get("token_logprobs") or []
    # Token ids are not in OpenAI logprobs; fall back to stable per-token hashes.
    tokens = lp.get("tokens") or []
    token_ids = [stable_token_id(t) for t in tokens]
    chosen_logprobs = [float(x) if x is not None else 0.0 for x in token_logprobs]
    return CompletionResult(
        text=text, token_ids=token_ids, chosen_logprobs=chosen_logprobs, raw={}
    )


def call_chat(
    url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    timeout: float,
    stream: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> CompletionResult:
    """Call the OpenAI-compatible /v1/chat/completions endpoint."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "stream": stream,
        "seed": 42,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    resp = requests.post(
        f"{url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=timeout,
        stream=stream,
    )
    resp.raise_for_status()
    if stream:
        chunks = list(iter_sse_json(resp))
        text = "".join(
            chunk["choices"][0].get("delta", {}).get("content") or ""
            for chunk in chunks
        )
        return CompletionResult(text=text, token_ids=[], chosen_logprobs=[], raw={})

    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    tool_calls = message.get("tool_calls") or []
    raw = {"reasoning": reasoning, "tool_calls": tool_calls}
    return CompletionResult(
        text=content,
        token_ids=[],
        chosen_logprobs=[],
        raw=raw,
    )


def run_case(url: str, model: str, case: TestCase, max_tokens: int, timeout: float):
    if case.endpoint == "completion":
        assert case.prompt is not None
        return call_completion(
            url, model, case.prompt, max_tokens, timeout, stream=case.stream
        )
    if case.endpoint == "chat":
        assert case.messages is not None
        return call_chat(
            url,
            model,
            case.messages,
            max_tokens,
            timeout,
            stream=case.stream,
            tools=case.tools,
        )
    raise ValueError(f"unsupported endpoint {case.endpoint!r}")


def run_cases(
    url: str,
    model: str,
    cases: List[TestCase],
    max_tokens: int,
    timeout: float,
    concurrency: int,
) -> Dict[str, CompletionResult]:
    if concurrency <= 1:
        return {
            case.name: run_case(url, model, case, max_tokens, timeout)
            for case in cases
        }

    results: Dict[str, CompletionResult] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(run_case, url, model, case, max_tokens, timeout): case.name
            for case in cases
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def save_results(path: str, results: Dict[str, CompletionResult]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {name: result.__dict__ for name, result in results.items()},
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def load_results(path: str) -> Dict[str, CompletionResult]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        name: CompletionResult(
            text=item.get("text", ""),
            token_ids=item.get("token_ids", []),
            chosen_logprobs=item.get("chosen_logprobs", []),
            raw=item.get("raw", {}),
        )
        for name, item in data.items()
    }


def compare_one(
    name: str,
    base: CompletionResult,
    cand: CompletionResult,
    logprob_atol: float,
) -> bool:
    """Return True if results match within tolerance."""
    ok = True
    if base.text != cand.text:
        ok = False
        print(f"[{name}] TEXT MISMATCH")
        print(f"  baseline:  {base.text!r}")
        print(f"  candidate: {cand.text!r}")
    if base.raw != cand.raw:
        ok = False
        print(f"[{name}] RAW MISMATCH")
        print(f"  baseline:  {json.dumps(base.raw, ensure_ascii=False, sort_keys=True)}")
        print(f"  candidate: {json.dumps(cand.raw, ensure_ascii=False, sort_keys=True)}")
    if base.token_ids != cand.token_ids:
        ok = False
        # Find first diverging position
        n = min(len(base.token_ids), len(cand.token_ids))
        first_div = next(
            (i for i in range(n) if base.token_ids[i] != cand.token_ids[i]),
            n,
        )
        print(
            f"[{name}] TOKEN MISMATCH at position {first_div} "
            f"(baseline_len={len(base.token_ids)}, "
            f"candidate_len={len(cand.token_ids)})"
        )
    # Logprob comparison only meaningful when token ids match
    if base.token_ids == cand.token_ids and base.chosen_logprobs and cand.chosen_logprobs:
        deltas = [
            abs(b - c) for b, c in zip(base.chosen_logprobs, cand.chosen_logprobs)
        ]
        max_delta = max(deltas) if deltas else 0.0
        if max_delta > logprob_atol:
            ok = False
            print(
                f"[{name}] LOGPROB DELTA exceeds atol={logprob_atol}: "
                f"max={max_delta:.6f}"
            )
        else:
            print(
                f"[{name}] OK (max_logprob_delta={max_delta:.6f}, "
                f"len={len(base.token_ids)})"
            )
    elif ok:
        print(f"[{name}] OK")
    return ok


def wait_for_health(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{url.rstrip('/')}/health", timeout=5.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(2.0)
    raise RuntimeError(f"server {url} not healthy within {timeout}s: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-url", help="dcp_size=1 server URL")
    ap.add_argument("--candidate-url", help="dcp_size>1 server URL")
    ap.add_argument("--capture-url", help="capture results from one endpoint and exit")
    ap.add_argument("--capture-output", help="JSON output path for --capture-url")
    ap.add_argument("--baseline-results", help="saved baseline JSON from --capture-url")
    ap.add_argument(
        "--model-path",
        required=True,
        help="model path / name as registered with the sglang server",
    )
    ap.add_argument("--num-prompts", type=int, default=len(DEFAULT_PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument(
        "--logprob-atol",
        type=float,
        default=1e-2,
        help="Max abs delta of chosen-token logprob between baseline/candidate",
    )
    ap.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional path to a newline-delimited prompts file",
    )
    ap.add_argument(
        "--skip-health-check", action="store_true", help="skip /health probe"
    )
    args = ap.parse_args()

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            cases = [
                TestCase(name=f"prompt_file_{idx}", endpoint="completion", prompt=line.rstrip("\n"))
                for idx, line in enumerate(f)
                if line.strip()
            ]
    else:
        cases = build_test_cases(args.num_prompts)

    if not cases:
        print("No test cases to evaluate.", file=sys.stderr)
        return 2

    urls_to_check = []
    if args.capture_url:
        urls_to_check.append(args.capture_url)
    else:
        if args.baseline_url:
            urls_to_check.append(args.baseline_url)
        if args.candidate_url:
            urls_to_check.append(args.candidate_url)
    if not args.skip_health_check:
        for url in urls_to_check:
            print(f"Waiting for {url} ...")
            wait_for_health(url)

    if args.capture_url:
        if not args.capture_output:
            print("--capture-output is required with --capture-url", file=sys.stderr)
            return 2
        results = run_cases(
            args.capture_url,
            args.model_path,
            cases,
            args.max_tokens,
            args.timeout,
            args.concurrency,
        )
        save_results(args.capture_output, results)
        print(f"Captured {len(results)} cases to {args.capture_output}")
        return 0

    if args.baseline_results:
        base_results = load_results(args.baseline_results)
    elif args.baseline_url:
        base_results = run_cases(
            args.baseline_url,
            args.model_path,
            cases,
            args.max_tokens,
            args.timeout,
            args.concurrency,
        )
    else:
        print("Either --baseline-url or --baseline-results is required", file=sys.stderr)
        return 2

    if not args.candidate_url:
        print("--candidate-url is required for comparison", file=sys.stderr)
        return 2

    cand_results = run_cases(
        args.candidate_url,
        args.model_path,
        cases,
        args.max_tokens,
        args.timeout,
        args.concurrency,
    )

    failures = 0
    for case in cases:
        try:
            base = base_results[case.name]
            cand = cand_results[case.name]
        except KeyError as e:
            failures += 1
            print(f"[{case.name}] MISSING RESULT: {e}")
            continue
        if not compare_one(case.name, base, cand, args.logprob_atol):
            failures += 1

    print()
    print(f"Total cases: {len(cases)}, failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
