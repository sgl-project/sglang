#!/usr/bin/env python3
"""
Manual test: NEXTN decode near the context boundary (FA3).

Exercises long decode toward the server context limit with overlapping
background prefills. Originally added to reproduce FA3 illegal memory access
when ``draft_extend`` publishes ``seq_lens + num_draft_tokens`` past a
page table sized to ``context_len`` only; still useful as a boundary stress
script after the fix.

Workload:
  1) One long decode (ignore_eos) climbing toward the effective context wall
  2) Background worker while decode is alive: 1 long-prefill thread + 2 short-prefill threads

Context-length tuning (automatic)
-------------------------------
On startup the script queries the running server:

- ``GET /server_info`` for ``context_length`` and ``speculative_num_draft_tokens``
- ``GET /v1/models`` for ``max_model_len`` when ``context_length`` is unset

It then **automatically sizes the long-decode request** so generation climbs
toward the server's context limit — no manual ``--context-length`` tuning:

- ``script_context = server_context_length``
- ``long_prompt_tokens = int(script_context * 0.9)`` — most of the window is prefilled prompt
- ``long_max_tokens = script_context - long_prompt_tokens + 200`` — first attempt asks for
  slightly *more* generate tokens than the nominal remainder, so the server often returns a
  4xx context overflow
- ``context_probe_step = speculative_num_draft_tokens // 2`` — on overflow, reduce
  ``max_tokens`` by this step and retry until the request is accepted

That step-down search finds the **largest generate budget whose prompt+gen total fits
the server wall** — i.e. long decode runs with total context as close as possible to
``context_length``, which is where speculative ``draft_extend`` boundary bugs show up.

Example:

  python3 test/manual/spec/test_nextn_decode_context_boundary.py \\
    --base-url http://127.0.0.1:8000/v1/chat/completions \\
    --tokenizer-path /path/to/model

Success signal: server dies / client sees connection or 5xx errors near the wall.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from urllib.parse import urlparse

import requests
from transformers import AutoTokenizer

TOKEN_SLACK = 8
SERVER_QUERY_TIMEOUT_SEC = 30.0
TIMEOUT_SEC = 3600.0
RESTART_DELAY_SEC = 2.0
ROUND_TIMEOUT_SEC = 7200.0
BG_PREFILL_INTERVAL_SEC = 1.0
SHORT_PREFILL_PROMPT_TOKENS = 32
PREFILL_MAX_TOKENS = 16

LONG_DECODE_INSTRUCTION = """
FINAL LONG-DECODE TASK:
Generate a very long deterministic continuation. Do not summarize. Do not stop early.
Print numbered diagnostic lines until max_tokens is reached:
line 000001 nextn context_wall
line 000002 nextn context_wall
Continue with increasing line numbers.
"""


def server_root(base_url: str) -> str:
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def fetch_server_context_settings(base_url: str) -> tuple[int, int, int]:
    """Return (server_context_length, script_context_length, context_probe_step)."""
    root = server_root(base_url)
    info_resp = requests.get(f"{root}/server_info", timeout=SERVER_QUERY_TIMEOUT_SEC)
    info_resp.raise_for_status()
    info = info_resp.json()

    server_context = info.get("context_length")
    if server_context is None:
        models_resp = requests.get(
            f"{root}/v1/models", timeout=SERVER_QUERY_TIMEOUT_SEC
        )
        models_resp.raise_for_status()
        models = models_resp.json()
        data = models.get("data") or []
        if not data:
            raise RuntimeError("/v1/models returned no models")
        server_context = data[0].get("max_model_len")
    server_context = int(server_context)

    num_draft = info.get("speculative_num_draft_tokens")
    if not num_draft:
        raise RuntimeError(
            "server speculative_num_draft_tokens is missing or zero; "
            "launch with speculative decoding enabled"
        )
    num_draft = int(num_draft)

    script_context = server_context
    probe_step = max(1, num_draft // 2)
    return server_context, script_context, probe_step


def content_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_text(tokenizer, target_tokens: int, label: str) -> str:
    if target_tokens <= 0:
        return ""
    seed = f"\n[{label}] nextn-decode-context-boundary.\n"
    seed_ids = tokenizer.encode(seed, add_special_tokens=False) or tokenizer.encode(
        " x", add_special_tokens=False
    )
    ids = (seed_ids * ((target_tokens // len(seed_ids)) + 2))[:target_tokens]
    text = tokenizer.decode(ids)
    for _ in range(16):
        n = content_tokens(tokenizer, text)
        if n == target_tokens:
            return text
        if n > target_tokens:
            text = tokenizer.decode(
                tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
            )
        else:
            text += " x" * (target_tokens - n)
    return text


def count_token_ids(value) -> int:
    if value is None:
        return 0
    if isinstance(value, dict) or hasattr(value, "get"):
        try:
            input_ids = value.get("input_ids")
            if input_ids is not None:
                return count_token_ids(input_ids)
        except Exception:
            pass
    if hasattr(value, "input_ids") and not isinstance(value, (list, tuple, str)):
        try:
            return count_token_ids(value.input_ids)
        except Exception:
            pass
    if hasattr(value, "shape"):
        shape = tuple(value.shape)
        if shape:
            return int(shape[-1])
    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        first = value[0]
        if isinstance(first, (list, tuple)) or hasattr(first, "shape"):
            return count_token_ids(first)
        return len(value)
    if isinstance(value, str):
        return 0
    return len(value)


def prompt_tokens(tokenizer, messages) -> int:
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        n = count_token_ids(ids)
        if n > 0:
            return n
    except Exception:
        pass
    raw = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)
    return content_tokens(tokenizer, raw)


def grow_prompt(tokenizer, target_tokens: int, label: str) -> tuple[list, int]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a deterministic assistant for a NEXTN decode "
                "context-boundary manual test."
            ),
        }
    ]
    low, high = 0, target_tokens + 2048
    best, best_n = messages, prompt_tokens(tokenizer, messages)
    while low <= high:
        mid = (low + high) // 2
        trial = messages + [
            {"role": "user", "content": make_text(tokenizer, mid, label)}
        ]
        n = prompt_tokens(tokenizer, trial)
        if n <= target_tokens:
            best, best_n = trial, n
            low = mid + 1
        else:
            high = mid - 1
    if best and best[-1]["role"] == "user":
        best[-1]["content"] = LONG_DECODE_INSTRUCTION + "\n" + best[-1]["content"]
    else:
        best = best + [{"role": "user", "content": LONG_DECODE_INSTRUCTION}]
    return best, prompt_tokens(tokenizer, best)


class LongDecodeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.alive = threading.Event()

    def start(self):
        self.alive.set()

    def finish(self):
        self.alive.clear()

    def wait_alive(self, timeout: float) -> bool:
        return self.alive.wait(timeout=timeout)


class CrashFlag:
    def __init__(self):
        self.lock = threading.Lock()
        self.reason = None

    def set(self, reason: str):
        with self.lock:
            if self.reason is None:
                self.reason = reason

    def get(self):
        with self.lock:
            return self.reason


def is_context_overflow(status: int, body: str) -> bool:
    if status not in (400, 413, 422):
        return False
    lowered = (body or "").lower()
    return any(
        n in lowered
        for n in ("context", "max_tokens", "too long", "exceed", "overflow")
    )


def parse_stream(resp, tokenizer, *, log_prefix: str | None = None):
    pieces: list[str] = []
    chunks = 0
    finish_reason = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        chunks += 1
        try:
            data = json.loads(payload)
            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta", {}) or {}
            for key in ("content", "reasoning_content"):
                part = delta.get(key)
                if part:
                    pieces.append(part)
        except Exception:
            pass
        if log_prefix and chunks % 256 == 0:
            full = content_tokens(tokenizer, "".join(pieces))
            print(f"{log_prefix}: gen≈{full} chunks={chunks}", flush=True)
    return "".join(pieces), finish_reason


def send_chat(
    args,
    messages,
    label: str,
    *,
    max_tokens: int,
    ignore_eos: bool = False,
    tokenizer=None,
):
    url = args.base_url
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    started = time.time()
    try:
        with requests.post(url, json=payload, stream=True, timeout=TIMEOUT_SEC) as resp:
            print(
                f"{label}: status={resp.status_code} max_tokens={max_tokens} "
                f"t={time.time() - started:.2f}s",
                flush=True,
            )
            if resp.status_code >= 400:
                body = resp.text[:1500]
                print(f"{label}: body={body[:800]}", flush=True)
                if resp.status_code >= 500:
                    return None, f"http_{resp.status_code}", body
                return "", f"http_{resp.status_code}", body
            text, finish_reason = parse_stream(
                resp,
                tokenizer,
                log_prefix=label if ignore_eos else None,
            )
            print(
                f"{label}: done chars={len(text)} finish_reason={finish_reason} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
            return text, None, ""
    except requests.exceptions.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}", ""


def long_decode_worker(
    args, tokenizer, crash: CrashFlag, stop: threading.Event, state: LongDecodeState
):
    messages, prompt_n = grow_prompt(tokenizer, args.long_prompt_tokens, "long-prompt")
    ceiling = max(1, args.long_max_tokens - TOKEN_SLACK)
    probe_max = ceiling
    step = max(1, args.context_probe_step)
    attempt = 0

    print(
        f"long-decode: prompt={prompt_n} context={args.context_length} "
        f"long_prompt={args.long_prompt_tokens} ceiling={ceiling} step={step}",
        flush=True,
    )

    while not stop.is_set() and crash.get() is None:
        attempt += 1
        max_tokens = max(1, min(probe_max, ceiling))
        state.start()
        print(f"long-decode: attempt={attempt} max_tokens={max_tokens}", flush=True)

        text, err, body = send_chat(
            args,
            messages,
            f"long-decode-a{attempt}",
            max_tokens=max_tokens,
            ignore_eos=True,
            tokenizer=tokenizer,
        )
        state.finish()

        status = None
        if err and err.startswith("http_"):
            try:
                status = int(err.split("_", 1)[1])
            except ValueError:
                status = None

        if status is not None and is_context_overflow(status, body):
            # Step down max_tokens to probe the server's effective context-length.
            probe_max = max(1, max_tokens - step)
            print(
                f"long-decode: CONTEXT_OVERFLOW status={status} "
                f"-> probe down max_tokens by {step} to {probe_max} "
                f"(effective full≈{prompt_n + probe_max})",
                flush=True,
            )
            continue

        if err and text is None:
            crash.set(f"long-decode: {err}")
            stop.set()
            return

        gen = content_tokens(tokenizer, text or "")
        print(
            f"long-decode: attempt={attempt} finished gen≈{gen} full≈{prompt_n + gen}",
            flush=True,
        )
        if stop.is_set() or crash.get() is not None:
            return
        time.sleep(RESTART_DELAY_SEC)


def _prefill_messages(tokenizer, prompt_tokens_n: int, label: str) -> tuple[list, int]:
    messages = [
        {"role": "user", "content": make_text(tokenizer, prompt_tokens_n, label)}
    ]
    return messages, prompt_tokens(tokenizer, messages)


def _background_prefill_loop(
    args,
    tokenizer,
    crash: CrashFlag,
    stop: threading.Event,
    state: LongDecodeState,
    *,
    name: str,
    prompt_tokens_n: int,
):
    seq = 0
    while not stop.is_set() and crash.get() is None:
        if not state.wait_alive(timeout=1.0):
            continue
        seq += 1
        messages, n = _prefill_messages(tokenizer, prompt_tokens_n, name)
        text, err, _ = send_chat(
            args,
            messages,
            f"{name}-{seq}-p{n}",
            max_tokens=PREFILL_MAX_TOKENS,
            tokenizer=tokenizer,
        )
        if err and text is None:
            crash.set(f"{name}: {err}")
            stop.set()
            return
        if stop.wait(BG_PREFILL_INTERVAL_SEC):
            return


def background_worker(
    args, tokenizer, crash: CrashFlag, stop: threading.Event, state: LongDecodeState
):
    long_tokens = max(1, args.long_prompt_tokens // 2)
    print(
        f"background: 1x long_prefill≈{long_tokens}tok "
        f"2x short_prefill≈{SHORT_PREFILL_PROMPT_TOKENS}tok",
        flush=True,
    )
    threads = [
        threading.Thread(
            target=_background_prefill_loop,
            args=(args, tokenizer, crash, stop, state),
            kwargs={
                "name": "bg-long-prefill",
                "prompt_tokens_n": long_tokens,
            },
            daemon=True,
            name="bg-long-prefill",
        ),
    ]
    for i in range(2):
        threads.append(
            threading.Thread(
                target=_background_prefill_loop,
                args=(args, tokenizer, crash, stop, state),
                kwargs={
                    "name": f"bg-short-prefill-{i}",
                    "prompt_tokens_n": SHORT_PREFILL_PROMPT_TOKENS,
                },
                daemon=True,
                name=f"bg-short-prefill-{i}",
            )
        )
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def derive_lengths(context_length: int) -> tuple[int, int]:
    # Size long decode so prompt+gen total is driven toward context_length via
    # overflow step-down in long_decode_worker (see module docstring).
    long_prompt_tokens = max(1, int(context_length * 0.9))
    long_max_tokens = max(1, context_length - long_prompt_tokens + 200)
    return long_prompt_tokens, long_max_tokens


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://127.0.0.1:8000/v1/chat/completions"),
    )
    p.add_argument(
        "--model",
        default=os.getenv("MODEL", "/data/models/Qwen/Qwen3.5-397B-A17B-FP8"),
    )
    p.add_argument(
        "--tokenizer-path",
        default=os.getenv(
            "TOKENIZER_PATH",
            os.getenv("MODEL_PATH", "/data/models/Qwen/Qwen3.5-397B-A17B-FP8"),
        ),
    )
    args = p.parse_args()

    try:
        (
            args.server_context_length,
            args.context_length,
            args.context_probe_step,
        ) = fetch_server_context_settings(args.base_url)
    except (requests.RequestException, RuntimeError, TypeError, ValueError) as exc:
        print(f"failed to read server context settings: {exc}", file=sys.stderr)
        return 2

    args.long_prompt_tokens, args.long_max_tokens = derive_lengths(args.context_length)
    if args.context_probe_step >= args.long_max_tokens:
        print(
            "derived context-probe-step must be < script long_max_tokens",
            file=sys.stderr,
        )
        return 2

    print(
        f"config: server_context={args.server_context_length} "
        f"script_context={args.context_length} "
        f"long_prompt={args.long_prompt_tokens} long_max={args.long_max_tokens} "
        f"step={args.context_probe_step} (num_draft//2)",
        flush=True,
    )
    print(f"loading tokenizer: {args.tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    stop = threading.Event()
    crash = CrashFlag()
    state = LongDecodeState()
    threads = [
        threading.Thread(
            target=long_decode_worker,
            args=(args, tokenizer, crash, stop, state),
            daemon=True,
            name="long-decode",
        ),
        threading.Thread(
            target=background_worker,
            args=(args, tokenizer, crash, stop, state),
            daemon=True,
            name="background",
        ),
    ]
    for t in threads:
        t.start()

    deadline = time.time() + ROUND_TIMEOUT_SEC
    while time.time() < deadline:
        reason = crash.get()
        if reason is not None:
            stop.set()
            print(f"CRASH / server unreachable: {reason}", flush=True)
            return 1
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.5)

    stop.set()
    for t in threads:
        t.join(timeout=5)
    reason = crash.get()
    if reason is not None:
        print(f"CRASH / server unreachable: {reason}", flush=True)
        return 1
    print("finished without observing a crash", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
