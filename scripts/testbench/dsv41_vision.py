#!/usr/bin/env python3
"""DSV4.1 vision release checks.

Unit (no server): checkpoint names land on the CPU tower, and the streaming
fp16 GPU path matches that tower. Live (server on :11435): text 17*19 is 323,
a red image answers red, a blue image answers blue.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _req(url: str, payload: dict | None = None, timeout: int = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        body = resp.read()
        return json.loads(body) if body else {"status": resp.status}


def wait_ready(base: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base.rstrip("/") + "/health", timeout=5)
            return
        except Exception as exc:
            last = exc
            time.sleep(5)
    raise SystemExit(f"server not ready after {timeout_s}s: {last}")


def chat(base: str, messages: list, max_tokens: int = 64) -> tuple[str, float]:
    t0 = time.time()
    out = _req(
        base.rstrip("/") + "/v1/chat/completions",
        {
            "model": "default",
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        },
    )
    text = out["choices"][0]["message"].get("content") or ""
    return text, time.time() - t0


def _png(color: tuple[int, int, int]) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _image_turn(base: str, color: tuple[int, int, int], word: str) -> tuple[bool, str]:
    text, elapsed = chat(
        base,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_png(color)}"},
                    },
                    {
                        "type": "text",
                        "text": "What is the dominant color of this image? Reply with one word.",
                    },
                ],
            }
        ],
    )
    ok = bool(text.strip()) and word.lower() in text.lower()
    print(f"{'PASS' if ok else 'FAIL'} image {word}: {elapsed:.1f}s text={text!r}")
    return ok, text


def run_unit() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "test/registered/unit/models/test_dsv41_flash_loader.py::TestDsv41CpuVisionLoad",
        "-q",
        "--tb=short",
    ]
    print("=== unit TestDsv41CpuVisionLoad ===", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def run_live(base: str) -> int:
    failed = 0
    print("=== text 17*19 ===", flush=True)
    text, elapsed = chat(
        base,
        [{"role": "user", "content": "What is 17*19? Reply with only the integer."}],
        max_tokens=32,
    )
    ok = "323" in text.replace(",", "").replace(" ", "")
    print(f"{'PASS' if ok else 'FAIL'} 17*19: {elapsed:.1f}s text={text!r}")
    failed += not ok
    failed += not _image_turn(base, (220, 20, 20), "red")[0]
    failed += not _image_turn(base, (20, 40, 220), "blue")[0]
    return failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:11435")
    parser.add_argument("--wait", type=int, default=0)
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--live-only", action="store_true")
    args = parser.parse_args()
    if args.wait:
        wait_ready(args.base, args.wait)
    failed = 0
    if not args.live_only:
        failed += run_unit() != 0
    if not args.unit_only:
        try:
            failed += run_live(args.base)
        except urllib.error.URLError as exc:
            print(f"FAIL live: {exc}")
            failed += 1
    print(f"dsv41 vision failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
