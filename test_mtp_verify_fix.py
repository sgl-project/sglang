#!/usr/bin/env python3
"""Discriminating test for MTP/NEXTN verify-state-layout fix on NPU.

The degraded-MTP symptom was a wrong FIRST token (verify output corrupted by a
transposed base recurrent state). These prompts distinguish the fix: each has a
deterministic, well-known continuation, and the first token is the tell.
"""
import requests
import sys
import time

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8898"

CASES = [
    # (name, prompt, must_contain_substrings, must_not_contain_substrings)
    (
        "Eiffel Tower",
        "Q: How many floors does the Eiffel Tower have?\nA:",
        ["7", "199"],
        [],
    ),
    (
        "Louvre",
        "Q: How many glass panes does the Louvre pyramid have?\nA:",
        ["199", "673"],
        [],
    ),
    (
        "Math 1.2.3.",
        "Q: Write the first three positive integers, one per line.\nA:",
        ["1", "2", "3"],
        [],
    ),
    (
        "France capital",
        "Q: What is the capital of France?\nA:",
        ["paris"],
        [],
    ),
    (
        "Australia capital",
        "Q: What is the capital of Australia?\nA:",
        ["canberra"],
        [],
    ),
    (
        "Paris location",
        "Q: In what country is Paris located?\nA:",
        ["france"],
        [],
    ),
    (
        "2+2",
        "Q: What is 2 + 2?\nA:",
        ["4"],
        [],
    ),
    (
        "Code gen",
        "def factorial(n):\n    ",
        ["return"],
        [],
    ),
    (
        "Chinese capital",
        "问：中国的首都是哪里？\n答：",
        ["北京"],
        [],
    ),
]


def main():
    # wait for health
    for _ in range(600):
        try:
            if requests.get(f"{BASE}/health", timeout=3).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("server never became healthy")
        sys.exit(2)

    passed = 0
    for name, prompt, wants, not_wants in CASES:
        try:
            r = requests.post(
                f"{BASE}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 300,
                        "ignore_eos": False,
                    },
                },
                timeout=60,
            )
            if r.status_code != 200:
                print(f"[{name:20s}] HTTP {r.status_code}: {r.text[:120]}")
                continue
            out = r.json()["text"]
            low = out.lower()
            hit = [w for w in wants if w.lower() in low]
            bad = [w for w in not_wants if w.lower() in low]
            ok = bool(hit) and not bad
            if ok:
                passed += 1
            print(f"[{name:20s}] {'PASS' if ok else 'FAIL'}  want={wants} hit={hit}")
            print(f"    {out[:90]!r}")
        except Exception as e:
            print(f"[{name:20s}] ERROR {type(e).__name__}: {e}")

    print(f"\nTOTAL {passed}/{len(CASES)} passed")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
