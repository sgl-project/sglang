"""Reproduce the /v1/systemone 500 under a breakable prefill CUDA graph.

A Clef checkpoint is served in embedding mode with one complete prefill per
request. Its outer tail -- the joint schema head -- runs eagerly on the
hidden states that the graph produces, and it reads `ForwardBatch
.decision_layouts` to score each request. The breakable prefill graph builds
its *static* ForwardBatch field by field, and before this fix it dropped
`decision_layouts`, so the head raised

    the joint schema head did not score this request

and the request came back as HTTP 500.

How to run (replace CLEF_CHECKPOINT with a local Clef / Clef-flash path):

    # 1. serve with the prefill graph selected explicitly -- the model hook
    #    keeps both CUDA graph phases off for Clef unless the caller picks a
    #    prefill backend
    python -m sglang.launch_server \
        --model-path /models/clef-flash-int8 --port 31510 \
        --mem-fraction-static 0.85 \
        --cuda-graph-backend-prefill breakable

    # 2. expect 200 on the fixed commit, 500 on its parent
    python3 test/manual/systemone/test_decision_layouts_prefill_graph.py \
        --base-url http://127.0.0.1:31510 --model /models/clef-flash-int8

    # on the unfixed parent: add --expect-status 500

The request below is deliberately one short prompt with three question
types (choice / noul / score) so a single call covers every score path.
Stdlib only.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

QUESTIONS = {
    "routing": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments", "technical": "Bugs", "sales": "Buying"},
    },
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "severity": {
        "type": "score",
        "instructions": "Impact?",
        "criteria": ["Can wait", "This week", "Today"],
    },
}


def post(base_url, body, timeout):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode(), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode(), (time.perf_counter() - t0) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31510")
    parser.add_argument("--model", default="/models/clef-flash-int8")
    parser.add_argument("--expect-status", type=int, default=200)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    body = {
        "model": args.model,
        "state": "Checkout is failing for every customer.",
        "questions": QUESTIONS,
    }

    failures = 0
    for rep in range(args.reps):
        status, text, ms = post(args.base_url, body, args.timeout)
        print(f"[{rep}] status={status} {ms:.1f} ms", flush=True)
        if status != args.expect_status:
            failures += 1
            print(text[:800], flush=True)
            continue
        if status == 200:
            payload = json.loads(text)
            answered = payload.get("decisions") or payload.get("answers") or payload
            print("decisions:", json.dumps(answered, sort_keys=True)[:400], flush=True)

    if failures:
        print(
            f"FAIL: {failures}/{args.reps} responses were not {args.expect_status}",
            file=sys.stderr,
        )
        return 1
    print(f"OK: all {args.reps} responses were {args.expect_status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
