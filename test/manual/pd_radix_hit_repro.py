"""Radix-hit repro driver for the A5 PD host_rdma MTE investigation.

Fires the SAME long prompt several times, spaced apart, against the PD
router. The spacing lets each prefill finish and populate the radix tree
before the next copy arrives, so later requests run as radix-hit
(cached-token > 0) prefill batches -- the batch class that faults in the
crash runs. Requests are expected to hang at decode when
SGLANG_DEBUG_SKIP_KV_SEND=1 is set; that is fine, only prefill matters.

Usage:
  python3 test/manual/pd_radix_hit_repro.py --url http://127.0.0.1:30000

Afterwards check the prefill log:
  grep "#cached-token" <prefill_log> | grep -v "cached-token: 0"
Non-empty output means radix-hit batches actually ran and the run counts.
"""

import argparse
import json
import threading
import time
import urllib.request

# ~10 tokens per repetition; deterministic so every run uses the same text.
BASE_SENTENCE = "The quick brown fox jumps over the lazy dog."
TOKENS_PER_REP = 10


def build_text(token_target: int) -> str:
    reps = (token_target + TOKENS_PER_REP - 1) // TOKENS_PER_REP
    return (BASE_SENTENCE + " ") * reps


def send_one(url, payload, timeout, idx, total, results):
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            outcome = f"status={response.status}"
    except Exception as exc:  # timeout/abort expected under the skip switch
        outcome = f"{type(exc).__name__}"
    results[idx] = outcome
    print(
        f"[{idx + 1}/{total}] {outcome} elapsed={time.time() - start:.1f}s", flush=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        required=True,
        help="PD ROUTER address (the one your benchmark targets). Pointing "
        "this at a prefill/decode instance trips a bootstrap_room assert "
        "and kills the whole stack.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=12000,
        help="prompt length target; radix in this setup needs 2k+",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--interval",
        type=float,
        default=25.0,
        help="seconds between sends; must exceed one prefill",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per-request client timeout; long so the server "
        "side finishes and caches even under launch blocking",
    )
    args = parser.parse_args()

    payload = {
        "text": build_text(args.tokens),
        "sampling_params": {"max_new_tokens": 8, "ignore_eos": True},
    }
    print(
        f"prompt ~{args.tokens} tokens, {args.count} sends, "
        f"interval {args.interval}s -> total ~{args.count * args.interval}s",
        flush=True,
    )

    results = [None] * args.count
    threads = []
    for i in range(args.count):
        # Stagger by starting each thread at its own scheduled time.
        def worker(idx=i):
            delay = idx * args.interval - (time.time() - t0)
            if delay > 0:
                time.sleep(delay)
            send_one(args.url, payload, args.timeout, idx, args.count, results)

        threads.append(threading.Thread(target=worker))
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("done; check the prefill log for radix hits:")
    print('  grep "#cached-token" <prefill_log> | grep -v "cached-token: 0"')


if __name__ == "__main__":
    main()
