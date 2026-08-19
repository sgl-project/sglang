"""TTFT benchmark for KDA prefill CP (manual).

Measures single-request time-to-first-token (max_new_tokens=1 wall clock)
across prompt lengths on one server. Each request uses fresh random
input_ids so the radix cache never short-circuits the prefill; per length:
1 warmup (Triton autotune / graph paths) + N measured runs, median reported.

Usage: python3 kda_cp_ttft_bench.py <port> <tag> [lengths...]
"""

import json
import random
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else str(PORT)
LENGTHS = (
    [int(x) for x in sys.argv[3:]]
    if len(sys.argv) > 3
    else [8192, 32768, 65536, 131072]
)
RUNS = 3


def gen(prompt_ids):
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
    }
    req = urllib.request.Request(
        f"http://localhost:{PORT}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        json.loads(resp.read())
    return time.perf_counter() - t0


def main():
    rng = random.Random(1234)
    for length in LENGTHS:
        times = []
        for run in range(RUNS + 1):
            ids = [rng.randint(10000, 40000) for _ in range(length)]
            elapsed = gen(ids)
            if run > 0:  # skip warmup
                times.append(elapsed)
        med = statistics.median(times)
        print(
            f"[{TAG}] len={length:>7d} ttft_median={med:8.3f}s "
            f"runs={['%.3f' % t for t in times]} "
            f"prefill_tok_per_s={length / med:,.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
