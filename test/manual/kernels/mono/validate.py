"""Validate a monokernel change end to end (RFC #43949): every check a change needs
before it merges, and decode speed against the recorded baseline. Needs the model
and the GPUs of the serving test (MI35x, TP4).

    python test/manual/kernels/mono/validate.py                  # every step
    python test/manual/kernels/mono/validate.py --steps unit,perf

Steps:
  unit     the plugin and the specs on CPU (test/registered/unit/mono)
  kernels  K1, K2 and the FFN launch against their references, and the guarded
           waits (test/registered/kernels/ops/mono)
  e2e      verify (DSpark) and decode steps past the largest width and back:
           GSM8K, and KL consistency against the normal prefill path; and a fault
           on one rank that stops every rank without sending a token
           (test/registered/e2e/mono)
  perf     the server with and without --enable-monokernel: per-user decode
           tok/s at 1, 4 and 8 concurrent requests, and at 16, past the largest
           width, against baselines.json (--record writes the measured rates
           there instead)

Prints PASS or FAIL for each step, and exits nonzero on any failure.
"""

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

from sglang.test.kernels.mono.deepseek_v41_hip.server import ARGS, ENV, MODEL
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

REPO = Path(__file__).resolve().parents[4]
BASELINES = Path(__file__).with_name("baselines.json")
SPEC = "deepseek_v41_hip"
SETUP = (
    "sglang.test.kernels.mono.deepseek_v41_hip.server (MI35x, TP4, DSpark), "
    "PERF_ENV: per-user decode tok/s of 1500-token requests"
)
TESTS = {
    "unit": "test/registered/unit/mono",
    "kernels": "test/registered/kernels/ops/mono",
    "e2e": "test/registered/e2e/mono",
}
# DSpark's acceptance is fixed, so the A/B measures the kernels, not the draft
PERF_ENV = {
    "SGLANG_SIMULATE_ACC_LEN": "3.51",
    "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
}
PROMPTS = (
    "Write a long detailed essay about the history of lighthouses and their keepers.",
    "Explain in great detail how TCP congestion control algorithms evolved.",
    "Describe the full lifecycle of a star from nebula to remnant, in depth.",
    "Give a thorough tutorial on writing a compiler front end in Python.",
)
# the largest width is 8 requests of 6 verify rows: 16 runs past it
CONCURRENCY = (1, 4, 8, 16)
TOKENS, ROUNDS = 1500, 3


def run_tests(path: str) -> bool:
    cmd = [sys.executable, "-m", "pytest", "-q", str(REPO / path)]
    return subprocess.run(cmd, cwd=REPO).returncode == 0


def _per_user_rate(i: int) -> float:
    body = {
        "text": PROMPTS[i % len(PROMPTS)],
        "sampling_params": {
            "max_new_tokens": TOKENS,
            "temperature": 0,
            "ignore_eos": True,
        },
    }
    start = time.perf_counter()
    r = requests.post(DEFAULT_URL_FOR_TEST + "/generate", json=body, timeout=900)
    r.raise_for_status()
    return r.json()["meta_info"]["completion_tokens"] / (time.perf_counter() - start)


def decode_rates(monokernel: bool) -> dict[int, float]:
    """The median over ROUNDS of the mean per-user decode tok/s, at each concurrency."""
    process = popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=3600,
        other_args=ARGS + (["--enable-monokernel"] if monokernel else []),
        env={**os.environ, **ENV, **PERF_ENV},
    )
    try:
        rates = {}
        for n in CONCURRENCY:
            rounds = []
            for _ in range(ROUNDS):
                with cf.ThreadPoolExecutor(n) as pool:
                    rounds.append(statistics.mean(pool.map(_per_user_rate, range(n))))
            rates[n] = statistics.median(rounds)
        return rates
    finally:
        terminate_and_kill_process_tree(process)


def perf(tolerance: float, record: bool) -> bool:
    normal, mono = decode_rates(monokernel=False), decode_rates(monokernel=True)
    for n in CONCURRENCY:
        print(
            f"  {n} concurrent: {normal[n]:.1f} -> {mono[n]:.1f} tok/s per user "
            f"({mono[n] / normal[n] - 1:+.1%})"
        )
    if record:
        rates = {"normal": normal, "monokernel": mono}
        entry = {
            k: {str(n): round(v[n], 1) for n in CONCURRENCY} for k, v in rates.items()
        }
        BASELINES.write_text(
            json.dumps({SPEC: {"setup": SETUP, **entry}}, indent=2) + "\n"
        )
        return True
    if not BASELINES.exists():
        print(f"  no baseline: run with --record to write {BASELINES.name}")
        return False
    base = json.loads(BASELINES.read_text())[SPEC]["monokernel"]
    missing = [n for n in CONCURRENCY if str(n) not in base]
    for n in missing:
        print(f"  {n} concurrent: no baseline: run with --record")
    slow = [
        n
        for n in CONCURRENCY
        if str(n) in base and mono[n] < base[str(n)] * (1 - tolerance)
    ]
    for n in slow:
        print(
            f"  {n} concurrent: {mono[n]:.1f} is below the baseline {base[str(n)]:.1f}"
        )
    return not slow and not missing


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", default="unit,kernels,e2e,perf")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="perf: the largest drop below the baseline that still passes",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="perf: write the measured rates as the baseline",
    )
    args = parser.parse_args()
    results = {}
    for step in args.steps.split(","):
        print(f"== {step}", flush=True)
        if step == "perf":
            results[step] = perf(args.tolerance, args.record)
        else:
            results[step] = run_tests(TESTS[step])
    for step, ok in results.items():
        print(f"{step:8s} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
