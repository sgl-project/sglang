#!/usr/bin/env python3
"""
Run manual/PCG bug tests one at a time for investigation.

These are tests from pcg_bugs.txt that need manual investigation.
Each test runs once (not 10x) so you can inspect output interactively.

Usage:
    python3 pcg_test/manual_test.py                # run all
    python3 pcg_test/manual_test.py --list          # list tests
    python3 pcg_test/manual_test.py --test 3        # run test #3 only
    python3 pcg_test/manual_test.py --gpu 2         # run only 2-GPU tests
    bash pcg_test/manual_test.sh                    # via tmux (1-GPU)
    bash pcg_test/manual_test.sh 2                  # via tmux (2-GPU)
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ===========================================================================
# Manual test list — from pcg_bugs.txt
# ===========================================================================

MANUAL_TESTS = [
    # --- 1-GPU PCG bugs ---
    {
        "gpu": 1,
        "short_name": "scheduler/test_retract_decode.py",
        "filepath": "registered/scheduler/test_retract_decode.py",
        "summary": "0/10; CUDA device-side assert (vectorized_gather_kernel index OOB) during decode retraction + re-prefill under PCG",
    },
    {
        "gpu": 1,
        "short_name": "core/test_gpt_oss_1gpu.py",
        "filepath": "registered/core/test_gpt_oss_1gpu.py",
        "summary": "3/10; CUDA device-side assert under PCG; both bf16 and mxfp4 models affected; garbage output after CUDA error",
    },
    {
        "gpu": 1,
        "short_name": "lora/test_lora_backend.py",
        "filepath": "registered/lora/test_lora_backend.py",
        "summary": "0/10; moved from rerun list for manual investigation",
    },
    {
        "gpu": 1,
        "short_name": "quant/test_torchao.py",
        "filepath": "registered/quant/test_torchao.py",
        "summary": "9/10; vectorized_gather_kernel index OOB + CUDA device-side assert; same pattern as test_retract_decode; flaky",
    },
    # --- 1-GPU needs manual run (timeout / infra) ---
    {
        "gpu": 1,
        "short_name": "perf/test_bench_serving_1gpu_part1.py",
        "filepath": "registered/perf/test_bench_serving_1gpu_part1.py",
        "summary": "0/10; 8 server launches + PCG warmup (~50s total) exceed 1200s timeout; leaked server cascades OOM to runs 2-10",
    },
    # --- 2-GPU tests ---
    {
        "gpu": 2,
        "short_name": "lora/test_lora_tp.py",
        "filepath": "registered/lora/test_lora_tp.py",
        "summary": "0/10; LoRA servers auto-disable PCG; crash is in no-LoRA baseline server; PCG + TP=2 on Llama-2-7b-hf crashes with CUDA illegal memory access",
    },
]

# ===========================================================================
# Config
# ===========================================================================

PCG_DIR = Path(__file__).resolve().parent
TEST_DIR = PCG_DIR.parent
LOG_DIR = PCG_DIR / "logs_manual"

HF_HOME = "/home/tensormesh/yuwei/huggingface"

HF_HUB_NAME = "hub-manual"


def build_env(gpu):
    hf_hub_cache = os.path.join(HF_HOME, HF_HUB_NAME)
    env = os.environ.copy()
    env["HF_HOME"] = HF_HOME
    env["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache
    env["HF_HUB_CACHE"] = hf_hub_cache
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    return env


def run_test(test, timeout=2400):
    filepath = str(TEST_DIR / test["filepath"])
    test_stem = Path(test["short_name"]).with_suffix("").as_posix().replace("/", "__")
    log_path = LOG_DIR / f"{test['gpu']}gpu" / f"{test_stem}.log"
    os.makedirs(log_path.parent, exist_ok=True)

    env = build_env(test["gpu"])

    print(f"\n{'=' * 80}")
    print(f"[{test['gpu']}-GPU] {test['short_name']}")
    print(f"  Summary: {test['summary']}")
    print(f"  Log: {log_path.relative_to(PCG_DIR)}")
    print(f"{'=' * 80}")

    start = time.monotonic()
    try:
        with open(log_path, "w") as logf:
            logf.write(f"# Test: {filepath}\n")
            logf.write(f"# Started: {datetime.now().isoformat()}\n")
            logf.write(f"# GPU: {test['gpu']}\n")
            logf.write(f"# Summary: {test['summary']}\n")
            logf.write("=" * 80 + "\n\n")
            logf.flush()

            proc = subprocess.run(
                [sys.executable, filepath],
                cwd=str(TEST_DIR),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        elapsed = time.monotonic() - start
        with open(log_path, "a") as logf:
            logf.write(f"\n{'=' * 80}\n")
            logf.write(f"# Finished: {datetime.now().isoformat()}\n")
            logf.write(f"# Return code: {proc.returncode}\n")
            logf.write(f"# Elapsed: {elapsed:.1f}s\n")
        status = "PASS" if proc.returncode == 0 else "FAIL"
        print(f"  Result: {status}  (rc={proc.returncode}, {elapsed:.1f}s)")
        return proc.returncode

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        with open(log_path, "a") as logf:
            logf.write(f"\n{'=' * 80}\n")
            logf.write(f"# TIMEOUT after {elapsed:.1f}s\n")
        print(f"  Result: TIMEOUT  ({elapsed:.1f}s)")
        return -1


def list_tests(gpu_filter=None):
    print(f"\n{'#':<4} {'GPU':<5} {'Test':<55} {'Summary'}")
    print("-" * 130)
    for i, t in enumerate(MANUAL_TESTS, 1):
        if gpu_filter and t["gpu"] != gpu_filter:
            continue
        print(f"{i:<4} {t['gpu']}-GPU  {t['short_name']:<55} {t['summary'][:60]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run manual PCG bug tests")
    parser.add_argument("--list", action="store_true", help="List all tests")
    parser.add_argument("--test", type=int, help="Run specific test by number (1-based)")
    parser.add_argument("--gpu", type=int, choices=[1, 2, 4], help="Filter by GPU count")
    parser.add_argument("--timeout", type=int, default=2400, help="Timeout per test in seconds")
    args = parser.parse_args()

    if args.list:
        list_tests(args.gpu)
        return 0

    if args.test:
        if args.test < 1 or args.test > len(MANUAL_TESTS):
            print(f"Invalid test number {args.test}. Use --list to see available tests.")
            return 1
        test = MANUAL_TESTS[args.test - 1]
        rc = run_test(test, args.timeout)
        return 0 if rc == 0 else 1

    # Run all (filtered by GPU if specified)
    tests = MANUAL_TESTS
    if args.gpu:
        tests = [t for t in tests if t["gpu"] == args.gpu]

    if not tests:
        print("No tests to run.")
        return 1

    print(f"Running {len(tests)} manual test(s)...\n")
    results = []
    start_all = time.monotonic()

    for i, t in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}]", end="")
        rc = run_test(t, args.timeout)
        results.append((t, rc))

    elapsed_all = time.monotonic() - start_all

    # Summary
    print(f"\n\n{'=' * 80}")
    print("MANUAL TEST SUMMARY")
    print(f"{'=' * 80}")
    passed = sum(1 for _, rc in results if rc == 0)
    failed = sum(1 for _, rc in results if rc != 0)
    print(f"Passed: {passed}  Failed: {failed}  Total: {len(results)}")
    print(f"Wall time: {elapsed_all:.0f}s ({elapsed_all / 3600:.1f}h)\n")

    print(f"{'Status':<8} {'GPU':<5} {'Test'}")
    print("-" * 80)
    for t, rc in results:
        status = "PASS" if rc == 0 else "FAIL"
        print(f"{status:<8} {t['gpu']}-GPU  {t['short_name']}")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
