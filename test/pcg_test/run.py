#!/usr/bin/env python3
"""
Piecewise CUDA Graph (PCG) stress test runner with checkpoint support.

Directory layout:
    pcg_test/
        run.py          -- this script
        run.sh          -- shell wrapper
        plan/
            plan.json   -- test plan with checkpoint state
        logs/
            <suite>/<test_name>/
                run_01.log
                run_02.log
                ...

Features:
    - Discovers all 1-GPU CUDA tests via AST parsing.
    - Runs each test N times (default 10).
    - Saves full stdout+stderr to per-run log files.
    - Maintains plan.json as a checkpoint: on restart, completed tests are
      skipped so you can resume from where you left off.
    - Clears HuggingFace model cache after each test to save disk.

Usage:
    bash pcg_test/run.sh                         # run everything, resume if plan exists
    bash pcg_test/run.sh --repeat 5              # 5 runs per test
    bash pcg_test/run.sh --suite stage-a-test-1   # single suite
    bash pcg_test/run.sh --reset                  # discard old plan, start fresh
"""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PCG_DIR = Path(__file__).resolve().parent           # pcg_test/
TEST_DIR = PCG_DIR.parent                           # test/
REGISTERED_DIR = TEST_DIR / "registered"
PLAN_DIR = PCG_DIR / "plan"
LOGS_DIR = PCG_DIR / "logs"
PLAN_FILE = PLAN_DIR / "plan.json"

HF_HOME = "/home/tensormesh/yuwei/huggingface"
HF_HUB_CACHE = os.path.join(HF_HOME, "hub")

ALL_1GPU_SUITES = [
    "stage-a-test-1",
    "stage-b-test-small-1-gpu",
    "stage-b-test-large-1-gpu",
    "nightly-1-gpu",
]

# ---------------------------------------------------------------------------
# Plan data model
# ---------------------------------------------------------------------------
# plan.json schema:
# {
#   "config": { "repeat": 10, "timeout": 1200, "suites": [...] },
#   "tests": [
#     {
#       "filepath": "/abs/path/to/test.py",
#       "short_name": "core/test_srt_endpoint.py",
#       "suite": "stage-b-test-small-1-gpu",
#       "est_time": 127,
#       "status": "pending" | "in_progress" | "passed" | "failed",
#       "runs_completed": 0,
#       "runs_passed": 0,
#       "runs_failed": 0,
#       "run_details": [
#         { "run": 1, "returncode": 0, "elapsed": 42.3, "log": "logs/..." }
#       ]
#     }, ...
#   ]
# }


def discover_tests(suites: List[str]):
    """Discover CUDA tests for given suites via AST parsing."""
    tests = []
    for pyfile in sorted(REGISTERED_DIR.rglob("*.py")):
        try:
            source = pyfile.read_text()
            tree = ast.parse(source, filename=str(pyfile))
        except SyntaxError:
            continue

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.Expr):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name != "register_cuda_ci":
                continue

            est_time, suite, disabled = 0, "", None
            positional_keys = ["est_time", "suite", "nightly", "disabled"]
            for i, arg in enumerate(call.args):
                if i < len(positional_keys):
                    val = ast.literal_eval(arg)
                    if positional_keys[i] == "est_time":
                        est_time = val
                    elif positional_keys[i] == "suite":
                        suite = val
                    elif positional_keys[i] == "disabled":
                        disabled = val

            for kw in call.keywords:
                val = ast.literal_eval(kw.value)
                if kw.arg == "est_time":
                    est_time = val
                elif kw.arg == "suite":
                    suite = val
                elif kw.arg == "disabled":
                    disabled = val

            if suite in suites and disabled is None:
                short = os.path.relpath(str(pyfile), REGISTERED_DIR)
                tests.append({
                    "filepath": str(pyfile),
                    "short_name": short,
                    "suite": suite,
                    "est_time": est_time,
                    "status": "pending",
                    "runs_completed": 0,
                    "runs_passed": 0,
                    "runs_failed": 0,
                    "run_details": [],
                })

    suite_order = {s: i for i, s in enumerate(suites)}
    tests.sort(key=lambda t: (suite_order.get(t["suite"], 999), t["filepath"]))
    return tests


def create_plan(suites, repeat, timeout):
    """Create a fresh plan.json."""
    tests = discover_tests(suites)
    plan = {
        "config": {
            "repeat": repeat,
            "timeout": timeout,
            "suites": suites,
            "created": datetime.now().isoformat(),
        },
        "tests": tests,
    }
    return plan


def load_plan():
    """Load existing plan from disk."""
    with open(PLAN_FILE) as f:
        return json.load(f)


def save_plan(plan):
    """Atomically save plan to disk."""
    tmp = PLAN_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(plan, f, indent=2)
    tmp.rename(PLAN_FILE)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def build_env():
    env = os.environ.copy()
    env["HF_HOME"] = HF_HOME
    env["HUGGINGFACE_HUB_CACHE"] = HF_HUB_CACHE
    env["HF_HUB_CACHE"] = HF_HUB_CACHE
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    return env


def run_single_test(filepath, timeout, env, log_path):
    """Run one test, writing full output to log_path. Return (returncode, elapsed)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    start = time.monotonic()
    try:
        with open(log_path, "w") as logf:
            logf.write(f"# Test: {filepath}\n")
            logf.write(f"# Started: {datetime.now().isoformat()}\n")
            logf.write(f"# Timeout: {timeout}s\n")
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
        # Append footer
        with open(log_path, "a") as logf:
            logf.write(f"\n{'=' * 80}\n")
            logf.write(f"# Finished: {datetime.now().isoformat()}\n")
            logf.write(f"# Return code: {proc.returncode}\n")
            logf.write(f"# Elapsed: {elapsed:.1f}s\n")
        return proc.returncode, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        with open(log_path, "a") as logf:
            logf.write(f"\n{'=' * 80}\n")
            logf.write(f"# TIMEOUT after {elapsed:.1f}s\n")
        return -1, elapsed


def clear_hf_cache():
    cache_path = Path(HF_HUB_CACHE)
    if not cache_path.exists():
        return
    for entry in cache_path.iterdir():
        if entry.name.startswith("models--") or entry.name.startswith("datasets--"):
            shutil.rmtree(entry, ignore_errors=True)
    print("    [cleanup] Cleared model cache")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_plan_status(plan):
    """Print current plan status as a table."""
    tests = plan["tests"]
    counts = {"pending": 0, "in_progress": 0, "passed": 0, "failed": 0}
    for t in tests:
        counts[t["status"]] = counts.get(t["status"], 0) + 1

    total = len(tests)
    done = counts["passed"] + counts["failed"]
    print(f"\nPlan: {done}/{total} completed  "
          f"({counts['passed']} passed, {counts['failed']} failed, "
          f"{counts['pending'] + counts['in_progress']} remaining)\n")


def print_summary(plan):
    repeat = plan["config"]["repeat"]
    print("\n" + "=" * 110)
    print("PCG STRESS TEST SUMMARY")
    print("=" * 110)
    print(f"{'Status':<10} {'Suite':<30} {'Test':<45} {'Pass':>5}/{repeat:<5}")
    print("-" * 110)

    failed_tests = []
    for t in plan["tests"]:
        if t["status"] == "pending":
            continue
        marker = "PASS" if t["status"] == "passed" else "FAIL"
        print(f"{marker:<10} {t['suite']:<30} {t['short_name']:<45} "
              f"{t['runs_passed']:>5}/{t['runs_completed']:<5}")
        if t["status"] == "failed":
            failed_tests.append(t)

    print("-" * 110)

    if failed_tests:
        print(f"\nFAILED ({len(failed_tests)}):")
        for t in failed_tests:
            failed_runs = [r for r in t["run_details"] if r["returncode"] != 0]
            indices = [r["run"] for r in failed_runs]
            print(f"  {t['short_name']}  (failed runs: {indices})")
            if failed_runs:
                print(f"    log: {failed_runs[0]['log']}")
    else:
        done = sum(1 for t in plan["tests"] if t["status"] in ("passed", "failed"))
        if done == len(plan["tests"]):
            print("\nAll tests passed!")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PCG stress test with checkpoint support"
    )
    parser.add_argument(
        "--repeat", type=int, default=10,
        help="Runs per test (default: 10)",
    )
    parser.add_argument(
        "--timeout", type=int, default=1200,
        help="Timeout per run in seconds (default: 1200)",
    )
    parser.add_argument(
        "--suite", nargs="+", default=ALL_1GPU_SUITES,
        help="Suites to test (default: all 1-GPU suites)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Discard existing plan and start fresh",
    )
    parser.add_argument(
        "--skip-cleanup", action="store_true",
        help="Skip HF cache cleanup between tests",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Just print current plan status and exit",
    )
    args = parser.parse_args()

    # Ensure directories exist
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    os.makedirs(HF_HUB_CACHE, exist_ok=True)

    # --status: print and exit
    if args.status:
        if PLAN_FILE.exists():
            plan = load_plan()
            print_plan_status(plan)
            print_summary(plan)
        else:
            print("No plan found. Run without --status to create one.")
        return

    # Load or create plan
    if PLAN_FILE.exists() and not args.reset:
        plan = load_plan()
        print(f"Resuming from existing plan ({PLAN_FILE})")
        print_plan_status(plan)
    else:
        if args.reset and PLAN_FILE.exists():
            print("Resetting: discarding old plan")
        plan = create_plan(args.suite, args.repeat, args.timeout)
        save_plan(plan)
        print(f"Created new plan with {len(plan['tests'])} tests")
        print(f"  Suites: {args.suite}")
        print(f"  Repeat: {args.repeat}x")
        print(f"  Timeout: {args.timeout}s per run")
        print(f"  Logs: {LOGS_DIR}")
        print(f"  Plan: {PLAN_FILE}")
        print(f"  HF cache: {HF_HUB_CACHE}")
        print()

    repeat = plan["config"]["repeat"]
    timeout = plan["config"]["timeout"]
    env = build_env()
    tests = plan["tests"]
    total = len(tests)

    # Count remaining
    remaining = [t for t in tests if t["status"] in ("pending", "in_progress")]
    if not remaining:
        print("All tests already completed.")
        print_summary(plan)
        return

    print(f"Running {len(remaining)} remaining tests ({repeat}x each)\n")
    start_all = time.monotonic()

    for t in tests:
        if t["status"] in ("passed", "failed"):
            continue

        idx = tests.index(t) + 1
        print(f"[{idx}/{total}] {t['short_name']} (suite={t['suite']}, est={t['est_time']}s)")

        # If previously interrupted (in_progress), reset and start from scratch
        if t["status"] == "in_progress":
            print(f"    Resetting partially completed test ({t['runs_completed']}/{repeat} runs)")
            t["runs_completed"] = 0
            t["runs_passed"] = 0
            t["runs_failed"] = 0
            t["run_details"] = []

        t["status"] = "in_progress"
        save_plan(plan)

        for run in range(1, repeat + 1):
            # Log path: logs/<suite>/<test_stem>/run_NN.log
            test_stem = Path(t["short_name"]).with_suffix("").as_posix().replace("/", "__")
            log_rel = f"{t['suite']}/{test_stem}/run_{run:02d}.log"
            log_path = str(LOGS_DIR / log_rel)

            rc, elapsed = run_single_test(t["filepath"], timeout, env, log_path)

            status_str = "PASS" if rc == 0 else "FAIL"
            print(f"    Run {run:>2}/{repeat}: {status_str}  ({elapsed:.1f}s, rc={rc})")

            t["runs_completed"] = run
            if rc == 0:
                t["runs_passed"] += 1
            else:
                t["runs_failed"] += 1
            t["run_details"].append({
                "run": run,
                "returncode": rc,
                "elapsed": round(elapsed, 2),
                "log": log_rel,
            })

            # Checkpoint after every run
            save_plan(plan)

        # Mark final status
        t["status"] = "passed" if t["runs_failed"] == 0 else "failed"
        save_plan(plan)

        print(f"    Result: {t['runs_passed']}/{t['runs_completed']} passed "
              f"-> {t['status'].upper()}")

        # Cleanup
        if not args.skip_cleanup:
            clear_hf_cache()

        print()

    elapsed_all = time.monotonic() - start_all
    print(f"Wall time: {elapsed_all:.0f}s ({elapsed_all / 3600:.1f}h)")
    print_summary(plan)


if __name__ == "__main__":
    main()
