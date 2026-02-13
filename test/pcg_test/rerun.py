#!/usr/bin/env python3
"""
Rerun specific tests from an existing PCG plan.

Edit the config below, then run:
    python3 pcg_test/rerun.py
    bash pcg_test/rerun.sh          # via tmux
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ===========================================================================
# CONFIG — edit these before running
# ===========================================================================

# GPU config: 1, 2, or 4
GPU = 1

# Tests to rerun.
# Use ["failed"] to rerun all failed tests, or list specific short_names
# (substring match, e.g. "test_dp_attention" matches "distributed/test_dp_attention.py").
RERUN_TESTS = [
    # "perf/test_bench_serving_1gpu_part1.py",
    # "core/test_gpt_oss_1gpu.py",
    # "perf/test_bench_serving_1gpu_part2.py",
    # "quant/test_w8a8_quantization.py",
    "openai_server/validation/test_large_max_new_tokens.py",
]

# Repeat count per test (None = use plan's original repeat)
REPEAT = None

# Timeout per run in seconds (None = use plan's original timeout)
TIMEOUT = None

# Skip HF cache cleanup between tests
SKIP_CLEANUP = False

# ===========================================================================
# End of config
# ===========================================================================

PCG_DIR = Path(__file__).resolve().parent
TEST_DIR = PCG_DIR.parent
PLAN_DIR = PCG_DIR / "plan"

HF_HOME = "/home/tensormesh/yuwei/huggingface"

GPU_CONFIGS = {
    1: ("plan.json", "logs", "hub"),
    2: ("plan_2gpu.json", "logs_2gpu", "hub2"),
    4: ("plan_4gpu.json", "logs_4gpu", "hub4"),
}


def get_paths(gpu):
    plan_name, logs_name, hub_name = GPU_CONFIGS[gpu]
    return (
        PLAN_DIR / plan_name,
        PCG_DIR / logs_name,
        os.path.join(HF_HOME, hub_name),
    )


# ---------------------------------------------------------------------------
# Plan I/O
# ---------------------------------------------------------------------------
def load_plan(plan_file):
    with open(plan_file) as f:
        return json.load(f)


def save_plan(plan, plan_file):
    tmp = plan_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(plan, f, indent=2)
    tmp.rename(plan_file)


# ---------------------------------------------------------------------------
# Test selection
# ---------------------------------------------------------------------------
def select_tests(plan, selectors):
    tests = plan["tests"]
    if selectors == ["failed"]:
        return [t for t in tests if t["status"] == "failed"]

    selected = []
    for t in tests:
        for sel in selectors:
            if sel in t["short_name"]:
                selected.append(t)
                break
    return selected


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def build_env(hf_hub_cache):
    env = os.environ.copy()
    env["HF_HOME"] = HF_HOME
    env["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache
    env["HF_HUB_CACHE"] = hf_hub_cache
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    return env


def run_single_test(filepath, timeout, env, log_path):
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


def clear_hf_cache(hf_hub_cache):
    cache_path = Path(hf_hub_cache)
    if not cache_path.exists():
        return
    for entry in cache_path.iterdir():
        if entry.name.startswith("models--") or entry.name.startswith("datasets--"):
            try:
                shutil.rmtree(entry)
                print(f"    [cleanup] Removed {entry.name}")
            except OSError as e:
                print(f"    [cleanup] Failed to remove {entry.name}: {e}")


def clear_old_logs(test, logs_dir):
    test_stem = Path(test["short_name"]).with_suffix("").as_posix().replace("/", "__")
    log_dir = logs_dir / test["suite"] / test_stem
    if log_dir.exists():
        shutil.rmtree(log_dir, ignore_errors=True)
        print(f"    [cleanup] Removed old logs: {log_dir.relative_to(PCG_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    plan_file, logs_dir, hf_hub_cache = get_paths(GPU)

    if not plan_file.exists():
        print(f"No plan found at {plan_file}")
        print(f"Run the {GPU}-GPU tests first to create a plan.")
        return 1

    plan = load_plan(plan_file)
    repeat = REPEAT if REPEAT is not None else plan["config"]["repeat"]
    timeout = TIMEOUT if TIMEOUT is not None else plan["config"]["timeout"]

    # Select tests
    selected = select_tests(plan, RERUN_TESTS)
    if not selected:
        print("No matching tests found.")
        print("\nAvailable tests:")
        for t in plan["tests"]:
            print(f"  [{t['status']:<10}] {t['short_name']}")
        return 1

    print(f"Rerunning {len(selected)} test(s) from {GPU}-GPU plan ({repeat}x each):\n")
    for t in selected:
        print(f"  {t['short_name']}  (was: {t['status']}, {t['runs_passed']}/{t['runs_completed']})")
    print()

    # Reset selected tests and clear their old logs
    for t in selected:
        clear_old_logs(t, logs_dir)
        t["status"] = "pending"
        t["runs_completed"] = 0
        t["runs_passed"] = 0
        t["runs_failed"] = 0
        t["run_details"] = []
    save_plan(plan, plan_file)

    # Run
    os.makedirs(hf_hub_cache, exist_ok=True)
    env = build_env(hf_hub_cache)
    start_all = time.monotonic()

    for i, t in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {t['short_name']} (suite={t['suite']}, est={t['est_time']}s)")

        t["status"] = "in_progress"
        save_plan(plan, plan_file)

        for run in range(1, repeat + 1):
            test_stem = Path(t["short_name"]).with_suffix("").as_posix().replace("/", "__")
            log_rel = f"{t['suite']}/{test_stem}/run_{run:02d}.log"
            log_path = str(logs_dir / log_rel)

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
            save_plan(plan, plan_file)

        t["status"] = "passed" if t["runs_failed"] == 0 else "failed"
        save_plan(plan, plan_file)

        print(f"    Result: {t['runs_passed']}/{t['runs_completed']} passed "
              f"-> {t['status'].upper()}")

        if not SKIP_CLEANUP:
            clear_hf_cache(hf_hub_cache)

        print()

    elapsed_all = time.monotonic() - start_all
    print(f"Wall time: {elapsed_all:.0f}s ({elapsed_all / 3600:.1f}h)")

    # Print full plan summary
    print("\n" + "=" * 110)
    print(f"FULL {GPU}-GPU PLAN STATUS")
    print("=" * 110)
    print(f"{'Status':<10} {'Suite':<30} {'Test':<45} {'Pass':>5}/{repeat:<5}")
    print("-" * 110)
    for t in plan["tests"]:
        if t["status"] == "pending":
            continue
        marker = "PASS" if t["status"] == "passed" else "FAIL"
        print(f"{marker:<10} {t['suite']:<30} {t['short_name']:<45} "
              f"{t['runs_passed']:>5}/{t['runs_completed']:<5}")
    print("-" * 110)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
