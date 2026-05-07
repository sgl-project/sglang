#!/usr/bin/env python3
"""Run the AITER custom all-reduce benchmark and check perf thresholds.

This wraps ``benchmark/kernels/all_reduce/benchmark_aiter.py`` for AMD CI to
detect regressions in aiter's custom all-reduce kernel. It launches the
benchmark via ``torchrun``, parses the per-size average latency from rank-0
stdout, and compares each measurement against an upper bound (ms) loaded from
a JSON config.

Threshold JSON format (see ``aiter_allreduce_thresholds.json``):

    {
      "tolerance_pct": 0.0,
      "world_size_2": {
        "32K": <float | null>,
        ...
      }
    }

A ``null`` threshold means the size is reported but not checked. A set
threshold fails the run if either the measurement is missing or
``aiter_ms > threshold * (1 + tolerance_pct / 100)``.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

AITER_LINE_RE = re.compile(r"^\[Aiter\]\s+(\S+):\s+([0-9]+\.[0-9]+)\s+ms")
SGLANG_LINE_RE = re.compile(r"^\[SGLang\]\s+(\S+):\s+([0-9]+\.[0-9]+)\s+ms")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nproc-per-node", type=int, default=2)
    p.add_argument(
        "--thresholds",
        type=Path,
        default=Path("scripts/ci/amd/aiter_allreduce_thresholds.json"),
    )
    p.add_argument(
        "--benchmark-script",
        type=Path,
        default=Path("benchmark/kernels/all_reduce/benchmark_aiter.py"),
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters-small", type=int, default=50)
    p.add_argument("--iters-large", type=int, default=20)
    p.add_argument(
        "--tolerance-pct",
        type=float,
        default=None,
        help="Override the tolerance_pct from the JSON config.",
    )
    return p.parse_args()


def run_benchmark(args) -> str:
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        str(args.benchmark_script),
        "--warmup",
        str(args.warmup),
        "--iters-small",
        str(args.iters_small),
        "--iters-large",
        str(args.iters_large),
    ]
    print(f"[runner] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    if proc.returncode != 0:
        print(
            f"[runner] FAIL: benchmark exited with code {proc.returncode}",
            flush=True,
        )
        sys.exit(proc.returncode)
    return proc.stdout


def parse_results(stdout: str):
    aiter, sgl = {}, {}
    for line in stdout.splitlines():
        m = AITER_LINE_RE.match(line)
        if m:
            aiter[m.group(1)] = float(m.group(2))
            continue
        m = SGLANG_LINE_RE.match(line)
        if m:
            sgl[m.group(1)] = float(m.group(2))
    return aiter, sgl


def main():
    args = parse_args()

    if not args.thresholds.is_file():
        print(f"[runner] FAIL: thresholds file not found: {args.thresholds}")
        sys.exit(2)
    cfg = json.loads(args.thresholds.read_text())

    section_key = f"world_size_{args.nproc_per_node}"
    thresholds = cfg.get(section_key)
    if thresholds is None:
        print(
            f"[runner] FAIL: thresholds JSON has no section '{section_key}'.",
            flush=True,
        )
        sys.exit(2)

    tol = (
        args.tolerance_pct
        if args.tolerance_pct is not None
        else float(cfg.get("tolerance_pct", 0.0))
    )

    stdout = run_benchmark(args)
    aiter_ms, _ = parse_results(stdout)

    if not aiter_ms:
        print(
            "[runner] FAIL: no aiter timings parsed from benchmark output. "
            "Aiter import / construction likely failed; check stderr above.",
            flush=True,
        )
        sys.exit(2)

    print(
        f"\n[runner] AITER custom all-reduce regression check "
        f"(world_size={args.nproc_per_node}, tolerance={tol}%):",
        flush=True,
    )
    header = f"  {'Size':>6}  {'Aiter(ms)':>10}  {'Bound(ms)':>10}  {'Status':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    failures = []
    for size, threshold in thresholds.items():
        actual = aiter_ms.get(size)
        if threshold is None:
            actual_str = f"{actual:.3f}" if actual is not None else "-"
            print(f"  {size:>6}  {actual_str:>10}  {'-':>10}  {'SKIP':>8}")
            continue

        bound = float(threshold) * (1.0 + tol / 100.0)
        if actual is None:
            print(
                f"  {size:>6}  {'-':>10}  {bound:>10.3f}  {'MISSING':>8}"
            )
            failures.append((size, None, bound))
            continue

        if actual > bound:
            status = "REGRESS"
            failures.append((size, actual, bound))
        else:
            status = "OK"
        print(f"  {size:>6}  {actual:>10.3f}  {bound:>10.3f}  {status:>8}")

    if failures:
        print("", flush=True)
        for size, actual, bound in failures:
            if actual is None:
                print(
                    f"[runner] REGRESSION: size={size} no measurement "
                    f"(threshold={bound:.3f} ms)",
                    flush=True,
                )
            else:
                print(
                    f"[runner] REGRESSION: size={size} aiter={actual:.3f} ms "
                    f"> bound={bound:.3f} ms",
                    flush=True,
                )
        sys.exit(1)

    print("[runner] All checked sizes within thresholds.", flush=True)


if __name__ == "__main__":
    main()
