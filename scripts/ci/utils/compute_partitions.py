"""Compute per-suite partition counts from the CI registry.

Reads test/registered/ (+ jit_kernel) for register_*_ci(...) calls via AST, sums
est_time per per-commit suite, and emits one JSON output containing each suite's
matrix partition count + max-parallel cap. Consumed by pr-test.yml stage jobs as
`fromJson(needs.check-changes.outputs.partitions)['<suite-name>']`.

Output format (single line for $GITHUB_OUTPUT):
    partitions={"stage-b-test-1-gpu-small": {"size": 8, "arr": [0,...,7], "max_parallel": 2}, ...}
"""

import argparse
import glob
import importlib.util
import json
import math
import os
from collections import defaultdict

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Load ci_register.py as a standalone module to avoid pulling sglang.__init__
# (which imports torch / orjson). check-changes runs on ubuntu-latest without
# sglang installed; ci_register itself only depends on stdlib + AST.
_CI_REGISTER_PATH = os.path.join(
    REPO_ROOT, "python", "sglang", "test", "ci", "ci_register.py"
)
_spec = importlib.util.spec_from_file_location("ci_register", _CI_REGISTER_PATH)
_ci_register = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ci_register)
collect_tests = _ci_register.collect_tests
HWBackend = _ci_register.HWBackend

# PR-A scope: CUDA + CPU per-commit suites. AMD / NPU workflows have their own
# dispatch (pr-test-amd.yml, pr-test-npu.yml) and can adopt this scheme later.
_TARGET_BACKENDS = {HWBackend.CUDA, HWBackend.CPU}

# stage-a is the critical-path entry gate. Its partition counts are fixed
# defaults rather than est_time-driven, because (a) every PR pays this latency,
# (b) the entry-stage budget is set by smoke coverage, not registry size,
# (c) max-parallel = size leaves no throttle since stage-a must finish fast.
_STAGE_A_OVERRIDES = {
    "stage-a-test-cpu": 4,
    "stage-a-test-1-gpu-small": 1,
}

# Wall-clock budget per partition. LPT-balanced est_time sum / TARGET_SECONDS
# gives the number of partitions. Single knob for the whole pipeline.
TARGET_SECONDS = 17 * 60

# LPT (Longest Processing Time first) worst case is 4/3 * OPT; the constant
# below pads the divisor by ~15% so a slightly-unlucky LPT result still fits
# inside MAX_PARTITION_SECONDS.
LPT_SLOP = 1.15

# Hard ceiling: if any partition's est_time after LPT would exceed this,
# compute_partitions exits non-zero so the maintainer is forced to either
# split a slow test file or raise TARGET_SECONDS deliberately.
MAX_PARTITION_SECONDS = 25 * 60


def discover_files(repo_root: str) -> list[str]:
    test_dir = os.path.join(repo_root, "test")
    files = [
        f
        for f in glob.glob(
            os.path.join(test_dir, "registered", "**", "*.py"), recursive=True
        )
        if not f.endswith("/conftest.py") and not f.endswith("/__init__.py")
    ]
    jit_kernel_dir = os.path.join(repo_root, "python", "sglang", "jit_kernel")
    files += glob.glob(
        os.path.join(jit_kernel_dir, "tests", "**", "test_*.py"), recursive=True
    )
    files += glob.glob(
        os.path.join(jit_kernel_dir, "benchmark", "**", "bench_*.py"), recursive=True
    )
    return files


def compute_max_parallel(size: int) -> int:
    return max(size // 4, 1)


def compute_partitions(tests):
    """Group per-commit tests by suite and emit partition metadata."""
    suite_tests = defaultdict(list)
    for t in tests:
        if t.backend not in _TARGET_BACKENDS:
            continue
        if t.nightly or t.disabled is not None:
            continue
        suite_tests[t.suite].append(t)

    result = {}
    for suite, group in suite_tests.items():
        total = sum(t.est_time for t in group)
        if suite in _STAGE_A_OVERRIDES:
            size = _STAGE_A_OVERRIDES[suite]
            max_parallel = size
        else:
            size = max(1, math.ceil(total * LPT_SLOP / TARGET_SECONDS))
            max_parallel = compute_max_parallel(size)
        # Coarse upper bound: assuming perfect LPT balance gives total/size
        # seconds per partition. Real LPT can be up to ~4/3 of that, but the
        # ceil + LPT_SLOP padding above already absorbs the slack, so the
        # naive average is a fair regression check.
        if total / size > MAX_PARTITION_SECONDS:
            raise RuntimeError(
                f"Suite {suite!r}: total est_time {total:.0f}s / size {size} "
                f"= {total / size:.0f}s exceeds MAX_PARTITION_SECONDS "
                f"({MAX_PARTITION_SECONDS}s). Split a slow file or raise "
                f"TARGET_SECONDS deliberately."
            )
        result[suite] = {
            "size": size,
            "arr": list(range(size)),
            "max_parallel": max_parallel,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=REPO_ROOT,
        help="Repo root for file discovery (default: derived from this script's location)",
    )
    parser.add_argument(
        "--output-format",
        choices=("gha", "json"),
        default="gha",
        help="`gha` emits `partitions=<json>` for $GITHUB_OUTPUT; `json` emits raw JSON",
    )
    args = parser.parse_args()

    files = discover_files(args.repo_root)
    # sanity_check=False so a file with no register_*_ci is a warning, not a
    # hard fail. The actual test runner (run_suite.py) uses sanity_check=True
    # to catch this at test-execution time; the CI dispatch should keep going.
    all_tests = collect_tests(files, sanity_check=False)

    result = compute_partitions(all_tests)
    payload = json.dumps(result, separators=(",", ":"), sort_keys=True)
    if args.output_format == "gha":
        print(f"partitions={payload}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
