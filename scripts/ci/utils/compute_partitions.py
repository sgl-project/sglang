"""Sum CIRegistry est_time per per-commit suite and emit one $GITHUB_OUTPUT line
keyed by suite name. Consumed by pr-test.yml stage jobs as
`fromJson(needs.check-changes.outputs.partitions)['<suite>']`.

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

# Load ci_register.py directly: `import sglang.test...` pulls torch/orjson via
# sglang.__init__ but check-changes runs on bare ubuntu-latest. ci_register
# itself is stdlib-only (AST).
_CI_REGISTER_PATH = os.path.join(
    REPO_ROOT, "python", "sglang", "test", "ci", "ci_register.py"
)
_spec = importlib.util.spec_from_file_location("ci_register", _CI_REGISTER_PATH)
_ci_register = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ci_register)
collect_tests = _ci_register.collect_tests
HWBackend = _ci_register.HWBackend

# pr-test-amd.yml / pr-test-npu.yml have their own dispatch.
_TARGET_BACKENDS = {HWBackend.CUDA, HWBackend.CPU}

# stage-a is the critical-path entry gate; pin its fanout to smoke-coverage
# defaults instead of est_time. max_parallel = size (no throttle).
_STAGE_A_OVERRIDES = {
    "stage-a-test-cpu": 4,
    "stage-a-test-1-gpu-small": 1,
}

# Per-partition wall-clock target + ceiling. Single knob for the whole
# pipeline. ~17 min avg under perfect LPT (TARGET / LPT_SLOP), ~22 min under
# worst-case LPT 4/3 imbalance, fail-fast above 30 min.
TARGET_SECONDS = 20 * 60

# LPT (Longest Processing Time first) worst case is 4/3 * OPT; pad ~15% so a
# slightly-unlucky LPT result still fits inside MAX_PARTITION_SECONDS.
LPT_SLOP = 1.15

# Hard ceiling. Exceeded → raise, forcing the maintainer to split a slow file
# or bump TARGET_SECONDS deliberately.
MAX_PARTITION_SECONDS = 30 * 60


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


def compute_partitions(tests, full_parallel=False):
    """Group per-commit tests by suite and emit partition metadata.

    `full_parallel=True` (scheduled cron or `high priority` PR) sets
    max_parallel = size, lifting the matrix-fanout throttle.
    """
    suite_tests = defaultdict(list)
    for t in tests:
        if t.backend not in _TARGET_BACKENDS:
            continue
        if t.nightly or t.disabled is not None:
            continue
        suite_tests[t.effective_suite].append(t)

    result = {}
    for suite, group in suite_tests.items():
        total = sum(t.est_time for t in group)
        if suite in _STAGE_A_OVERRIDES:
            size = _STAGE_A_OVERRIDES[suite]
            max_parallel = size
        else:
            size = max(1, math.ceil(total * LPT_SLOP / TARGET_SECONDS))
            max_parallel = size if full_parallel else compute_max_parallel(size)
        # Check naive average (total/size). LPT can be ~4/3 of that, but the
        # ceil + LPT_SLOP padding above absorbs that slack.
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
    parser.add_argument("--repo-root", default=REPO_ROOT)
    parser.add_argument(
        "--output-format",
        choices=("gha", "json"),
        default="gha",
        help="`gha` emits `partitions=<json>` for $GITHUB_OUTPUT; `json` is raw",
    )
    parser.add_argument(
        "--full-parallel",
        choices=("true", "false"),
        default="false",
        help="Lift the max_parallel throttle (set by schedule / `high priority`)",
    )
    args = parser.parse_args()

    files = discover_files(args.repo_root)
    # Warn-not-fail on unregistered files: run_suite.py catches this at
    # test-execution time with sanity_check=True; dispatch should keep going.
    all_tests = collect_tests(files, sanity_check=False)

    result = compute_partitions(all_tests, full_parallel=(args.full_parallel == "true"))
    payload = json.dumps(result, separators=(",", ":"), sort_keys=True)
    if args.output_format == "gha":
        print(f"partitions={payload}")
    else:
        print(payload)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## Partitions\n\n")
            f.write(
                f"`full_parallel={args.full_parallel}` "
                f"(`size//4` throttle is lifted when true)\n\n"
            )
            f.write("| Suite | size | max_parallel |\n")
            f.write("|---|---:|---:|\n")
            for suite, info in sorted(result.items()):
                f.write(f"| `{suite}` | {info['size']} | {info['max_parallel']} |\n")


if __name__ == "__main__":
    main()
