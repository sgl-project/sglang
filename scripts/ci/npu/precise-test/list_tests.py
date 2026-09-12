#!/usr/bin/env python3
"""
List the test files selected for a CI suite without running them.

Standalone extraction of `test/run_suite.py --list-tests-output`: discover
the registered tests under test/registered/, filter them by hardware
backend / suite / nightly flag, then write the selected test file paths
(one per line) to the output file.

Usage:
    python3 list_tests.py --hw npu --suite base-b-test-1-npu-a3 \
        [--nightly] [--auto-partition-id N --auto-partition-size M] \
        -o /tmp/selected_tests.txt
"""

import argparse
import glob
import os
import sys
from pathlib import Path

# Repo layout: this script lives at <repo>/scripts/ci/npu/precise-test/.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

# ci_register.py is stdlib-only; import it directly (bypassing the sglang
# package __init__, which pulls in torch) so this script runs anywhere.
sys.path.insert(0, str(REPO_ROOT / "python" / "sglang" / "test" / "ci"))

from ci_register import HWBackend, auto_partition, collect_tests  # noqa: E402

HW_MAPPING = {
    "cpu": HWBackend.CPU,
    "cuda": HWBackend.CUDA,
    "amd": HWBackend.AMD,
    "musa": HWBackend.MUSA,
    "npu": HWBackend.NPU,
    "xpu": HWBackend.XPU,
    "mlx": HWBackend.MLX,
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Write the test files selected for a CI suite (one per line) "
            "without running them."
        )
    )
    parser.add_argument(
        "--hw",
        type=str,
        choices=HW_MAPPING.keys(),
        required=True,
        help="Hardware backend to select tests for.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help=(
            "Test suite to select. Accepts a comma-separated list of suites; "
            "their tests are unioned."
        ),
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Include tests registered with nightly=True.",
    )
    parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Write selected test file paths (one per line) to this file.",
    )
    args = parser.parse_args()

    # Validate auto-partition arguments (same rules as run_suite.py).
    if (args.auto_partition_id is not None) != (args.auto_partition_size is not None):
        parser.error(
            "--auto-partition-id and --auto-partition-size must be specified together."
        )
    if args.auto_partition_size is not None:
        if args.auto_partition_size <= 0:
            parser.error("--auto-partition-size must be positive.")
        if not 0 <= args.auto_partition_id < args.auto_partition_size:
            parser.error(
                f"--auto-partition-id must be in range [0, {args.auto_partition_size}), "
                f"but got {args.auto_partition_id}"
            )

    hw = HW_MAPPING[args.hw]
    suites = {s.strip() for s in args.suite.split(",") if s.strip()}

    # Registered tests under <repo>/test/registered/
    files = [
        f
        for f in glob.glob(
            str(REPO_ROOT / "test" / "registered" / "**" / "*.py"), recursive=True
        )
        # conftest.py / __init__.py are pytest+package structure, never
        # registered tests, and must not be listed as one.
        if os.path.basename(f) not in ("conftest.py", "__init__.py")
    ]
    all_tests = collect_tests(files)

    # Same filter as run_suite.py: backend + suite + nightly, enabled only.
    ci_tests = [
        t
        for t in all_tests
        if t.backend == hw
        and t.effective_suite in suites
        and t.nightly == args.nightly
        and t.disabled is None
    ]

    # Shard the selected tests across runners (LPT, same as run_suite.py).
    # NPU workflows rely on this to split one suite across matrix jobs.
    if args.auto_partition_size:
        ci_tests = auto_partition(
            ci_tests, args.auto_partition_id, args.auto_partition_size
        )

    with open(args.output, "w") as f:
        for t in ci_tests:
            f.write(t.filename + "\n")


if __name__ == "__main__":
    main()
