import argparse
import glob
import sys
from typing import List

from sglang.test.ci.ci_register import CIRegistry, HWBackend, collect_tests
from sglang.test.ci.ci_utils import TestFile, run_unittest_files

HW_MAPPING = {
    "cpu": HWBackend.CPU,
    "cuda": HWBackend.CUDA,
    "amd": HWBackend.AMD,
    "npu": HWBackend.NPU,
}

# Per-commit test suites (run on every PR)
PER_COMMIT_SUITES = {
    HWBackend.CPU: ["default"],
    HWBackend.AMD: ["stage-a-test-1"],
    HWBackend.CUDA: ["stage-a-test-1", "stage-b-test-small-1-gpu"],
    HWBackend.NPU: [],
}

# Nightly test suites (run nightly, organized by GPU configuration)
NIGHTLY_SUITES = {
    HWBackend.CUDA: [
        "nightly-1-gpu",
        "nightly-2-gpu",
        "nightly-4-gpu",
        "nightly-4-gpu-b200",
        "nightly-8-gpu",
        "nightly-8-gpu-h200",
        "nightly-8-gpu-h20",
        "nightly-8-gpu-b200",
    ],
    HWBackend.AMD: ["nightly-amd"],
    HWBackend.CPU: [],
    HWBackend.NPU: [
        "nightly-1-npu-a3",
        "nightly-2-npu-a3",
        "nightly-4-npu-a3",
        "nightly-16-npu-a3",
    ],
}


def filter_tests(
    ci_tests: List[CIRegistry], hw: HWBackend, suite: str, nightly: bool = False
) -> List[CIRegistry]:
    ci_tests = [
        t
        for t in ci_tests
        if t.backend == hw and t.suite == suite and t.nightly == nightly
    ]

    ret = []
    valid_suites = (
        NIGHTLY_SUITES.get(hw, []) if nightly else PER_COMMIT_SUITES.get(hw, [])
    )

    if suite not in valid_suites:
        print(
            f"Warning: Unknown suite {suite} for backend {hw.name}, nightly={nightly}"
        )

    for t in ci_tests:
        if t.disabled is None:
            ret.append(t)
            print(f"Including test {t.filename}")
        else:
            print(f"Skipping disabled test {t.filename} due to: {t.disabled}")

    return ret


def auto_partition(files, rank, size):
    """
    Partition files into size sublists with approximately equal sums of estimated times
    using a greedy algorithm (LPT heuristic), and return the partition for the specified rank.
    """
    if not files or size <= 0:
        return []

    # Sort files by estimated_time in descending order (LPT heuristic)
    sorted_files = sorted(files, key=lambda f: f.estimated_time, reverse=True)

    partitions = [[] for _ in range(size)]
    partition_sums = [0.0] * size

    # Greedily assign each file to the partition with the smallest current total time
    for file in sorted_files:
        min_sum_idx = min(range(size), key=partition_sums.__getitem__)
        partitions[min_sum_idx].append(file)
        partition_sums[min_sum_idx] += file.estimated_time

    if rank < size:
        return partitions[rank]
    return []


def run_a_suite(args):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    nightly = args.nightly
    auto_partition_id = args.auto_partition_id
    auto_partition_size = args.auto_partition_size

    # Temporary: search broadly for nightly tests during migration to registered/
    if nightly:
        files = glob.glob("**/*.py", recursive=True)
        sanity_check = False  # Allow files without registration during migration
    else:
        files = glob.glob("registered/**/*.py", recursive=True)
        sanity_check = (
            True  # Strict: all registered files must have proper registration
        )

    ci_tests = filter_tests(
        collect_tests(files, sanity_check=sanity_check), hw, suite, nightly
    )
    test_files = [TestFile(t.filename, t.est_time) for t in ci_tests]

    if not test_files:
        print(f"No tests found for hw={hw.name}, suite={suite}, nightly={nightly}")
        print("This is expected during incremental migration. Skipping.")
        return 0

    if auto_partition_size:
        test_files = auto_partition(test_files, auto_partition_id, auto_partition_size)

    print(
        f"Running {len(test_files)} test(s) for hw={hw.name}, suite={suite}, nightly={nightly}"
    )

    return run_unittest_files(
        test_files,
        timeout_per_file=args.timeout_per_file,
        continue_on_error=args.continue_on_error,
        enable_retry=args.enable_retry,
        max_attempts=args.max_attempts,
        retry_wait_seconds=args.retry_wait_seconds,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run CI test suites from test/registered/"
    )
    parser.add_argument(
        "--hw",
        type=str,
        choices=HW_MAPPING.keys(),
        required=True,
        help="Hardware backend to run tests on.",
    )
    parser.add_argument("--suite", type=str, required=True, help="Test suite to run.")
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Run nightly tests instead of per-commit tests.",
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1200,
        help="The time limit for running one file in seconds (default: 1200).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails (default: False, useful for nightly tests).",
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
        "--enable-retry",
        action="store_true",
        default=False,
        help="Enable smart retry for accuracy/performance assertion failures (not code errors)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum number of attempts per file including initial run (default: 2)",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=int,
        default=60,
        help="Seconds to wait between retries (default: 60)",
    )
    args = parser.parse_args()

    # Validate auto-partition arguments
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

    exit_code = run_a_suite(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
