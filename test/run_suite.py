import argparse
import glob
from typing import List

from sglang.test.ci.ci_register import CIRegistry, HWBackend, collect_tests
from sglang.test.ci.ci_utils import TestFile, run_unittest_files

HW_MAPPING = {
    "cpu": HWBackend.CPU,
    "cuda": HWBackend.CUDA,
    "amd": HWBackend.AMD,
}

LABEL_MAPPING = {
    HWBackend.CPU: ["default"],
    HWBackend.AMD: ["stage-a-test-1"],
    HWBackend.CUDA: [
        "stage-a-test-1",
        "stage-a-test-2",
        "stage-b-test-small-1-gpu",
        "stage-b-test-large-1-gpu",
        # "stage-b-test-large-2-gpu",  # TODO: Uncomment when multi-GPU tests are migrated
        "stage-c-test-large-1-gpu",
        # "stage-c-test-large-2-gpu",  # TODO: Uncomment when multi-GPU tests are migrated
        # "stage-c-test-large-4-gpu",  # TODO: Uncomment when multi-GPU tests are migrated
    ],
}


def _filter_tests(
    ci_tests: List[CIRegistry],
    hw: HWBackend,
    suite: str,
    nightly: bool,
) -> List[CIRegistry]:
    """Filter tests by hardware backend, suite, and nightly flag.

    Args:
        ci_tests: List of CI registry entries.
        hw: Hardware backend to filter by.
        suite: Test suite name to filter by.
        nightly: If True, include nightly tests. If False, exclude nightly-only tests.

    Returns:
        Filtered list of CI registry entries.
    """
    ret = []
    disabled_tests = []

    for t in ci_tests:
        # Filter by hardware backend
        if t.backend != hw:
            continue

        # Filter by suite
        if t.suite != suite:
            continue

        # Validate suite is in LABEL_MAPPING
        assert t.suite in LABEL_MAPPING[hw], f"Unknown stage {t.suite} for backend {hw}"

        # Filter by nightly flag
        # - If running nightly (nightly=True): include all tests (nightly and per-commit)
        # - If running per-commit (nightly=False): exclude nightly-only tests
        if not nightly and t.nightly:
            continue

        # Check if test is disabled
        if t.disabled:
            disabled_tests.append((t.filename, t.disabled))
            continue

        ret.append(t)

    # Warn about disabled tests
    if disabled_tests:
        print(f"\nSkipping {len(disabled_tests)} disabled test(s):")
        for filename, reason in disabled_tests:
            print(f"  - {filename}: {reason}")
        print()

    return ret


def auto_partition(files, rank, size):
    """
    Partition files into size sublists with approximately equal sums of estimated times
    using stable sorting, and return the partition for the specified rank.
    """
    weights = [f.estimated_time for f in files]

    if not weights or size <= 0 or size > len(weights):
        return []

    # Create list of (weight, original_index) tuples
    indexed_weights = [(w, -i) for i, w in enumerate(weights)]
    # Stable sort in descending order by weight
    indexed_weights = sorted(indexed_weights, reverse=True)

    # Extract original indices (negate back to positive)
    indexed_weights = [(w, -i) for w, i in indexed_weights]

    # Initialize partitions and their sums
    partitions = [[] for _ in range(size)]
    sums = [0.0] * size

    # Greedy approach: assign each weight to partition with smallest current sum
    for weight, idx in indexed_weights:
        min_sum_idx = sums.index(min(sums))
        partitions[min_sum_idx].append(idx)
        sums[min_sum_idx] += weight

    # Return the files corresponding to the indices in the specified rank's partition
    indices = partitions[rank]
    return [files[i] for i in indices]


def run_suite(
    hw: HWBackend,
    suite: str,
    nightly: bool,
    auto_partition_id: int = None,
    auto_partition_size: int = None,
):
    """Run tests from the registered/ directory.

    Args:
        hw: Hardware backend to run tests on.
        suite: Test suite name to run.
        nightly: If True, include nightly tests.
        auto_partition_id: Partition ID for load balancing.
        auto_partition_size: Number of partitions for load balancing.
    """
    files = glob.glob("registered/**/*.py", recursive=True)
    ci_tests = _filter_tests(collect_tests(files), hw, suite, nightly)
    test_files = [TestFile(t.filename, t.est_time) for t in ci_tests]

    if auto_partition_size:
        test_files = auto_partition(test_files, auto_partition_id, auto_partition_size)

    if not test_files:
        print(f"No tests found for hw={hw.name}, suite={suite}, nightly={nightly}")
        return

    print(f"Running {len(test_files)} test(s) for hw={hw.name}, suite={suite}, nightly={nightly}")

    run_unittest_files(
        test_files,
        timeout_per_file=1200,
        continue_on_error=nightly,  # Continue on error for nightly, fail fast for per-commit
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run CI test suites from test/registered/"
    )
    parser.add_argument(
        "--hw",
        type=str,
        choices=["cpu", "cuda", "amd"],
        required=True,
        help="Hardware backend to run tests on.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="Test suite to run (e.g., stage-a-test-1, nightly-1-gpu).",
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        default=False,
        help="Include nightly tests. If not set, only per-commit tests are run.",
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
    args = parser.parse_args()
    hw = HW_MAPPING[args.hw]
    run_suite(hw, args.suite, args.nightly, args.auto_partition_id, args.auto_partition_size)


if __name__ == "__main__":
    main()
