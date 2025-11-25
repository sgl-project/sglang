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

PER_COMMIT_SUITES = {
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


def filter_tests(
    ci_tests: List[CIRegistry], hw: HWBackend, suite: str, nightly: bool = False
) -> List[CIRegistry]:
    ci_tests = [
        t
        for t in ci_tests
        if t.backend == hw and t.suite == suite and t.nightly == nightly
    ]

    ret = []
    disabled_tests = []

    for t in ci_tests:
        if not nightly:
            assert (
                t.suite in PER_COMMIT_SUITES[hw]
            ), f"Unknown stage {t.suite} for backend {hw}"
        else:
            raise NotImplementedError("Nightly tests are not implemented yet.")

        if t.disabled is None:
            ret.append(t)
            print(f"Including test {t.filename}")
        else:
            print(f"Skipping disabled test {t.filename} due to: {t.disabled}")

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


def run_a_suite(args):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    nightly = args.nightly
    auto_partition_id = args.auto_partition_id
    auto_partition_size = args.auto_partition_size

    files = glob.glob("registered/**/*.py", recursive=True)
    ci_tests = filter_tests(collect_tests(files), hw, suite, nightly)
    test_files = [TestFile(t.filename, t.est_time) for t in ci_tests]

    if not test_files:
        raise ValueError(
            f"No tests found for hw={hw.name}, suite={suite}, nightly={nightly}"
        )

    if auto_partition_size:
        test_files = auto_partition(test_files, auto_partition_id, auto_partition_size)

    print(
        f"Running {len(test_files)} test(s) for hw={hw.name}, suite={suite}, nightly={nightly}"
    )

    run_unittest_files(
        test_files,
        timeout_per_file=1200,
        continue_on_error=nightly,
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
    parser.add_argument("--nightly", action="store_true")
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
    run_a_suite(args)


if __name__ == "__main__":
    main()
