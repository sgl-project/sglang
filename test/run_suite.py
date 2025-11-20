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
    ci_tests: List[CIRegistry], hw: HWBackend, suite: str
) -> List[CIRegistry]:
    ci_tests = [t for t in ci_tests if t.backend == hw]
    ret = []
    for t in ci_tests:
        assert t.suite in LABEL_MAPPING[hw], f"Unknown stage {t.suite} for backend {hw}"
        if t.suite == suite:
            ret.append(t)
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


def run_per_commit(
    hw: HWBackend,
    suite: str,
    auto_partition_id: int = None,
    auto_partition_size: int = None,
):
    files = glob.glob("per_commit/**/*.py", recursive=True)
    ci_tests = _filter_tests(collect_tests(files), hw, suite)
    test_files = [TestFile(t.filename, t.est_time) for t in ci_tests]

    if auto_partition_size:
        test_files = auto_partition(test_files, auto_partition_id, auto_partition_size)

    run_unittest_files(
        test_files,
        timeout_per_file=1200,
        continue_on_error=False,
    )


def main():
    parser = argparse.ArgumentParser()
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
        help="Test suite to run.",
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
    run_per_commit(hw, args.suite, args.auto_partition_id, args.auto_partition_size)


if __name__ == "__main__":
    main()
