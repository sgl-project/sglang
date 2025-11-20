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
    HWBackend.CUDA: ["stage-a-test-1"],
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


def run_per_commit(hw: HWBackend, suite: str):
    files = glob.glob("per_commit/**/*.py", recursive=True)
    ci_tests = _filter_tests(collect_tests(files), hw, suite)
    test_files = [TestFile(t.filename, t.est_time) for t in ci_tests]

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
    args = parser.parse_args()
    hw = HW_MAPPING[args.hw]
    run_per_commit(hw, args.suite)


if __name__ == "__main__":
    main()
