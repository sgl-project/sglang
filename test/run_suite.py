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
    HWBackend.CUDA: ["stage-a-test-1"],
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


def run_a_suite(hw: HWBackend, suite: str, nightly: bool = False):
    files = glob.glob("registered/**/*.py", recursive=True)
    ci_tests = filter_tests(collect_tests(files), hw, suite, nightly)
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
        choices=HW_MAPPING.keys(),
        required=True,
        help="Hardware backend to run tests on.",
    )
    parser.add_argument("--suite", type=str, required=True, help="Test suite to run.")
    parser.add_argument("--nightly", action="store_true")
    args = parser.parse_args()
    hw = HW_MAPPING[args.hw]
    run_a_suite(hw, args.suite, args.nightly)


if __name__ == "__main__":
    main()
