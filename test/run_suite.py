import glob
from typing import List

from sglang.test.ci.ci_register import CIRegistry, HWBackend, collect_tests
from sglang.test.ci.ci_utils import TestFile, run_unittest_files

LABEL_MAPPING = {HWBackend.CUDA: ["stage-a-test-1"]}


def _filter_tests(
    ci_tests: List[CIRegistry], hw: HWBackend, suite: str
) -> List[CIRegistry]:
    ci_tests = [t for t in ci_tests if t.backend == hw]
    ret = []
    for t in ci_tests:
        assert t.stage in LABEL_MAPPING[hw], f"Unknown stage {t.stage} for backend {hw}"
        if t.stage == suite:
            ret.append(t)
    return ret


def run_per_commit(hw: HWBackend, suite: str):
    files = glob.glob("per_commit/**/*.py", recursive=True)
    ci_tests = _filter_tests(collect_tests(files), hw, suite)
    test_files = [TestFile(t.filename, t.estimation_time) for t in ci_tests]

    run_unittest_files(
        test_files,
        timeout_per_file=1200,
        continue_on_error=False,
    )


def main():
    run_per_commit(HWBackend.CUDA, "stage-a-test-1")


if __name__ == "__main__":
    main()
