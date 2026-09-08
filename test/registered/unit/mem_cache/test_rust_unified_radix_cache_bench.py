"""Run the UnifiedRadixCache benchmark/fuzz suite with the Rust TreeCore."""

import unittest

import test_unified_radix_cache_bench as shared_suite

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")


class RustBackendSuite(unittest.TestSuite):
    """Scope the backend override to this suite and restore it afterward."""

    def run(self, result, debug=False):
        previous = shared_suite._TREE_CORE_TEST_BACKEND
        shared_suite._TREE_CORE_TEST_BACKEND = "rust"
        try:
            return super().run(result, debug)
        finally:
            shared_suite._TREE_CORE_TEST_BACKEND = previous


def load_tests(loader, standard_tests, pattern):
    return RustBackendSuite(loader.loadTestsFromModule(shared_suite))


if __name__ == "__main__":
    unittest.main()
