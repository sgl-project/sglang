"""Rerun the UnifiedRadixCache topic test modules with the Rust TreeCore, including the centralized Python fallbacks."""

import unittest

import test_unified_radix_cache_core
import test_unified_radix_cache_hicache
import test_unified_radix_cache_insert_walk
import test_unified_radix_cache_kv_events
import test_unified_radix_cache_load_back
import test_unified_radix_cache_storage

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache import fixtures

register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")

_SHARED_SUITES = (
    test_unified_radix_cache_core,
    test_unified_radix_cache_hicache,
    test_unified_radix_cache_insert_walk,
    test_unified_radix_cache_kv_events,
    test_unified_radix_cache_load_back,
    test_unified_radix_cache_storage,
)


class RustBackendSuite(unittest.TestSuite):
    """Scope the test backend to this suite without polluting discovery."""

    def run(self, result, debug=False):
        previous = fixtures.TREE_CORE_TEST_BACKEND
        fixtures.TREE_CORE_TEST_BACKEND = "rust"
        try:
            return super().run(result, debug)
        finally:
            fixtures.TREE_CORE_TEST_BACKEND = previous


def load_tests(loader, standard_tests, pattern):
    """Load every topic module under the Rust backend override."""
    return RustBackendSuite(
        loader.loadTestsFromModule(module) for module in _SHARED_SUITES
    )


if __name__ == "__main__":
    unittest.main()
