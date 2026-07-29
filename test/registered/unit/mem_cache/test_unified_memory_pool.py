"""Unit tests for the unified memory pool."""

import inspect
import unittest

from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedHybridReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnifiedHybridReqToTokenPool(CustomTestCase):
    def test_mamba_pool_override_accepts_parent_keyword_arguments(self):
        parent_parameters = inspect.signature(
            HybridReqToTokenPool._init_mamba_pool
        ).parameters
        override_parameters = inspect.signature(
            UnifiedHybridReqToTokenPool._init_mamba_pool
        ).parameters

        self.assertLessEqual(
            parent_parameters.keys(),
            override_parameters.keys(),
        )


if __name__ == "__main__":
    unittest.main()
