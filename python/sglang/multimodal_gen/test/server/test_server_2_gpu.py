"""
2 GPU Performance tests (A14B models) with --num-gpus 2 --ulysses-degree 2.
"""

from __future__ import annotations

import pytest

# Import the base class to reuse logic
from sglang.multimodal_gen.test.server.test_server_performance import (
    TestDiffusionPerformanceBase,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    TWO_GPU_CASES,
    DiffusionTestCase,
)


class TestDiffusionPerformanceTwoGpu(TestDiffusionPerformanceBase):
    """Performance tests for 2-GPU diffusion cases."""

    @pytest.fixture(params=TWO_GPU_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 2-GPU test."""
        return request.param
