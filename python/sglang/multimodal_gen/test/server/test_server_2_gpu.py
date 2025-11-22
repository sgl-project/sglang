"""
2 GPU Performance tests (A14B models) with --num-gpus 2 --ulysses-degree 2.
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.test.server.test_server_performance import (
    DiffusionPerformanceBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    TWO_GPU_CASES,
    DiffusionTestCase,
)


class TestDiffusionPerformanceTwoGpu(DiffusionPerformanceBase):
    """Performance tests for 2-GPU diffusion cases."""

    @pytest.fixture(params=TWO_GPU_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 2-GPU test."""
        return request.param
