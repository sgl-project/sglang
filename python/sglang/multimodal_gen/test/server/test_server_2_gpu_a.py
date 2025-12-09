"""
2 GPU tests
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    TWO_GPU_CASES_A,
    DiffusionTestCase,
)


class TestDiffusionServerTwoGpu(DiffusionServerBase):
    """Performance tests for 2-GPU diffusion cases."""

    @pytest.fixture(params=TWO_GPU_CASES_A, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 2-GPU test."""
        return request.param
