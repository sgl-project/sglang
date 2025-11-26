"""
Config-driven diffusion performance test with pytest parametrization.


If the actual run is significantly better than the baseline, the improved cases with their updated baseline will be printed
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    ONE_GPU_CASES_A,
    DiffusionTestCase,
)

logger = init_logger(__name__)


class TestDiffusionServerOneGpu(DiffusionServerBase):
    """Performance tests for 1-GPU diffusion cases."""

    @pytest.fixture(params=ONE_GPU_CASES_A, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 1-GPU test."""
        return request.param
