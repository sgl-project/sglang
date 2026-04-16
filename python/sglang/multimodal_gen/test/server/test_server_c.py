"""
Config-driven diffusion performance test with pytest parametrization.
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    ONE_GPU_CASES_C,
    DiffusionTestCase,
)

logger = init_logger(__name__)


class TestDiffusionServerOneGpuB200(DiffusionServerBase):
    """B200-targeted CI tests for 1-GPU ModelOpt diffusion cases."""

    @pytest.fixture(params=ONE_GPU_CASES_C, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 1-GPU B200 test."""
        return request.param
