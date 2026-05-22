"""
MUSA-specific 2-GPU diffusion performance test.
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.musa.testcase_configs_musa import (
    TWO_GPU_MUSA_CASES_A,
)
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase

logger = init_logger(__name__)


class TestDiffusionServerTwoGpuMusaA(DiffusionServerBase):
    """Performance tests for 2-GPU diffusion cases on MUSA."""

    @pytest.fixture(params=TWO_GPU_MUSA_CASES_A, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 2-GPU MUSA test."""
        return request.param
