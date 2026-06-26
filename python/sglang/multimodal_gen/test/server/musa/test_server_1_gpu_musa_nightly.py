"""
MUSA-specific 1-GPU diffusion performance tests for nightly suite.
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.musa.testcase_configs_musa import (
    ONE_GPU_NIGHTLY_MUSA_CASES,
)
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase

logger = init_logger(__name__)


class TestDiffusionServerOneGpuMusaNightly(DiffusionServerBase):
    """Performance tests for 1-GPU diffusion cases on MUSA (nightly-only)."""

    @pytest.fixture(params=ONE_GPU_NIGHTLY_MUSA_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 1-GPU MUSA nightly test."""
        return request.param
