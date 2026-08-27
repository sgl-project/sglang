"""
MUSA-specific diffusion performance test (1-GPU).
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.musa.testcase_configs_musa import (
    ONE_GPU_MUSA_CASES_A,
)
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase

logger = init_logger(__name__)


class TestDiffusionServerOneGpuMusaImage(DiffusionServerBase):
    """Performance tests for 1-GPU diffusion cases on MUSA"""

    @pytest.fixture(params=ONE_GPU_MUSA_CASES_A, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 1-GPU MUSA test."""
        return request.param
