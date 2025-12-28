"""
End-to-end functional correctness tests for 2-GPU diffusion server.
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.test_server_correctness import (
    CorrectnessTestMixin,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    CORRECTNESS_2_GPU_CASES,
    DiffusionTestCase,
)

logger = init_logger(__name__)


class TestDiffusionServerTwoGpuCorrectness(CorrectnessTestMixin, DiffusionServerBase):
    """
    Functional correctness tests for 2-GPU diffusion cases.
    Inherits shared functional logic from CorrectnessTestMixin.
    """

    @pytest.fixture(params=CORRECTNESS_2_GPU_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 2-GPU correctness test."""
        return request.param
