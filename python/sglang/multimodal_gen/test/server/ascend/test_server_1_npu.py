"""
Config-driven diffusion performance test with pytest parametrization.


If the actual run is significantly better than the baseline, the improved cases with their updated baseline will be printed
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.ascend.testcase_configs_npu import ONE_NPU_CASES
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase
from sglang.test.ci.ci_register import register_npu_ci

logger = init_logger(__name__)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestDiffusionServerOneNpu(DiffusionServerBase):
    """Performance tests for 1-NPU diffusion cases."""

    @pytest.fixture(params=ONE_NPU_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 1-NPU test."""
        return request.param
