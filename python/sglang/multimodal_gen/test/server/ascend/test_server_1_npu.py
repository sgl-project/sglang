"""
Config-driven diffusion performance test with pytest parametrization.


If the actual run is significantly better than the baseline, the improved cases with their updated baseline will be printed
"""

from __future__ import annotations

from sglang.multimodal_gen.test.server.ascend.testcase_configs_npu import ONE_NPU_CASES
from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)


class TestDiffusionServerOneNpu(DiffusionServerBase):
    """Performance tests for 1-NPU diffusion cases."""

    case = diffusion_case_fixture(ONE_NPU_CASES)
