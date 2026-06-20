"""
Config-driven diffusion performance test with pytest parametrization.
"""

from __future__ import annotations

from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.gpu_cases import ONE_GPU_B200_CASES
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)


class TestDiffusionServerOneGpuB200(DiffusionServerBase):
    """B200-targeted CI tests for 1-GPU Blackwell-only diffusion cases."""

    case = diffusion_case_fixture(ONE_GPU_B200_CASES)
