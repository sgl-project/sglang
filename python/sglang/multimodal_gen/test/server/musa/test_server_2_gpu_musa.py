"""
MUSA-specific 2-GPU diffusion performance tests.
"""

from __future__ import annotations

from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.musa.testcase_configs_musa import (
    TWO_GPU_MUSA_CASES,
)
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)


class TestDiffusionServerTwoGpuMusa(DiffusionServerBase):
    """Performance tests for 2-GPU diffusion cases on MUSA."""

    case = diffusion_case_fixture(TWO_GPU_MUSA_CASES)
