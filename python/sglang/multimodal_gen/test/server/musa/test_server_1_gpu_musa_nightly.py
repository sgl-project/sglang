"""
MUSA-specific 1-GPU diffusion performance tests for nightly suite.
"""

from __future__ import annotations

from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.musa.testcase_configs_musa import (
    ONE_GPU_NIGHTLY_MUSA_CASES,
)
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)


class TestDiffusionServerOneGpuMusaNightly(DiffusionServerBase):
    """Performance tests for 1-GPU diffusion cases on MUSA (nightly-only)."""

    case = diffusion_case_fixture(ONE_GPU_NIGHTLY_MUSA_CASES)
