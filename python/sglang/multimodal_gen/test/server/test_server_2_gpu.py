"""
2 GPU tests
"""

from __future__ import annotations

from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.gpu_cases import TWO_GPU_CASES
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)


class TestDiffusionServerTwoGpu(DiffusionServerBase):
    """Performance tests for 2-GPU diffusion cases."""

    case = diffusion_case_fixture(TWO_GPU_CASES)
