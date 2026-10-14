"""
Config-driven diffusion canary tests for the 1-GPU 5090 PR runner.
"""

from __future__ import annotations

from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.gpu_cases import ONE_GPU_5090_CASES
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)


class TestDiffusionServerOneGpu5090(DiffusionServerBase):
    """Canary tests for lightweight 1-GPU diffusion cases on 5090."""

    case = diffusion_case_fixture(ONE_GPU_5090_CASES)
