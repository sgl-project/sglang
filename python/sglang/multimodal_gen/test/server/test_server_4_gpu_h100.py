"""PR-blocking diffusion smoke tests that require four H100 GPUs."""

from __future__ import annotations

from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.gpu_cases import (
    MINIMAX_H3_FOUR_GPU_H100_CASES,
)
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)


class TestDiffusionServerFourGpuH100(DiffusionServerBase):
    case = diffusion_case_fixture(MINIMAX_H3_FOUR_GPU_H100_CASES)
