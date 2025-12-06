"""
2 GPU tests
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    TWO_GPU_CASES_A,
    DiffusionTestCase,
)


class TestDiffusionServerTwoGpu(DiffusionServerBase):
    """Performance tests for 2-GPU diffusion cases."""

    @pytest.fixture(params=TWO_GPU_CASES_A, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 2-GPU test."""
        test_case = request.param
        # Skip video tests due to missing openai client support for videos API
        # The openai==2.6.1 client doesn't have a 'videos' attribute
        # TODO: Fix this by updating openai or creating custom client wrapper
        if test_case.server_args.modality == "video":
            pytest.skip(f"Skipping video test {test_case.id}: openai client doesn't support videos API")
        return test_case
