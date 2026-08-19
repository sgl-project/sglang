from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import pytest

if TYPE_CHECKING:
    from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase


def diffusion_case_fixture(cases: Sequence[DiffusionTestCase]):
    @pytest.fixture(params=cases, ids=lambda case: case.id)
    def case(self, request) -> DiffusionTestCase:
        return request.param

    return case
