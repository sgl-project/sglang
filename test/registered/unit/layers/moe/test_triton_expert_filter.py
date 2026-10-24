from typing import Optional

import pytest

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import _topk_ids_may_be_nonlocal
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "num_experts,num_local_experts,num_weight_experts,expected",
    [
        (8, 8, 8, False),
        # Weights holding a different expert count than the config claims.
        (8, 8, 7, True),
        (8, 4, 4, True),
        (None, 8, 8, True),
        # No weight count: the config alone decides.
        (8, 8, None, False),
        (8, 4, None, True),
    ],
)
def test_topk_ids_may_be_nonlocal(
    num_experts: Optional[int],
    num_local_experts: int,
    num_weight_experts: Optional[int],
    expected: bool,
) -> None:
    config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
    )

    assert _topk_ids_may_be_nonlocal(config, num_weight_experts) is expected
