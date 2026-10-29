import unittest

from test_paged_experts import _PagedMatchesUnpaged

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    CustomTestCase,
)

register_cuda_ci(est_time=113, stage="extra-a", runner_config="1-gpu-large")


class TestPagedExpertsMxfp4(_PagedMatchesUnpaged, CustomTestCase):
    """MXFP4 experts with biases (gpt-oss) on the Marlin runner, which pads them: loaded from
    stacked checkpoint tensors, compared like GPTQ."""

    model = "openai/gpt-oss-20b"
    server_args = ("--moe-runner-backend", "marlin")
    exact = False
    median_abs_tolerance = 0.1


if __name__ == "__main__":
    unittest.main()
