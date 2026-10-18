"""DeepSeek-V4.1-Flash + DSPARK accuracy on four GB300 GPUs: DP-attention4 + EP4.

The DP+EP recipe is intentionally exercised rather than skipped: startup or
accuracy failures must be visible in CI while this path is being validated.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dsv41_flash_common import DSV41FlashAccuracyMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="base-c", runner_config="4-gpu-gb300")


class TestDSV41FlashDP4EP4DSpark(DSV41FlashAccuracyMixin, CustomTestCase):
    """DP attention and EP MoE using the built-in gather/reduce path.

    DSPARK with DP attention does not support the DeepEP backend.
    """

    parallel_args = [
        "--attn-dp-size",
        "4",
        "--ep-size",
        "4",
        "--enable-dp-lm-head",
        "--moe-a2a-backend",
        "none",
    ]


if __name__ == "__main__":
    unittest.main()
