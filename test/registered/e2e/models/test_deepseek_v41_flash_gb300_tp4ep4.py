"""DeepSeek-V4.1-Flash + DSPARK accuracy on four GB300 GPUs: TP4 + EP4.

GSM8K and MMMU-Pro with tensor-parallel attention and expert-parallel MoE,
without DP attention.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dsv41_flash_common import TP4_EP4_ARGS, DSV41FlashAccuracyMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="base-c", runner_config="4-gpu-gb300")


class TestDSV41FlashTP4EP4DSpark(DSV41FlashAccuracyMixin, CustomTestCase):
    """Tensor-parallel attention and expert-parallel MoE, without DP attention."""

    parallel_args = TP4_EP4_ARGS


if __name__ == "__main__":
    unittest.main()
