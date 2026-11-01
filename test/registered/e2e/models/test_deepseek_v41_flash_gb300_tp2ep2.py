"""DeepSeek-V4.1-Flash + DSPARK accuracy on two GB300 GPUs: TP2 + EP2.

GSM8K and MMMU-Pro with tensor-parallel attention and expert-parallel MoE,
without DP attention.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dsv41_flash_common import TP2_EP2_ARGS, DSV41FlashAccuracyMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="base-c", runner_config="2-gpu-gb300")


class TestDSV41FlashTP2EP2DSpark(DSV41FlashAccuracyMixin, CustomTestCase):
    """Tensor-parallel attention and expert-parallel MoE, without DP attention."""

    parallel_args = TP2_EP2_ARGS


if __name__ == "__main__":
    unittest.main()
