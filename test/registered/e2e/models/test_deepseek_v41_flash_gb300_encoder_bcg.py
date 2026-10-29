"""DeepSeek-V4.1-Flash + DSPARK accuracy on GB300 with encoder bounded replay.

Run GSM8K and MMMU-Pro with TP2 + EP2, encoder SWA bounded replay, and
breakable prefill graphs using the shared accuracy recipe.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dsv41_flash_common import TP2_EP2_ARGS, DSV41FlashAccuracyMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="base-c", runner_config="2-gpu-gb300")


class TestDSV41FlashEncoderBoundedBCG(DSV41FlashAccuracyMixin, CustomTestCase):
    parallel_args = TP2_EP2_ARGS
    graph_args = [
        *DSV41FlashAccuracyMixin.graph_args,
        "--enable-encoder-swa-bounded-replay",
    ]


if __name__ == "__main__":
    unittest.main()
