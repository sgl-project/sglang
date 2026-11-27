"""DeepSeek-V4.1-Flash + DSPARK MMMUPro accuracy with TP2 + EP2."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dsv41_flash_common import TP2_EP2_ARGS, DSV41FlashMMMUProMixin
from sglang.test.test_utils import CustomTestCase

# ci-cost-override: Split the existing base accuracy gate by dataset so each
# file can be partitioned independently; each now includes a server launch.
# Conservative initial estimate; live CI timings will calibrate each file.
register_cuda_ci(est_time=900, stage="base-c", runner_config="2-gpu-gb300")


class TestDSV41FlashTP2EP2DSparkMMMUPro(DSV41FlashMMMUProMixin, CustomTestCase):
    parallel_args = TP2_EP2_ARGS


if __name__ == "__main__":
    unittest.main()
