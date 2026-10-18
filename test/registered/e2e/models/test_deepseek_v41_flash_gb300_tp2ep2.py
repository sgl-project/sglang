"""DeepSeek-V4.1-Flash + DSPARK accuracy on GB300 GPUs: TP2 + EP2.

GSM8K and MMMU-Pro at TP2, which exercises the fused all-reduce + mHC, WO-A
and swap-AB decode paths.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dsv41_flash_common import DSV41FlashAccuracyMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="base-c", runner_config="4-gpu-gb300")


class TestDSV41FlashTP2EP2DSpark(DSV41FlashAccuracyMixin, CustomTestCase):
    """TP2 exercises the fused all-reduce + mHC, WO-A and swap-AB decode paths."""

    tp_size = 2
    parallel_args = ["--attn-dp-size", "1", "--ep-size", "2"]
    # Engram tables on the GPU leave no KV cache at TP2 under mem-fraction 0.8.
    server_env = {
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank",
    }


if __name__ == "__main__":
    unittest.main()
