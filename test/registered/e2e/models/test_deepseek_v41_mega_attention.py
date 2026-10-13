import unittest

from test_deepseek_v41_flash_gb300 import DSV41FlashAccuracyMixin

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1100, stage="nightly", runner_config="4-gpu-gb300")


class TestDSV41FlashMegaDP4EP4DSpark(DSV41FlashAccuracyMixin, CustomTestCase):
    parallel_args = [
        "--attn-dp-size",
        "4",
        "--ep-size",
        "4",
        "--moe-a2a-backend",
        "none",
        "--dsv4-attn-backend",
        "flashmla_mega",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]


if __name__ == "__main__":
    unittest.main()
