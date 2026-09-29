import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import KIMI_K3_W4A8_INT_MOE_WEIGHTS_PATH
from sglang.test.test_utils import CustomTestCase


class TestKimiK3MixedWithHiCacheL2(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify the inference accuracy of Kimi-K3 (MLA + KDA hybrid) on GSM8K
    with mixed (non-PD) serving and HiCache L2 cache on NPU.

    [Test Category] HiCache
    [Test Target] Kimi-K3 (MLA + KDA hybrid, mamba layers)
    [Test Config] Mixed deployment, NPU, HiCache L2 (kernel_ascend IO backend)
    """

    model = KIMI_K3_W4A8_INT_MOE_WEIGHTS_PATH
    accuracy = 0.9
    other_args = [
        "--trust-remote-code",
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--quantization",
        "modelslim",
        "--dtype",
        "bfloat16",
        "--tp-size",
        "64",
        "--enable-dp-attention",
        "--dp-size",
        "4",
        "--enable-dp-lm-head",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--mem-fraction-static",
        "0.75",
        "--max-mamba-cache-size",
        "240",
        "--enable-hierarchical-cache",
        "--hicache-io-backend",
        "kernel_ascend",
        "--enable-cache-report",
        "--hicache-ratio",
        "4.0",
    ]


if __name__ == "__main__":
    unittest.main()
