import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    MIMO_V2_5_PRO_FP4_DFLASH_DRAFT_WEIGHTS_PATH,
    MIMO_V2_5_PRO_FP4_DFLASH_WEIGHTS_PATH,
)
from sglang.test.test_utils import CustomTestCase


class TestMiMoV25ProFP4GraphWithDFlash(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify the inference accuracy of MiMo-V2.5-Pro-FP4 on GSM8K with npu graph and DFlash speculative decoding.

    [Test Category] Model
    [Test Target] MiMo-V2.5-Pro-FP4-DFlash
    [Test Config] Prefill+Decode, npu graph enabled, DFLASH speculative decoding, mxfp4 quantization, dp attention
    """

    model = MIMO_V2_5_PRO_FP4_DFLASH_WEIGHTS_PATH
    accuracy = 0.9
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.87",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "8",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        MIMO_V2_5_PRO_FP4_DFLASH_DRAFT_WEIGHTS_PATH,
        "--speculative-num-draft-tokens",
        "8",
        "--dp-size",
        "2",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
    ]


if __name__ == "__main__":
    unittest.main()
