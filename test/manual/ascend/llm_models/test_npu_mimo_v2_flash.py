import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import MIMO_V2_FLASH_WEIGHTS_PATH
from sglang.test.test_utils import CustomTestCase


class TestMiMoV2FlashGraphWithMTP(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify the inference accuracy of MiMo-V2-Flash on GSM8K with cuda graph and MTP (speculative decoding).

    [Test Category] Model
    [Test Target] XiaomiMiMo/MiMo-V2-Flash
    [Test Config] Prefill+Decode, cuda graph enabled, EAGLE speculative decoding
    """

    model = MIMO_V2_FLASH_WEIGHTS_PATH
    accuracy = 0.9
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "16",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--enable-multi-layer-eagle",
    ]


if __name__ == "__main__":
    unittest.main()
