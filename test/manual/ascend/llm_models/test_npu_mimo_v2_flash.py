import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import MIMO_V2_FLASH_WEIGHTS_PATH
from sglang.test.test_utils import CustomTestCase

_BASE_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.8",
    "--attention-backend",
    "ascend",
    "--tp-size",
    "16",
]

_MTP_ARGS = [
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


class _Base(GSM8KAscendMixin, CustomTestCase):
    model = MIMO_V2_FLASH_WEIGHTS_PATH
    accuracy = 0.9
    other_args = _BASE_ARGS


class TestMiMoV2FlashNoGraph(_Base):
    """Testcase: Verify the inference accuracy of MiMo-V2-Flash on GSM8K without cuda graph.

    [Test Category] Model
    [Test Target] XiaomiMiMo/MiMo-V2-Flash
    [Test Config] Prefill+Decode, --disable-cuda-graph
    """

    other_args = _BASE_ARGS + ["--disable-cuda-graph"]


class TestMiMoV2FlashNoGraphWithMTP(_Base):
    """Testcase: Verify the inference accuracy of MiMo-V2-Flash on GSM8K without cuda graph and with MTP (speculative decoding).

    [Test Category] Model
    [Test Target] XiaomiMiMo/MiMo-V2-Flash
    [Test Config] Prefill+Decode, --disable-cuda-graph, EAGLE speculative decoding
    """

    other_args = _BASE_ARGS + ["--disable-cuda-graph"] + _MTP_ARGS


class TestMiMoV2FlashGraph(_Base):
    """Testcase: Verify the inference accuracy of MiMo-V2-Flash on GSM8K with cuda graph enabled.

    [Test Category] Model
    [Test Target] XiaomiMiMo/MiMo-V2-Flash
    [Test Config] Prefill+Decode, cuda graph enabled
    """

    other_args = _BASE_ARGS


class TestMiMoV2FlashGraphWithMTP(_Base):
    """Testcase: Verify the inference accuracy of MiMo-V2-Flash on GSM8K with cuda graph and MTP (speculative decoding).

    [Test Category] Model
    [Test Target] XiaomiMiMo/MiMo-V2-Flash
    [Test Config] Prefill+Decode, cuda graph enabled, EAGLE speculative decoding
    """

    other_args = _BASE_ARGS + _MTP_ARGS


if __name__ == "__main__":
    unittest.main()
