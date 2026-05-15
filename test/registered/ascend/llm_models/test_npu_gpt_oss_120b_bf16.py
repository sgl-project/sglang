import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import GPT_OSS_120B_BF16_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-8-npu-a3",
    nightly=True,
    disabled="https://github.com/Ascend/sglang/issues/122",
)


class TestAFM(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the eigen-ai-labs/gpt-oss-120b-bf16 model on the GSM8K dataset is no less than 0.

    [Test Category] Model
    [Test Target] eigen-ai-labs/gpt-oss-120b-bf16
    """

    model = GPT_OSS_120B_BF16_WEIGHTS_PATH
    accuracy = 0
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.7",
        "--attention-backend",
        "ascend",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--max-running-requests",
        "32",
        "--chunked-prefill-size",
        "3276800",
        "--max-prefill-tokens",
        "32768",
        "--watchdog-timeout",
        "9000",
        "--tp-size",
        "8",
        "--sampling-backend",
        "ascend",
    ]


if __name__ == "__main__":
    unittest.main()
