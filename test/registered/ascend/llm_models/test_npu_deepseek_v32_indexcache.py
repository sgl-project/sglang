import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
)

register_npu_ci(
    est_time=400, suite="full-16-npu-a3", nightly=True, disabled="unsupported feature"
)


class TestDeepseekV32IndexTopkPattern(GSM8KAscendMixin, CustomTestCase):
    model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
    accuracy = 0.93
    timeout_for_server_launch = 3000
    other_args = [
        "--trust-remote-code",
        "--tp",
        "16",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
        "--json-model-override-args",
        '{"index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSF"}',
    ]


class TestDeepseekV32IndexFreq(GSM8KAscendMixin, CustomTestCase):
    model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
    accuracy = 0.935
    timeout_for_server_launch = 3000
    other_args = [
        "--trust-remote-code",
        "--tp",
        "16",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
        "--json-model-override-args",
        '{"index_topk_freq": 4}',
    ]


if __name__ == "__main__":
    unittest.main()
