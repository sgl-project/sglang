import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestBaichuan(GSM8KAscendMixin, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/baichuan-inc/Baichuan2-13B-Chat"
    accuracy = 0.48
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--max-running-requests",
        "128",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
    ]
    gsm8k_num_shots = 1


if __name__ == "__main__":
    unittest.main()
