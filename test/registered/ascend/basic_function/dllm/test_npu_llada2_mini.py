import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import LLaDA2_0_MINI_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    CustomTestCase,
    is_in_ci,
    write_github_step_summary,
)

register_npu_ci(est_time=400, suite="stage-b-test-4-npu-a3", nightly=False)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestLLaDA2Mini(GSM8KAscendMixin, CustomTestCase):
    model = LLaDA2_0_MINI_WEIGHTS_PATH

    other_args = [
        "--trust-remote-code",
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.9",
        "--max-running-requests",
        "1",
        "--attention-backend",
        "ascend",
        "--dllm-algorithm",
        "LowConfidence",  # TODO: Add dLLM configurations
    ]
    env = {
        **os.environ,
        "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1",  # Need to avoid OOM issue
    }
    accuracy = 0.88
    output_throughput = 70

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (llada2-mini) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, 130)


if __name__ == "__main__":
    unittest.main()
