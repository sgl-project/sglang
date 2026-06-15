import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_amd_ci,
    is_in_ci,
    write_github_step_summary,
)

register_cuda_ci(est_time=339, stage="base-c", runner_config="8-gpu-h200")

FULL_DEEPSEEK_V3_MODEL_PATH = "deepseek-ai/DeepSeek-V3-0324"

_OTHER_ARGS = [
    "--tp",
    "8",
    "--trust-remote-code",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 64}',
]
if not is_in_amd_ci():
    _OTHER_ARGS += ["--mem-frac", "0.7"]


class TestDeepseekV3MTP(GSM8KMixin, DefaultServerBase):
    model = FULL_DEEPSEEK_V3_MODEL_PATH
    other_args = _OTHER_ARGS
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5

    gsm8k_accuracy_thres = 0.935
    gsm8k_accept_length_thres = 2.8

    # `test_z_bs_1_speed` runs after `test_gsm8k` (alphabetical) so it
    # measures steady-state speed on a warmed server.
    def test_z_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (deepseek-v3 mtp)\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(acc_length, 2.8)
            if is_in_amd_ci():
                self.assertGreater(speed, 15)
            else:
                self.assertGreater(speed, 130)


if __name__ == "__main__":
    unittest.main()
