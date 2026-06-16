import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import is_in_ci, write_github_step_summary

register_cuda_ci(est_time=641, stage="base-c", runner_config="4-gpu-b200")

DSV32_FP4_MODEL = "nvidia/DeepSeek-V3.2-NVFP4"


class TestDeepseekV32FP4TPSpec(GSM8KMixin, DefaultServerBase):
    model = DSV32_FP4_MODEL
    timeout = 1200
    other_args = [
        "--tp",
        "4",
        "--attention-backend",
        "dsa",
        "--moe-runner-backend",
        "flashinfer_trtllm",
        "--quantization",
        "modelopt_fp4",
        "--tool-call-parser",
        "deepseekv32",
        "--reasoning-parser",
        "deepseek-v3",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true,"num_threads": 64}',
    ]

    gsm8k_accuracy_thres = 0.93
    gsm8k_num_questions = 500
    gsm8k_num_threads = 500
    gsm8k_num_shots = 20
    gsm8k_accept_length_thres = 2.7

    def test_z_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (deepseek-v32 mtp tp)\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(acc_length, 2.7)
            self.assertGreater(speed, 150)


if __name__ == "__main__":
    unittest.main()
