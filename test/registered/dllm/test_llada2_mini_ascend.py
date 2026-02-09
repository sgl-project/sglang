import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class TestLLaDA2Mini(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_disable_acl = os.environ.get("SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT")
        os.environ["SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT"] = "1"

        cls.model = "inclusionAI/LLaDA2.0-mini"
        cls.base_url = DEFAULT_URL_FOR_TEST

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

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,  # downloading model takes time, may change back to DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH after caching the model locally
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

        if cls._old_disable_acl is None:
            os.environ.pop("SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT", None)
        else:
            os.environ["SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT"] = cls._old_disable_acl

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.88)
        self.assertGreater(metrics["output_throughput"], 70)

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
