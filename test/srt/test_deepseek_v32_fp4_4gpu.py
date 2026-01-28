import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

FULL_DEEPSEEK_V3_FP4_MODEL_PATH = "nvidia/DeepSeek-V3.2-NVFP4"
SERVER_LAUNCH_TIMEOUT = 1200


class TestDeepseekV32FP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FULL_DEEPSEEK_V3_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "4",
            "--dp",
            "4",
            "--enable-dp-attention",
            "--attention-backend",
            "nsa",
            "--moe-runner-backend",
            "flashinfer_trtllm",
            "--quantization",
            "modelopt_fp4",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--tool-call-parser",
            "deepseekv32",
            "--reasoning-parser",
            "deepseek-v3",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true,"num_threads": 64}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            num_shots=20,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v3-fp4)\n" f'{metrics["accuracy"]=:.3f}\n'
            )

        self.assertGreater(metrics["accuracy"], 0.935)


if __name__ == "__main__":
    unittest.main()
