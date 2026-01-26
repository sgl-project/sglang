import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

FULL_DEEPSEEK_V3_MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528"


class TestMoriEPE2E(CustomTestCase):
    """
    Run:
        python3 test/manual/test_moriep_e2e.py

    """

    @classmethod
    def setUpClass(cls):
        cls.model = FULL_DEEPSEEK_V3_MODEL_PATH

        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp-size",
            "8",
            "--ep-size",
            "8",
            "--dp-size",
            "8",
            "--enable-dp-attention",
            "--moe-a2a-backend",
            "mori",
            "--trust-remote-code",
            "--load-balance-method",
            "round_robin",
            "--moe-dense-tp-size",
            "1",
            "--enable-dp-lm-head",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.6",
            "--chunked-prefill-size",
            "131072",
            "--max-running-requests",
            "128",
            "--context-length",
            "12288",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--attention-backend",
            "aiter",
        ]

        env = dict(os.environ)
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_MORI_FP8_DISP"] = "True"
        env["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "16384"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1400,
            parallel=1400,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v3)\n" f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], 0.935)


if __name__ == "__main__":
    unittest.main()
