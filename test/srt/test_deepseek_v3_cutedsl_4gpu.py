import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)


class TestDeepseekR1Nvfp4CuteDSLDeepEP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.89",
            "--max-prefill-tokens",
            "16384",
            "--max-running-requests",
            "256",
            "--chunked-prefill-size",
            "1024",
            "--tp",
            "4",
            "--dp",
            "4",
            "--ep",
            "4",
            "--moe-dense-tp-size",
            "1",
            "--enable-dp-attention",
            "--quantization",
            "modelopt_fp4",
            "--attention-backend",
            "trtllm_mla",
            "--moe-a2a-backend",
            "deepep",
            "--moe-runner-backend",
            "flashinfer_cutedsl",
            "--deepep-mode",
            "low_latency",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={
                **os.environ,
                "SGLANG_DEEPEP_BF16_DISPATCH": "1",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
                "SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH": "0",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=512,
            parallel=512,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["accuracy"], 0.92)


if __name__ == "__main__":
    unittest.main()
