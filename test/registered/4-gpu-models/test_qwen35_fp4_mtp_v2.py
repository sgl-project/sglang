import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.reasoning_kit import ReasoningTokenUsageMixin
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=406, suite="stage-c-test-4-gpu-b200")

QWEN35_FP4_MODEL = "nvidia/Qwen3.5-397B-A17B-NVFP4"
ACC_THRESHOLDS = {QWEN35_FP4_MODEL: {"gsm8k": 0.95}}


class TestQwen35FP4MTPV2(ReasoningTokenUsageMixin, CustomTestCase):
    reasoning_parser_name = "qwen3"

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_FP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.init_reasoning_token_verifier()
        envs.SGLANG_ENABLE_SPEC_V2.set(True)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--chunked-prefill-size",
                "2048",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                "128",
                "--mamba-ssm-dtype",
                "bfloat16",
                "--max-running-requests",
                "128",
                "--reasoning-parser",
                "qwen3",
                "--attention-backend",
                "trtllm_mha",
                "--quantization",
                "modelopt_fp4",
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--mem-fraction-static",
                "0.8",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true,"num_threads": 64}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        envs.SGLANG_ENABLE_SPEC_V2.set(False)
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            model=self.model,
            eval_name="gsm8k",
            num_shots=5,
            num_examples=200,
            max_tokens=16000,
            num_threads=128,
            repeat=1,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            base_url=self.base_url,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], ACC_THRESHOLDS[self.model]["gsm8k"])

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 3.3)


if __name__ == "__main__":
    unittest.main()
