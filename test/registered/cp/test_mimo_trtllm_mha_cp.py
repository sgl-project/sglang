import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=400, stage="extra-b", runner_config="4-gpu-b200")

MIMO_V2_MODEL_PATH = "XiaomiMiMo/MiMo-V2.5"
GSM8K_BASELINE_ACCURACY = 0.93


class TestMiMoV2TRTLLMMHAPrefillContextParallel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MIMO_V2_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--prefill-attention-backend",
                "trtllm_mha",
                "--decode-attention-backend",
                "fa4",
                "--mm-attention-backend",
                "fa4",
                "--enable-prefill-cp",
                "--cp-strategy",
                "zigzag",
                "--mem-fraction-static",
                "0.8",
                "--chunked-prefill-size",
                "8192",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_attention_backend_configuration(self):
        response = requests.get(self.base_url + "/server_info", timeout=10)
        response.raise_for_status()
        server_info = response.json()

        self.assertEqual(server_info["prefill_attention_backend"], "trtllm_mha")
        self.assertEqual(server_info["decode_attention_backend"], "fa4")
        self.assertTrue(server_info["enable_prefill_cp"])
        self.assertEqual(server_info["cp_strategy"], "zigzag")
        self.assertEqual(server_info["attn_cp_size"], 4)

    def test_gsm8k(self):
        metrics = run_eval(
            SimpleNamespace(
                model=self.model,
                eval_name="gsm8k",
                api="chat",
                num_shots=5,
                num_examples=200,
                max_tokens=4096,
                num_threads=8,
                repeat=1,
                temperature=0.0,
                top_p=1.0,
                base_url=self.base_url,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
        )
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], GSM8K_BASELINE_ACCURACY)


if __name__ == "__main__":
    unittest.main()
