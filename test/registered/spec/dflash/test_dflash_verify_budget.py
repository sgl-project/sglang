import os
import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-large")

TARGET_MODEL = os.environ.get("SGLANG_TEST_QWEN35_MODEL", "Qwen/Qwen3.5-4B")
DRAFT_MODEL = os.environ.get(
    "SGLANG_TEST_QWEN35_DFLASH_MODEL", "z-lab/Qwen3.5-4B-DFlash"
)


class TestDFlashVerifyBudget(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(
            TARGET_MODEL, trust_remote_code=True
        )
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--speculative-num-draft-tokens",
                "16",
                "--speculative-dflash-verify-budget",
                "4",
                "--max-running-requests",
                "16",
                "--cuda-graph-max-bs-decode",
                "16",
                "--context-length",
                "8192",
                "--mem-fraction-static",
                "0.65",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate_batch(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": [
                    self.tokenizer.encode(
                        f"Request {i}: explain why regression tests are useful.",
                        add_special_tokens=False,
                    )
                    for i in range(16)
                ],
                "sampling_params": [
                    {
                        "temperature": 0,
                        "max_new_tokens": 32,
                        "ignore_eos": True,
                    }
                    for _ in range(16)
                ],
            },
        )
        response.raise_for_status()
        return response.json()

    def test_cuda_graph_batch_completes(self):
        results = self._generate_batch()

        self.assertEqual(len(results), 16)
        self.assertTrue(all(len(result["output_ids"]) == 32 for result in results))
        self.assertTrue(
            all(result["meta_info"]["spec_verify_ct"] > 0 for result in results)
        )
        self.assertIsNone(self.process.poll())


if __name__ == "__main__":
    unittest.main()
