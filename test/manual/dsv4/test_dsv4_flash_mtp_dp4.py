"""DSV4 Flash MTP test using EAGLE speculative algorithm.

DSV4 Flash MTP shares the EAGLE wire path: EAGLE algo + NextN head built
into the target model weights. No separate draft model is needed (sglang
auto-falls back `--speculative-draft-model-path` to the target model).

Test matrix mirrors test_eagle_infer_b.TestEAGLEServerBasic to maximize
cuda-graph + buffer-pool coverage on the DSV4 path:
  - test_gsm8k         (accuracy + spec path full forward)
  - test_max_token_one (degenerate spec step, still cuda-graph captured)
  - test_request_abort (cuda-graph buffer pool survives abort+restart)

Server launch matches `run_flash_dp4.sh`: tp=4, dp=4, deepep MoE backend,
DSV4 FP8 (FP4 experts disabled).
"""

import random
import threading
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DSV4_FLASH_MODEL_PATH = "sgl-project/DeepSeek-V4-Flash-FP8"

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    # MTP runs ~num_draft_tokens forward passes per step, so the deepep
    # dispatch input size scales by that factor. Default 256 (used by the
    # plain server) overflows once cuda-graph-max-bs * num_draft_tokens
    # > 256. 1024 covers bs=128 * 4 draft tokens with headroom.
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}

DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

PROMPTS = [
    "[INST] You are a helpful assistant.\\nWhere are you from? [/INST]",
    "[INST] You are a helpful assistant.\\nSummarize gradient descent in 2 sentences. [/INST]",
    "[INST] You are a helpful assistant.\\nWhat is 17*23? [/INST]",
    "[INST] You are a helpful assistant.\\nList three primary colors. [/INST]",
]


class DSV4FlashMTPServerBase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DSV4_FLASH_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "4",
            "--dp",
            "4",
            "--enable-dp-attention",
            "--moe-a2a-backend",
            "deepep",
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "256",
            "--deepep-config",
            DEEPEP_CONFIG,
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-fraction-static",
            "0.7",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=DSV4_FLASH_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def send_request(self):
        time.sleep(random.uniform(0, 2))
        for prompt in PROMPTS:
            resp = requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 256},
                },
            )
            assert resp.status_code == 200

    def send_requests_abort(self):
        for prompt in PROMPTS:
            try:
                time.sleep(random.uniform(0, 2))
                requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 256},
                    },
                    timeout=0.5,
                )
            except requests.exceptions.Timeout:
                pass


class TestDSV4FlashMTPBasic(DSV4FlashMTPServerBase):
    def test_gsm8k(self):
        """Accuracy + spec path full forward."""
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_gsm8k_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.95)

    def test_max_token_one(self):
        """Degenerate spec step (still cuda-graph captured)."""
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=100,
            max_new_tokens=1,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_gsm8k_eval(args)
        self.assertGreater(metrics["output_throughput"], 50)

    def test_request_abort(self):
        """Cuda-graph buffer pool must survive abort+restart cycles."""
        concurrency = 4
        threads = [
            threading.Thread(target=self.send_request) for _ in range(concurrency)
        ] + [
            threading.Thread(target=self.send_requests_abort)
            for _ in range(concurrency)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertIsNone(self.process.poll())


if __name__ == "__main__":
    unittest.main()
