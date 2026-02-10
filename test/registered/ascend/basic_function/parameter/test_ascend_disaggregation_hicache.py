import os
import random
import tempfile
import time
import unittest
from typing import Dict

import requests
from types import SimpleNamespace
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.bench_serving import get_tokenizer
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class DisaggregationHiCacheBase(PDDisaggregationServerBase):
    """Testcase: Vaildate Prefill/Decode disaggregated services with hicache write policy configuration, Repeated long hints hit the prefix cache.
                 and on the GSM8K dataset is no less than 0.86

    [Test Category] Parameter
    [Test Target]  --hicache-write-policy
    """

    @classmethod
    def setUpClass(cls):
        super(DisaggregationHiCacheBase, cls).setUpClass()

        cls.model = QWEN3_32B_WEIGHTS_PATH

        cls.tokenizer = get_tokenizer(cls.model)
        cls.temp_dir = tempfile.mkdtemp()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        # Prefill with HiCache enabled
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "2",
            "--page-size",
            "128",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "1.2",
            "--hicache-write-policy",
            "write_through",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
            "--attention-backend",
            "ascend",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:26666",
        }
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls):
        pass

    def gen_prompt(self, token_num: int) -> str:
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def send_request(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0
    ) -> Dict:
        """Send a generate request and return response"""
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def trigger_offloading_and_flush(self):
        """Helper method to trigger offloading and flush cache"""
        # Trigger offloading
        self.send_request(self.gen_prompt(1), max_tokens=150)

        # Flush device cache to force remote storage access
        time.sleep(2)
        requests.post(self.prefill_url + "/flush_cache")


class TestDisaggregationPrefillWithHiCache(DisaggregationHiCacheBase):
    """Test disaggregation with HiCache enabled only on Prefill side"""

    @classmethod
    def start_decode(cls):
        # Decode without HiCache offload
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "2",
            "--page-size",
            "128",
            "--mem-fraction-static",
            "0.9",
            "--base-gpu-id",
            "2",
            "--attention-backend",
            "ascend",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:26666",
        }
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_prefill_cache_hit(self):
        """Test that prefill cache works with repeated queries"""

        repeated_prompt = self.gen_prompt(800)

        # First request - should miss cache
        self.send_request(repeated_prompt, max_tokens=100)
        # Flush cache
        # Second request - should hit cache (faster)
        response2 = self.send_request(repeated_prompt, max_tokens=100)
        # Assert cached tokens cnt
        self.assertGreater(response2["meta_info"]["cached_tokens"], 700)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=21000,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics['accuracy'], 0.86)

    @classmethod
    def tearDownClass(cls):
        # Test class cleanup: call parent class cleanup to terminate all processes
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
