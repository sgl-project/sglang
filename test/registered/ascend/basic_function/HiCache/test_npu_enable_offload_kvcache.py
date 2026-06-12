import os
import random
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace
from typing import Dict

import requests

from sglang.bench_serving import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)
standard_accuracy = 0.9


class DisaggregationHiCacheBase(PDDisaggregationServerBase):
    """Testcase: In PD separation scenarios, enabling offset does not degrade accuracy compared to not enabling offset.

    [Test Category] Parameter
    [Test Target] --disaggregation-decode-enable-offload-kvcache
    """

    @classmethod
    def setUpClass(cls):
        super(DisaggregationHiCacheBase, cls).setUpClass()

        cls.model = QWEN3_8B_WEIGHTS_PATH

        cls.tokenizer = get_tokenizer(cls.model)
        cls.temp_dir = tempfile.mkdtemp()
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        # Startup prefill parameters
        prefill_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "2",
            "--enable-hierarchical-cache",
            "--hicache-io-backend",
            "kernel_ascend",
            "--hicache-mem-layout",
            "page_first_direct",
            "--hicache-ratio",
            "1.2",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
        ]
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
        }
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        # Test class cleanup: call parent class cleanup to terminate all processes
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        super().tearDownClass()

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
        # Send a generate request and return response
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
        # Helper method to trigger offloading and flush cache
        # Trigger offloading
        self.send_request(self.gen_prompt(1), max_tokens=150)

        # Flush device cache to force remote storage access
        time.sleep(2)
        requests.post(self.prefill_url + "/flush_cache")


class TestDisaggregationDecodeDisableOffload(DisaggregationHiCacheBase):
    """Decode startup parameters, disable offload-kvcache"""

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "2",
            "--page-size",
            "128",
            "--mem-fraction-static",
            "0.9",
            "--base-gpu-id",
            "2",
        ]
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
        }
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=self.lb_port,
        )
        metrics = run_eval(args)
        self.disable_offload_accuracy = metrics["accuracy"]

        # Contrast accuracy
        self.assertGreaterEqual(
            self.disable_offload_accuracy,
            standard_accuracy,
            f"The accuracy did not meet the standard.",
        )


class TestDisaggregationDecodeEnableOffload(DisaggregationHiCacheBase):
    """Decode startup parameters, enable offload-kvcache"""

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "2",
            "--page-size",
            "128",
            "--mem-fraction-static",
            "0.9",
            "--base-gpu-id",
            "2",
            "--disaggregation-decode-enable-offload-kvcache",
            "--hicache-io-backend",
            "kernel_ascend",
            "--hicache-mem-layout",
            "page_first_direct",
            "--hicache-ratio",
            "1.2",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
        ]

        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
        }
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=self.lb_port,
        )
        metrics = run_eval(args)
        self.enable_offload_accuracy = metrics["accuracy"]

        # Contrast accuracy
        self.assertGreaterEqual(
            self.enable_offload_accuracy,
            standard_accuracy,
            f"The accuracy did not meet the standard.",
        )


if __name__ == "__main__":
    unittest.main()
