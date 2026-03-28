import unittest
import threading
import requests
import time

from sglang.test.ascend.test_ascend_utils import QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=500, suite="nightly-8-npu-a3", nightly=True)


class TestMambaCache(CustomTestCase):
    """Testcase：Verify the MambaCache with different parameters, concurrent requests, long sequence,
    Inference request successful

    [Test Category] Parameter
    [Test Target] --mamba-full-memory-ratio, --mamba-ssm-dtype, --mamba-track-interval, --mamba-track-size,
    --max-mamba-cache-size
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.model_path

    test_prompt = "The capital of France is"
    expected_output = "Paris"

    @classmethod
    def setUpClass(cls):
        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def _launch_server_with_mamba_params(
        self,
        max_mamba_cache_size=None,
        mamba_ssm_dtye=None,
        mamba_full_memory_ratio=0.9,
        mamba_scheduler_strategy="auto",
        mamba_track_interval=256,
    ):
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.5",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mamba-full-memory-ratio",
            mamba_full_memory_ratio,
            "--max-mamba-cache-size",
            "1024",
            "--mamba-ssm-dtype",
            "float32",
            "--mamba-scheduler-strategy",
            mamba_scheduler_strategy,
            "--mamba-track-interval",
            mamba_track_interval,
            "--tp-size",
            8,
            "--disable-radix-cache",
        ]
        if max_mamba_cache_size is not None:
            other_args.extend(["--max-mamba-cache-size", max_mamba_cache_size])
        if mamba_ssm_dtye is not None:
            other_args.extend(["--mamba-ssm-dtype", mamba_ssm_dtye])
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        return process

    def _tes_basic_inference(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": self.test_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        print("=--------------response.json()--------------------------")
        print(response.json())
        self.assertIn(self.expected_output, response.text)
        return response.text

    '''
        def _send_concurrent_requests(self, num_requests=10):
        results = []
        threads = []

        def send_request(rid):
            try:
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": f"Test request{rid}: What is AI?",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 32,
                        },
                    },
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                )
                results.append(rid, response.status_code, response.text)
            except Exception as e:
                results.append(rid, -1, str(e))

        for i in range(num_requests):
            thread = threading.Thread(target=send_request, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return results
    '''

    '''
    def test_mamba_size_large(self):
        self.process = self._launch_server_with_mamba_params(
            max_mamba_cache_size=2048,
        )
        try:
            time.sleep(5)
            result = self._tes_basic_inference()
            print(result)
        finally:
            kill_process_tree(self.process.pid)
    '''

    '''
    def test_mamba_long_sequence(self):
        self.process = self._launch_server_with_mamba_params(
            max_mamba_cache_size=2048
        )
        try:
            time.sleep(5)
            long_prompt = "Explain the concept of machine learning in detail." * 10
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": long_prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 128,
                    },
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.text), 0)
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_concurrent_requests(self):
        self.process = self._launch_server_with_mamba_params()

        try:
            time.sleep(5)
            results = self._send_concurrent_requests(num_requests=10)
            success_count = sum(1 for r in results if r[1] == 200)
            self.assertEqual(success_count, 10)
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_track_interval(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_track_interval=128
        )
        try:
            time.sleep(5)
            result = self._tes_basic_inference()
            print(result)
        finally:
            kill_process_tree(self.process.pid)
    '''

    def test_mamba_scheduler_no_buffer(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_scheduler_strategy="no_buffer",
        )
        try:
            time.sleep(5)
            result = self._tes_basic_inference()
            print(result)
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_ssm_dtype_float32(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_ssm_dtye="float32",
        )
        try:
            time.sleep(5)
            result = self._tes_basic_inference()
            print(result)
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_ssm_dtype_bfloat16(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_ssm_dtye="bfloat16",
        )
        try:
            time.sleep(5)
            result = self._tes_basic_inference()
            print(result)
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_ssm_dtype_float16(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_ssm_dtye="float16",
        )
        try:
            time.sleep(5)
            result = self._tes_basic_inference()
            print(result)
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_max_mamba_cache_size_512(self):
        self.process = self._launch_server_with_mamba_params(
            max_mamba_cache_size=512,
        )
        try:
            time.sleep(5)
            result = self._tes_basic_inference()
            print(result)
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_full_memory_ratio(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_full_memory_ratio=0.5,
        )
        try:
            time.sleep(5)
            result = self._tes_basic_inference()
            print(result)
        finally:
            kill_process_tree(self.process.pid)


if __name__ == "__main__":
    unittest.main()
