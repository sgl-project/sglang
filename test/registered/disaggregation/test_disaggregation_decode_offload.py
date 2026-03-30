import os
import shutil
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_ci,
    popen_launch_pd_server,
)

# Registering the test for CUDA CI with appropriate parameters
# Increasing estimated time since we run evaluation twice
register_cuda_ci(est_time=600, suite="stage-b-test-2-gpu-large")


@unittest.skipIf(is_in_ci(), "Temporarily disable the flaky test.")
class TestDisaggregationDecodeOffload(PDDisaggregationServerBase):
    """
    Test class for verifying KV cache offloading on the decode side in a
    prefill-decode disaggregation setup.
    """

    @classmethod
    def setUpClass(cls):
        # Set environment variable to make offloading more frequent for testing purposes
        cls.old_stride = os.environ.get("SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE")
        cls.hicache_dir = "/tmp/hicache_test"
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = cls.hicache_dir
        os.environ["SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE"] = "16"

        # Ensure a clean cache directory
        if os.path.exists(cls.hicache_dir):
            shutil.rmtree(cls.hicache_dir)
        os.makedirs(cls.hicache_dir, exist_ok=True)

        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST

        # Non-blocking start of prefill and decode servers
        cls.start_prefill()
        cls.start_decode()

        # Wait for both servers to be ready before proceeding
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        # Restore the original environment variable state
        super().tearDownClass()
        if cls.old_stride is not None:
            os.environ["SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE"] = cls.old_stride
        else:
            os.environ.pop("SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE", None)

        os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", None)

        # Clean up the cache directory
        if os.path.exists(cls.hicache_dir):
            shutil.rmtree(cls.hicache_dir)

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--page-size",
            "16",
            "--enable-hierarchical-cache",
            "--hicache-storage-backend",
            "file",
            "--hicache-ratio",
            "2",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--disaggregation-decode-enable-offload-kvcache",
            "--num-reserved-decode-tokens",
            "128",
            "--hicache-ratio",
            "2",
            "--page-size",
            "16",
            "--hicache-storage-backend",
            "file",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_mmlu_double_eval(self):
        """
        Run two rounds of MMLU evaluation:
        1. First round: Decode node offloads KV cache back to disk (HiCache).
        2. Restart All Nodes to clear memory cache.
        3. Second round: Prefill node loads KV cache from disk (HiCache).
        Verify that both rounds produce consistent scores.
        """
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics1 = run_eval(args)

        # Ensure all offloads are committed to disk
        import time

        time.sleep(10)

        kill_process_tree(self.process_prefill.pid)
        kill_process_tree(self.process_decode.pid)
        kill_process_tree(self.process_lb.pid)
        self.process_prefill.wait()
        self.process_decode.wait()
        self.process_lb.wait()

        self.start_prefill()
        self.start_decode()
        self.launch_lb()
        self.wait_server_ready(self.prefill_url + "/health")
        self.wait_server_ready(self.decode_url + "/health")

        metrics2 = run_eval(args)

        # Assert score is above a minimum threshold for both rounds
        self.assertGreater(metrics1["score"], 0.65)
        self.assertGreater(metrics2["score"], 0.65)

        # Score should be consistent: round 2 should be >= round 1, or at least within a 0.05 margin if slightly lower
        self.assertGreaterEqual(metrics2["score"], metrics1["score"] - 0.05)

    def test_abort_no_memory_leak(self):
        """
        Verify that aborting requests mid-flight in decode-offload mode does not
        leak GPU KV cache or host memory.

        Strategy:
        1. Fire a batch of long-running requests through the load balancer.
        2. After a short delay (so some requests have started offloading), abort
           all of them via the decode server's /abort_request endpoint.
        3. Wait for the server to stabilize, then send a fresh batch of requests
           and verify they complete successfully — which would fail if KV cache
           or host memory had been exhausted by the leak.
        """
        num_requests = 8
        prompt = (
            "Repeat the following sentence 100 times: The quick brown fox jumps over the lazy dog. "
            * 3
        )

        def send_one(_):
            try:
                requests.post(
                    f"{self.lb_url}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 512,
                            "ignore_eos": True,
                        },
                    },
                    timeout=60,
                )
            except Exception:
                pass

        # Start requests concurrently
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(send_one, i) for i in range(num_requests)]

            # Give the decode server time to start offloading KV cache
            time.sleep(3)

            # Abort all in-flight requests on the decode server directly
            abort_resp = requests.post(
                f"{self.decode_url}/abort_request",
                json={"abort_all": True},
                timeout=10,
            )
            self.assertIn(abort_resp.status_code, (200, 204))

            # Wait for all client threads to finish (they may get abort responses)
            for future in as_completed(futures, timeout=30):
                future.result()

        # Allow the server to fully process the abort and reclaim resources
        time.sleep(5)

        # Verify the server is still healthy after the abort
        health_resp = requests.get(f"{self.decode_url}/health", timeout=10)
        self.assertEqual(health_resp.status_code, 200)

        # Send a fresh batch of short requests; if memory leaked these would
        # fail with OOM or hang indefinitely
        def send_short(_):
            resp = requests.post(
                f"{self.lb_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                },
                timeout=60,
            )
            return resp

        with ThreadPoolExecutor(num_requests) as executor:
            results = list(executor.map(send_short, range(num_requests)))

        for resp in results:
            self.assertEqual(resp.status_code, 200)
            self.assertIn("text", resp.json())


if __name__ == "__main__":
    unittest.main()
