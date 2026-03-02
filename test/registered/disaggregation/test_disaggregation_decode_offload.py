import os
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

# Registering the test for CUDA CI with appropriate parameters
register_cuda_ci(est_time=400, suite="stage-b-test-large-2-gpu")


class TestDisaggregationDecodeOffload(PDDisaggregationServerBase):
    """
    Test class for verifying KV cache offloading on the decode side in a 
    prefill-decode disaggregation setup.
    """

    @classmethod
    def setUpClass(cls):
        # Set environment variable to make offloading more frequent for testing purposes
        cls.old_stride = os.environ.get("SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE")
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = "/tmp/hicache"
        os.environ["SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE"] = "16"

        super().setUpClass()
        # Using a small model for faster test execution and reduced memory footprint
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        # Non-blocking start of prefill and decode servers
        cls.start_prefill()
        cls.start_decode()

        # Wait for both servers to be ready before proceeding
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        # Launch the load balancer
        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        # Restore the original environment variable state
        super().tearDownClass()
        if cls.old_stride is not None:
            os.environ["SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE"] = cls.old_stride
        else:
            os.environ.pop("SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE", None)

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            "1",
            "--page-size",
            "16",
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

    def test_gsm8k(self):
        """
        Run a few-shot GSM8K evaluation to ensure end-to-end correctness 
        while offloading logic is active.
        """
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=20,
            max_new_tokens=512,
            parallel=16,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["accuracy"], 0.30)


if __name__ == "__main__":
    unittest.main()