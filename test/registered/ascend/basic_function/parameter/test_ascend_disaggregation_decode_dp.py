import os
import unittest
import logging

import requests

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_pd_server,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

base_port = int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0])
BASE_PORT_FOR_ASCEND_MF = 20000 + base_port * 1000 +66
os.environ["ASCEND_MF_STORE_URL"] = f"tcp://127.0.0.1:{BASE_PORT_FOR_ASCEND_MF}"

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestDisaggregationDecodeDp(TestDisaggregationBase):
    """Testcase：Verify the correctness of --disaggregation-decode-dp=2 and Prefill/Decode disaggregated services availability on Ascend NPU backend.

    [Test Category] Parameter
    [Test Target] --disaggregation-decode-dp; --disaggregation-mode; --disaggregation-transfer-backend
    """

    @classmethod
    def setUpClass(cls):
        # Test class initialization: Launch Prefill/Decode disaggregated services and load balancer, then wait for services to be ready
        logger.info(os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
        super().setUpClass()
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        # Launch the Prefill service with disaggregation-decode-dp=2 configuration for Ascend NPU
        prefill_args = (
            [
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-decode-dp",
                "2",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disable-cuda-graph",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.8,
            ]
        )

        env = os.environ.copy()

        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls):
        # Launch the Decode service with specified configuration for Ascend NPU (disaggregated architecture)
        ascend_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ascend_devices
        base_gpu_id = ascend_devices.split(",")[2] if len(ascend_devices.split(",")) >= 3 else "2"
        decode_args = (
            [
                "--disaggregation-mode",
                "decode",
                "--base-gpu-id",
                base_gpu_id,
                "--disaggregation-transfer-backend",
                "ascend",
                "--disable-cuda-graph",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.8,
            ]
        )

        env = os.environ.copy()

        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_disaggregation_decode_dp(self):
        """Test core functionality of disaggregation-decode-tp parameter.

        Test Steps:
        1. Verify LB service health (basic availability check)
        2. Validate inference correctness (France capital = Paris)
        3. Confirm disaggregation_decode_dp=2 in Prefill server info (parameter validation)
        """
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        response = requests.get(self.prefill_url + "/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["disaggregation_decode_dp"], 2)

    @classmethod
    def tearDownClass(cls):
        # Test class cleanup: Remove the Ascend MF store environment variable and call parent class cleanup to terminate all processes
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
