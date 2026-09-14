import os
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_cuda_ci(
    est_time=7200,
    suite="nightly-4-gpu-gb300-glm5-nvfp4",
    nightly=True,
)

MODEL_PATH = "nvidia/GLM-5.2-NVFP4"

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=glm45",
    "--tool-call-parser=glm47",
    "--quantization=modelopt_fp4",
    "--moe-runner-backend=flashinfer_trtllm",
]

DP_MTP_ARGS = [
    "--dp-size=2",
    "--enable-dp-attention",
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=1",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=2",
]


class TestGlm52Nvfp4PdDpaMtp(PDDisaggregationServerBase):
    """Regression coverage for GLM-5.2 PD + DP attention + MTP."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ["SGLANG_MOONCAKE_CUSTOM_MEM_POOL"] = "true"
        os.environ["MC_FORCE_MNNVL"] = "true"
        cls.model = MODEL_PATH

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_MOONCAKE_CUSTOM_MEM_POOL")
        os.environ.pop("MC_FORCE_MNNVL")
        super().tearDownClass()

    @classmethod
    def start_prefill(cls):
        prefill_args = COMMON_ARGS + [
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
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
        decode_args = (
            COMMON_ARGS
            + [
                "--disaggregation-mode",
                "decode",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--tp",
                "2",
                "--base-gpu-id",
                "2",
            ]
            + DP_MTP_ARGS
        )
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_first_routed_request_completes(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)

        data = response.json()
        self.assertIn("text", data, f"Unexpected response shape: {data}")
        self.assertIsInstance(data["text"], str)
        self.assertTrue(data["text"].strip(), "Generated text should not be empty")


if __name__ == "__main__":
    unittest.main()
