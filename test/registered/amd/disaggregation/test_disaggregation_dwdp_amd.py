import os
import unittest

import requests

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    configure_nixl_pd_backend,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_amd_ci(
    est_time=1800,
    suite="stage-c-test-large-8-gpu-amd-mi35x",
)

MODEL_PATH = os.environ.get(
    "SGLANG_DWDP_TEST_MODEL",
    "/models/DeepSeek-R1-0528-MXFP4-th",
)
DWDP_BACKEND = os.environ.get("SGLANG_DWDP_TEST_BACKEND") or "ipc"


class TestDisaggregationDwdpAmd(PDDisaggregationServerBase):
    NUM_PREFILL_GPUS = 4
    NUM_DECODE_GPUS = 4

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"DWDP test model is unavailable: {MODEL_PATH}")
        os.environ.setdefault("SGLANG_USE_AITER", "1")
        os.environ.setdefault("GPU_ARCHS", "gfx950")
        super().setUpClass()
        configure_nixl_pd_backend(cls)
        cls.model = MODEL_PATH
        cls.start_prefill()
        cls.start_decode()
        cls.wait_server_ready(
            cls.prefill_url + "/health",
            timeout=1800,
            process=cls.process_prefill,
        )
        cls.wait_server_ready(
            cls.decode_url + "/health",
            timeout=1800,
            process=cls.process_decode,
        )
        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            str(cls.NUM_PREFILL_GPUS),
            "--dwdp-size",
            str(cls.NUM_PREFILL_GPUS),
            "--dwdp-weight-backend",
            DWDP_BACKEND,
            "--attention-backend",
            "aiter",
            "--disable-cuda-graph",
            "--log-level",
            "warning",
            "--mem-fraction-static",
            "0.80",
            "--disable-radix-cache",
        ]
        args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
        )

    @classmethod
    def start_decode(cls):
        args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            str(cls.NUM_DECODE_GPUS),
            "--dp-size",
            str(cls.NUM_DECODE_GPUS),
            "--enable-dp-attention",
            "--ep-size",
            str(cls.NUM_DECODE_GPUS),
            "--moe-dense-tp-size",
            "1",
            "--attention-backend",
            "aiter",
            "--disable-cuda-graph",
            "--log-level",
            "warning",
            "--mem-fraction-static",
            "0.80",
            "--disable-radix-cache",
            "--base-gpu-id",
            str(cls.NUM_PREFILL_GPUS),
        ]
        args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
        )

    def test_deterministic_completion(self):
        response = requests.post(
            self.base_url + "/v1/completions",
            json={
                "model": self.model,
                "prompt": "The capital of France is",
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=300,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["text"].strip()
        self.assertIn("paris", text.lower(), msg=f"unexpected completion: {text!r}")


if __name__ == "__main__":
    unittest.main()
