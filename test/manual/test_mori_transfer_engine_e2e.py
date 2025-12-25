import os
import subprocess
import unittest

import requests

from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)


class TestMoriTransferEngineE2E(PDDisaggregationServerBase):
    """
    Run:
        SGLANG_MORI_MANUAL_E2E=1 python3 test/manual/test_mori_transfer_engine_e2e.py

    Optional:
    - SGLANG_MORI_E2E_TEST_MODEL: override model (defaults to a small test model)
    - SGLANG_TEST_PD_DISAGG_DEVICES: RDMA devices string, e.g. "mlx5_roce0,mlx5_roce4"
    """

    @classmethod
    def setUpClass(cls):
        if os.environ.get("SGLANG_MORI_MANUAL_E2E", "") not in ("1", "true", "True"):
            raise unittest.SkipTest(
                "Set SGLANG_MORI_MANUAL_E2E=1 to run this manual MORI E2E test."
            )

        try:
            import torch

            if not torch.cuda.is_available():
                raise unittest.SkipTest("torch.cuda is not available.")
        except Exception as e:
            raise unittest.SkipTest(f"torch is not available/usable: {e}")

        # Force the disaggregation fixture to use MORI backend in local/manual runs.
        os.environ["SGLANG_TEST_PD_DISAGG_BACKEND"] = "mori"

        super().setUpClass()

        cls.model = os.environ.get(
            "SGLANG_MORI_E2E_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        )

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(
            cls.prefill_url + "/health", timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )
        cls.wait_server_ready(
            cls.decode_url + "/health", timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_TEST_PD_DISAGG_BACKEND", None)
        super().tearDownClass()

    @classmethod
    def launch_lb(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", " ".join(lb_command))
        cls.process_lb = subprocess.Popen(lb_command, stdout=None, stderr=None)
        cls.wait_server_ready(
            cls.lb_url + "/health", timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            "1",
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
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_generate_smoke(self):
        resp = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        out = resp.json()
        self.assertIn("text", out)
        self.assertIsInstance(out["text"], str)
        self.assertGreater(len(out["text"]), 0)


class TestMoriTransferEngineTPMismatchE2E(PDDisaggregationServerBase):
    """Manual MORI PD-disaggregation E2E with TP mismatch.

    Scenario:
    - prefill: tp=2 (GPU 0-1)
    - decode:  tp=4 (GPU 2-5)

    Manual-only and requires >= 6 visible GPUs.
    """

    _PORT_DELTA = 10

    @classmethod
    def setUpClass(cls):
        if os.environ.get("SGLANG_MORI_MANUAL_E2E", "") not in ("1", "true", "True"):
            raise unittest.SkipTest(
                "Set SGLANG_MORI_MANUAL_E2E=1 to run this manual MORI E2E test."
            )

        try:
            import torch

            if not torch.cuda.is_available():
                raise unittest.SkipTest("torch.cuda is not available.")
            if torch.cuda.device_count() < 6:
                raise unittest.SkipTest(
                    "TP-mismatch test requires >= 6 visible GPUs (prefill tp=2 + decode tp=4)."
                )
        except Exception as e:
            raise unittest.SkipTest(f"torch is not available/usable: {e}")

        os.environ["SGLANG_TEST_PD_DISAGG_BACKEND"] = "mori"
        super().setUpClass()

        # Shift ports to avoid clashing with TestMoriTransferEngineE2E.
        cls.lb_port = str(int(cls.lb_port) + cls._PORT_DELTA)
        cls.prefill_port = str(int(cls.prefill_port) + cls._PORT_DELTA)
        cls.decode_port = str(int(cls.decode_port) + cls._PORT_DELTA)
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"

        cls.model = os.environ.get(
            "SGLANG_MORI_E2E_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        )

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(
            cls.prefill_url + "/health", timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )
        cls.wait_server_ready(
            cls.decode_url + "/health", timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )
        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_TEST_PD_DISAGG_BACKEND", None)
        super().tearDownClass()

    @classmethod
    def launch_lb(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", " ".join(lb_command))
        cls.process_lb = subprocess.Popen(lb_command, stdout=None, stderr=None)
        cls.wait_server_ready(
            cls.lb_url + "/health", timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
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
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp",
            "4",
            "--base-gpu-id",
            "2",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_generate_smoke_tp_mismatch(self):
        resp = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        out = resp.json()
        self.assertIn("text", out)
        self.assertIsInstance(out["text"], str)
        self.assertGreater(len(out["text"]), 0)


if __name__ == "__main__":
    unittest.main()
