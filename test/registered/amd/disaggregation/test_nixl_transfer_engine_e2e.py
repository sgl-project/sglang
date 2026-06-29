import os
import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_amd_ci(est_time=900, suite="stage-b-test-large-8-gpu-mi35x-disaggregation-amd")


class NixlTransferEngineBase(PDDisaggregationServerBase):
    """PD-over-NIXL e2e on ROCm. NIXL (upstream ai-dynamo/nixl + UCX --with-rocm)
    is enabled by default in the ROCm image; when the image was built with
    `--build-arg ENABLE_NIXL=0`, `import nixl` fails and the test skips rather
    than failing the suite."""

    port_delta = 0
    prefill_tp = 1
    decode_tp = 1
    decode_base_gpu_id = 1
    required_gpus = 2

    model_default = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    model_env_var = "SGLANG_NIXL_E2E_TEST_MODEL"
    extra_prefill_args: list = []
    extra_decode_args: list = []

    @classmethod
    def setUpClass(cls):
        try:
            import torch

            if not torch.cuda.is_available():
                raise unittest.SkipTest("torch.cuda is not available.")
            if torch.cuda.device_count() < cls.required_gpus:
                raise unittest.SkipTest(
                    f"NIXL PD smoke test requires >= {cls.required_gpus} visible GPUs."
                )
        except unittest.SkipTest:
            raise
        except Exception as e:
            raise unittest.SkipTest(f"torch is not available/usable: {e}")

        try:
            import nixl  # noqa: F401
        except Exception as e:
            raise unittest.SkipTest(
                "nixl not importable; image may have been built with "
                "--build-arg ENABLE_NIXL=0 "
                f"({e})."
            )

        super().setUpClass()

        cls._old_use_aiter = os.environ.get("SGLANG_USE_AITER")
        os.environ["SGLANG_USE_AITER"] = "1"

        # The shared fixture defaults to Mooncake in CI; pin NIXL explicitly here.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]

        rdma_env = os.environ.get("SGLANG_TEST_RDMA_DEVICE")
        if rdma_env:
            cls.rdma_devices = ["--disaggregation-ib-device", rdma_env]
            print(f"Found RDMA devices in env: {rdma_env}")
        else:
            print("SGLANG_TEST_RDMA_DEVICE is not set! Running without RDMA.")
            cls.rdma_devices = []

        cls._shift_ports()
        cls.model = try_cached_model(
            os.environ.get(cls.model_env_var, cls.model_default)
        )

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(
            cls.prefill_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_prefill,
        )
        cls.wait_server_ready(
            cls.decode_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_decode,
        )
        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_old_use_aiter", None) is None:
            os.environ.pop("SGLANG_USE_AITER", None)
        else:
            os.environ["SGLANG_USE_AITER"] = cls._old_use_aiter
        super().tearDownClass()

    @classmethod
    def _shift_ports(cls):
        if cls.port_delta == 0:
            return

        cls.lb_port = str(int(cls.lb_port) + cls.port_delta)
        cls.prefill_port = str(int(cls.prefill_port) + cls.port_delta)
        cls.decode_port = str(int(cls.decode_port) + cls.port_delta)
        cls.bootstrap_port = str(int(cls.bootstrap_port) + cls.port_delta)
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.base_url = cls.lb_url

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.prefill_tp),
            "--attention-backend",
            "aiter",
        ] + list(cls.extra_prefill_args)
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
            str(cls.decode_tp),
            "--base-gpu-id",
            str(cls.decode_base_gpu_id),
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.8",
        ] + list(cls.extra_decode_args)
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def _assert_generate_smoke(self):
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


class TestNixlTransferEngineE2E(NixlTransferEngineBase):
    def test_generate_smoke(self):
        self._assert_generate_smoke()


class TestNixlTransferEngineAccuracy(NixlTransferEngineBase):
    port_delta = 10
    model_default = DEFAULT_MODEL_NAME_FOR_TEST

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Evaluation metrics: {metrics}")
        self.assertGreater(metrics["accuracy"], 0.70)


if __name__ == "__main__":
    unittest.main()
