import os
import subprocess
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_pd_server,
    run_with_timeout,
)


class TestDisaggregationMooncakePrefillLargerTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Temporarily disable JIT DeepGEMM
        cls.original_jit_deepgemm = os.environ.get("SGL_ENABLE_JIT_DEEPGEMM")
        os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = "false"

        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        base_port = str(parsed_url.port)
        cls.lb_port = base_port
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        print(f"{cls.base_host=} {cls.lb_port=} {cls.prefill_port=} {cls.decode_port=}")

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        lb_command = [
            "python3",
            "-m",
            "sglang.srt.disaggregation.mini_lb",
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
        cls.process_lb = subprocess.Popen(
            lb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        cls.wait_server_ready(cls.lb_url + "/health")

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            "2",
            "--disaggregation-ib-device",
            "mlx5_roce0,mlx5_roce1",
        ]
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
            "2",
            "--disaggregation-ib-device",
            "mlx5_roce2",
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    @classmethod
    def wait_server_ready(cls, url, timeout=60):
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        # Restore JIT DeepGEMM environment variable
        if cls.original_jit_deepgemm is not None:
            os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = cls.original_jit_deepgemm
        else:
            os.environ.pop("SGL_ENABLE_JIT_DEEPGEMM", None)

        for process in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")
        # wait for 5 seconds
        time.sleep(5)

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

        self.assertGreater(metrics["accuracy"], 0.60)


class TestDisaggregationMooncakeDecodeLargerTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Temporarily disable JIT DeepGEMM
        cls.original_jit_deepgemm = os.environ.get("SGL_ENABLE_JIT_DEEPGEMM")
        os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = "false"

        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        base_port = str(parsed_url.port)
        cls.lb_port = base_port
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        print(f"{cls.base_host=} {cls.lb_port=} {cls.prefill_port=} {cls.decode_port=}")

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        lb_command = [
            "python3",
            "-m",
            "sglang.srt.disaggregation.mini_lb",
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
        cls.process_lb = subprocess.Popen(
            lb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        cls.wait_server_ready(cls.lb_url + "/health")

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            "1",
            "--disaggregation-ib-device",
            "mlx5_roce0",
        ]
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
            "2",
            "--base-gpu-id",
            "1",
            "--disaggregation-ib-device",
            "mlx5_roce1,mlx5_roce2",
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    @classmethod
    def wait_server_ready(cls, url, timeout=60):
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        # Restore JIT DeepGEMM environment variable
        if cls.original_jit_deepgemm is not None:
            os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = cls.original_jit_deepgemm
        else:
            os.environ.pop("SGL_ENABLE_JIT_DEEPGEMM", None)

        for process in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")
        # wait for 5 seconds
        time.sleep(5)

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

        self.assertGreater(metrics["accuracy"], 0.60)


if __name__ == "__main__":
    unittest.main()
