import subprocess
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_pd_server,
    run_with_timeout,
)


class TestDisaggregationMooncake(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        base_port = str(parsed_url.port)
        cls.lb_port = base_port
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        print(f"{cls.base_host=} {cls.lb_port=} {cls.prefill_port=} {cls.decode_port=}")
        run_with_timeout(cls.start_prefill, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)
        run_with_timeout(cls.start_decode, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)

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
            str(cls.base_port),
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
            "4",
            "--disaggregation-ib-device",
            "mlx5_roce0,mlx5_roce1,mlx5_roce2,mlx5_roce3",
        ]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            f"http://{cls.base_host}:{cls.prefill_port}",
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
            "4",
            "--disaggregation-ib-device",
            "mlx5_roce4,mlx5_roce5,mlx5_roce6,mlx5_roce7",
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            f"http://{cls.base_host}:{cls.decode_port}",
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
        for process in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")

    # def test_gsm8k(self):
    #     args = SimpleNamespace(
    #         num_shots=5,
    #         data_path=None,
    #         num_questions=200,
    #         max_new_tokens=512,
    #         parallel=128,
    #         host="http://127.0.0.1",
    #         port=int(self.lb_url.split(":")[-1]),
    #     )
    #     metrics = run_eval_few_shot_gsm8k(args)
    #     print(f"Evaluation metrics: {metrics}")

    #     self.assertGreater(metrics["accuracy"], 0.62)

    def test_logprob(self):
        prompt = "The capital of taiwan is "
        response = requests.post(
            f"http://{self.base_host}:{self.lb_port}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0},
                "return_logprob": True,
                "return_input_logprob": True,
                "logprob_start_len": 0,
            },
        )

        j = response.json()
        input_logprobs = j["meta_info"]["input_token_logprobs"]
        output_logprobs = j["meta_info"]["output_token_logprobs"]

        print(len(input_logprobs), len(output_logprobs))


# class TestDisaggregationFailure(CustomTestCase):
#     ...

if __name__ == "__main__":
    unittest.main()
