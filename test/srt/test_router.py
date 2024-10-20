import socket
import subprocess
import time
import unittest
from types import SimpleNamespace
from typing import List
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_child_process
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
)

# DEFAULT_MODEL_NAME_FOR_TEST = "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/"


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # specify 0 tells the OS to assign a random port
        s.bind(("127.0.0.1", 0))
        # call listen to make it a server socket
        s.listen(1)
        port = s.getsockname()[1]
    return port


def popen_launch_server(model, url, device_id, timeout):
    # NOTE: not reusing popen_launch_server from test_utils because passing CUDA_VISIBLE_DEVICES=0 as env causes silent failure.
    # using shell mode works instead
    parsed_url = urlparse(url)

    host = parsed_url.hostname
    port = parsed_url.port

    command = f"export CUDA_VISIBLE_DEVICES={device_id}; python -m sglang.launch_server --model-path {model} --host {host} --port {port}"

    process = subprocess.Popen(command, stdout=None, stderr=None, shell=True)

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/health")
            if response.status_code == 200:
                return process
        except requests.RequestException:
            pass
        time.sleep(10)

    raise TimeoutError("Server failed to start within the timeout period.")


def popen_launch_router(
    router_url: str,
    worker_urls: List[str],
    policy: str,
    timeout: float,
):
    parsed_url = urlparse(router_url)

    host = parsed_url.hostname
    port = parsed_url.port

    command = f"python -m sglang.srt.router.launch_router --host {host} --port {port} --policy {policy} --worker-urls {' '.join(worker_urls)}"

    process = subprocess.Popen(command, stdout=None, stderr=None, shell=True)

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{router_url}/health")
            if response.status_code == 200:
                return process
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Router failed to start within the timeout period.")


class TestRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.proc_list = []
        NUM_WORKERS = 2

        host = "127.0.0.1"
        worker_urls = []

        for i in range(NUM_WORKERS):
            port = find_available_port()
            url = f"http://{host}:{port}"

            cls.proc_list.append(
                popen_launch_server(
                    cls.model,
                    url,
                    device_id=str(i),
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                )
            )
            worker_urls.append(url)

        router_port = find_available_port()

        router_url = f"http://{host}:{router_port}"

        cls.proc_list.append(
            popen_launch_router(
                router_url=router_url,
                worker_urls=worker_urls,
                policy="round_robin",
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            )
        )

        cls.router_url = router_url

    @classmethod
    def tearDownClass(cls):
        for p in cls.proc_list:
            kill_child_process(p.pid)

    def test_gsm8k(self):

        parsed_url = urlparse(self.router_url)

        host = parsed_url.hostname
        port = parsed_url.port

        args = SimpleNamespace(
            num_shots=5,
            # data_path="/home/jobuser/resources/data/test.jsonl",
            data_path="test.jsonl",
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{host}",
            port=port,
        )

        metrics = run_eval(args)

        assert metrics["accuracy"] >= 0.70


if __name__ == "__main__":
    unittest.main()
