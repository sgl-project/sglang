import socket
import subprocess
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
)


def popen_launch_router(
    model: str,
    base_url: str,
    dp_size: int,
    timeout: float,
    policy: str = "cache_aware",
):
    """
    Launch the router server process.

    Args:
        model: Model path/name
        base_url: Server base URL
        dp_size: Data parallel size
        timeout: Server launch timeout
        policy: Router policy, one of "cache_aware", "round_robin", "random"
    """
    _, host, port = base_url.split(":")
    host = host[2:]

    command = [
        "python3",
        "-m",
        "sglang_router.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        port,
        "--dp",
        str(dp_size),  # Convert dp_size to string
        "--router-eviction-interval",
        "5",  # frequent eviction for testing
        "--router-policy",
        policy,
    ]

    process = subprocess.Popen(command, stdout=None, stderr=None)

    start_time = time.time()
    with requests.Session() as session:
        while time.time() - start_time < timeout:
            try:
                response = session.get(f"{base_url}/health")
                if response.status_code == 200:
                    print(f"Router {base_url} is healthy")
                    return process
            except requests.RequestException:
                pass
            time.sleep(10)

    raise TimeoutError("Router failed to start within the timeout period.")


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def popen_launch_server(
    model: str,
    base_url: str,
    timeout: float,
):
    _, host, port = base_url.split(":")
    host = host[2:]

    command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        port,
        "--base-gpu-id",
        "1",
    ]

    process = subprocess.Popen(command, stdout=None, stderr=None)

    # intentionally don't wait and defer the job to the router health check
    return process


def terminate_and_wait(process, timeout=300):
    """Terminate a process and wait until it is terminated.

    Args:
        process: subprocess.Popen object
        timeout: maximum time to wait in seconds

    Raises:
        TimeoutError: if process does not terminate within timeout
    """
    if process is None:
        return

    process.terminate()
    start_time = time.time()

    while process.poll() is None:
        print(f"Terminating process {process.pid}")
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Process {process.pid} failed to terminate within {timeout}s"
            )
        time.sleep(1)

    print(f"Process {process.pid} is successfully terminated")


class TestLaunchServer(unittest.TestCase):
    def setUp(self):
        self.model = DEFAULT_MODEL_NAME_FOR_TEST
        self.base_url = DEFAULT_URL_FOR_TEST
        self.process = None
        self.other_process = []

    def tearDown(self):
        print("Running tearDown...")
        if self.process:
            terminate_and_wait(self.process)
        for process in self.other_process:
            terminate_and_wait(process)
        print("tearDown done")

    def test_1_mmlu(self):
        print("Running test_1_mmlu...")
        # DP size = 2
        self.process = popen_launch_router(
            self.model,
            self.base_url,
            dp_size=2,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            policy="cache_aware",
        )

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )

        metrics = run_eval(args)
        score = metrics["score"]
        THRESHOLD = 0.65
        passed = score >= THRESHOLD
        msg = f"MMLU test {'passed' if passed else 'failed'} with score {score:.3f} (threshold: {THRESHOLD})"
        self.assertGreaterEqual(score, THRESHOLD, msg)

    def test_2_add_and_remove_worker(self):
        print("Running test_2_add_and_remove_worker...")
        # DP size = 1
        self.process = popen_launch_router(
            self.model,
            self.base_url,
            dp_size=1,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            policy="round_robin",  # use round robin to make sure every worker processes requests
        )
        # 1. start a worker, and wait until it is healthy
        port = find_available_port()
        worker_url = f"http://127.0.0.1:{port}"
        worker_process = popen_launch_server(
            self.model, worker_url, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )
        self.other_process.append(worker_process)

        # 2. use /add_worker api to add it the the router. It will be used by router after it is healthy
        with requests.Session() as session:
            response = session.post(f"{self.base_url}/add_worker?url={worker_url}")
            print(f"status code: {response.status_code}, response: {response.text}")
            self.assertEqual(response.status_code, 200)

        # 3. run mmlu
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )
        metrics = run_eval(args)
        score = metrics["score"]
        THRESHOLD = 0.65
        passed = score >= THRESHOLD
        msg = f"MMLU test {'passed' if passed else 'failed'} with score {score:.3f} (threshold: {THRESHOLD})"
        self.assertGreaterEqual(score, THRESHOLD, msg)

        # 4. use /remove_worker api to remove it from the router
        with requests.Session() as session:
            response = session.post(f"{self.base_url}/remove_worker?url={worker_url}")
            print(f"status code: {response.status_code}, response: {response.text}")
            self.assertEqual(response.status_code, 200)

        # 5. run mmlu again
        metrics = run_eval(args)
        score = metrics["score"]
        THRESHOLD = 0.65
        passed = score >= THRESHOLD
        msg = f"MMLU test {'passed' if passed else 'failed'} with score {score:.3f} (threshold: {THRESHOLD})"
        self.assertGreaterEqual(score, THRESHOLD, msg)


if __name__ == "__main__":
    unittest.main()
