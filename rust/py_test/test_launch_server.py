import subprocess
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
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
):
    """
    Launch the router server process.

    Args:
        model: Model path/name
        base_url: Server base URL
        dp_size: Data parallel size
        timeout: Server launch timeout
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
    ]

    # Use current environment
    env = None

    process = subprocess.Popen(command, stdout=None, stderr=None, env=env)

    start_time = time.time()
    with requests.Session() as session:
        while time.time() - start_time < timeout:
            try:
                response = session.get(f"{base_url}/health")
                if response.status_code == 200:
                    return process
            except requests.RequestException:
                pass
            time.sleep(10)

    raise TimeoutError("Server failed to start within the timeout period.")


class TestEvalAccuracyMini(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_router(
            cls.model,
            cls.base_url,
            dp_size=1,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)


if __name__ == "__main__":
    unittest.main()
