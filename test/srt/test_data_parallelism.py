import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestDataParallelism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--dp", "2"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid, include_self=True)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)

    def test_update_weight(self):
        response = requests.post(
            self.base_url + "/update_weights",
            json={"model_path": DEFAULT_MODEL_NAME_FOR_TEST},
        )

        # check if the response is 200
        assert response.status_code == 200

        # pause a few seconds then send again
        time.sleep(5)

        response = requests.post(
            self.base_url + "/update_weights",
            json={"model_path": DEFAULT_MODEL_NAME_FOR_TEST},
        )

        # check if the response is 200
        assert response.status_code == 200

    def test_get_memory_pool_size(self):
        # use `get_server_info` instead since `get_memory_pool_size` is merged into `get_server_info`
        response = requests.get(self.base_url + "/get_server_info")
        assert response.status_code == 200

        time.sleep(5)

        response = requests.get(self.base_url + "/get_server_info")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
