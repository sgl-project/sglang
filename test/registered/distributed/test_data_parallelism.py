import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=78, suite="stage-b-test-2-gpu-large")
register_amd_ci(est_time=73, suite="stage-b-test-2-gpu-large-amd")


class TestDataParallelism(CustomTestCase, GSM8KMixin):
    gsm8k_accuracy_thres = 0.7

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--dp", 2],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_update_weight(self):
        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={"model_path": DEFAULT_MODEL_NAME_FOR_TEST},
        )

        # check if the response is 200
        assert response.status_code == 200

        # pause a few seconds then send again
        time.sleep(1)

        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={"model_path": DEFAULT_MODEL_NAME_FOR_TEST},
        )

        # check if the response is 200
        assert response.status_code == 200

    def test_get_memory_pool_size(self):
        # use `server_info` instead since `get_memory_pool_size` is merged into `server_info`
        response = requests.get(self.base_url + "/server_info")
        assert response.status_code == 200

        time.sleep(1)

        response = requests.get(self.base_url + "/server_info")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
