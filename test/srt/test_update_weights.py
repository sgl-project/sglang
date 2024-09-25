import json
import unittest

import requests

from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestUpdateWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def run_decode(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        print(json.dumps(response.json()))
        print("=" * 100)
        text = response.json()["text"]
        return text

    def get_model_info(self):
        response = requests.get(self.base_url + "/get_model_info")
        model_path = response.json()["model_path"]
        print(json.dumps(response.json()))
        return model_path

    def run_update_weights(self, model_path):
        response = requests.post(
            self.base_url + "/update_weights",
            json={
                "model_path": model_path,
            },
        )
        ret = response.json()
        print(json.dumps(response.json()))
        return ret

    def test_update_weights(self):
        origin_model_path = self.get_model_info()
        print(f"origin_model_path: {origin_model_path}")
        origin_response = self.run_decode()

        # update weights
        new_model_path = "meta-llama/Meta-Llama-3.1-8B"
        ret = self.run_update_weights(new_model_path)
        assert ret["success"]

        updated_model_path = self.get_model_info()
        print(f"updated_model_path: {updated_model_path}")
        assert updated_model_path == new_model_path
        assert updated_model_path != origin_model_path

        updated_response = self.run_decode()
        assert origin_response[:32] != updated_response[:32]

        # update weights back
        ret = self.run_update_weights(origin_model_path)
        assert ret["success"]

        updated_model_path = self.get_model_info()
        assert updated_model_path == origin_model_path

        updated_response = self.run_decode()
        assert origin_response[:32] == updated_response[:32]

    def test_update_weights_unexist_model(self):
        origin_model_path = self.get_model_info()
        print(f"origin_model_path: {origin_model_path}")
        origin_response = self.run_decode()

        # update weights
        new_model_path = "meta-llama/Meta-Llama-3.1-8B-1"
        ret = self.run_update_weights(new_model_path)
        assert not ret["success"]

        updated_model_path = self.get_model_info()
        print(f"updated_model_path: {updated_model_path}")
        assert updated_model_path == origin_model_path

        updated_response = self.run_decode()
        assert origin_response[:32] == updated_response[:32]


if __name__ == "__main__":
    unittest.main()
