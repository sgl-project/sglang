import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")
register_xpu_ci(est_time=90, suite="stage-b-test-1-gpu-xpu")


class TestEmbeddingLengthValidation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=("--is-embedding", "--max-total-tokens", "1000"),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _embed(self, text: str) -> requests.Response:
        return requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": text},
            timeout=60,
        )

    def test_input_longer_than_kv_pool_is_rejected(self):
        """An embedding prompt that fits the context length but not the KV pool used to hang the request."""
        response = self._embed("hello " * 1500)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("exceeds the maximum allowed length", response.text)

        response = self._embed("hello world")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(len(response.json()["data"][0]["embedding"]), 0)


if __name__ == "__main__":
    unittest.main()
