"""
XPU embedding server test: validates the OpenAI-compatible /v1/embeddings
endpoint on Intel XPU using a small embedding model. Lives in its own file
because embedding models load with --is-embedding and use a different model
than the chat fixtures in test_xpu_serving_features.py.

Usage:
python3 -m unittest test_xpu_embedding.TestXPUEmbedding
"""

import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=120, suite="stage-b-test-1-gpu-xpu")


class TestXPUEmbedding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--is-embedding",
                "--device",
                "xpu",
            ],
        )
        cls.openai_url = cls.base_url + "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _client(self) -> openai.Client:
        # Server has no API key, but openai client still requires a non-empty string.
        return openai.Client(api_key="EMPTY", base_url=self.openai_url)

    def test_embedding_single(self):
        response = self._client().embeddings.create(
            model=self.model, input="Hello world"
        )
        self.assertEqual(len(response.data), 1)
        self.assertGreater(len(response.data[0].embedding), 0)

    def test_embedding_batch(self):
        response = self._client().embeddings.create(
            model=self.model, input=["Hello world", "Test text"]
        )
        self.assertEqual(len(response.data), 2)
        self.assertGreater(len(response.data[0].embedding), 0)
        self.assertGreater(len(response.data[1].embedding), 0)


if __name__ == "__main__":
    unittest.main()
