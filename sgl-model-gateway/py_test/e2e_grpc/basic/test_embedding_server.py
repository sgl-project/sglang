"""
gRPC Router E2E Test - Embedding Server

Test the embedding functionality of the gRPC router.
"""

import sys
import unittest
from pathlib import Path

import openai

_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR.parent))
from fixtures import popen_launch_workers_and_router
from util import (
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
)


class TestEmbeddingServer(CustomTestCase):
    """
    Test Embedding API through gRPC router.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_EMBEDDING_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Launch workers with --is-embedding flag
        cls.cluster = popen_launch_workers_and_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            num_workers=1,
            tp_size=1,
            policy="round_robin",
            api_key=cls.api_key,
            worker_args=["--is-embedding"],
        )

        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        # Cleanup router and workers
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

    def test_embedding(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        input_text = "Hello world"
        response = client.embeddings.create(
            model=self.model,
            input=input_text,
        )

        assert response.object == "list"
        assert len(response.data) == 1
        embedding = response.data[0]
        assert embedding.object == "embedding"
        assert embedding.index == 0
        assert len(embedding.embedding) > 0
        assert isinstance(embedding.embedding[0], float)

        # Verify usage statistics
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens

    def test_embedding_batch(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        input_texts = ["Hello world", "SGLang is fast"]
        response = client.embeddings.create(
            model=self.model,
            input=input_texts,
        )

        assert len(response.data) == 1
        assert response.data[0].index == 0
        assert len(response.data[0].embedding) > 0

    def test_embedding_dimensions(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response1 = client.embeddings.create(
            model=self.model,
            input="A short text",
        )
        dim1 = len(response1.data[0].embedding)

        response2 = client.embeddings.create(
            model=self.model,
            input="A much longer text to ensure dimensions match",
        )
        dim2 = len(response2.data[0].embedding)

        assert dim1 == dim2


if __name__ == "__main__":
    unittest.main()
