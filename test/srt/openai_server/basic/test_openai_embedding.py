import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestOpenAIEmbedding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Configure embedding-specific args
        other_args = ["--is-embedding", "--enable-metrics"]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_embedding_single(self):
        """Test single embedding request"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(model=self.model, input="Hello world")
        self.assertEqual(len(response.data), 1)
        self.assertTrue(len(response.data[0].embedding) > 0)

    def test_embedding_batch(self):
        """Test batch embedding request"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(
            model=self.model, input=["Hello world", "Test text"]
        )
        self.assertEqual(len(response.data), 2)
        self.assertTrue(len(response.data[0].embedding) > 0)
        self.assertTrue(len(response.data[1].embedding) > 0)

    def test_embedding_single_batch_str(self):
        """Test embedding with a List[str] and length equals to 1"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(model=self.model, input=["Hello world"])
        self.assertEqual(len(response.data), 1)
        self.assertTrue(len(response.data[0].embedding) > 0)

    def test_embedding_single_int_list(self):
        """Test embedding with a List[int] or List[List[int]]]"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(
            model=self.model,
            input=[[15339, 314, 703, 284, 612, 262, 10658, 10188, 286, 2061]],
        )
        self.assertEqual(len(response.data), 1)
        self.assertTrue(len(response.data[0].embedding) > 0)

        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(
            model=self.model,
            input=[15339, 314, 703, 284, 612, 262, 10658, 10188, 286, 2061],
        )
        self.assertEqual(len(response.data), 1)
        self.assertTrue(len(response.data[0].embedding) > 0)

    def test_empty_string_embedding(self):
        """Test embedding an empty string."""

        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Text embedding example with empty string
        text = ""
        # Expect a BadRequestError for empty input
        with self.assertRaises(openai.BadRequestError) as cm:
            client.embeddings.create(
                model=self.model,
                input=text,
            )
        # check the status code
        self.assertEqual(cm.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
