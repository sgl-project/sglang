import json
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=70, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=141, suite="stage-b-test-small-1-gpu-amd")


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

    def test_embedding_with_dimensions_parameter(self):
        """Test that non-Matryoshka models reject dimensions parameter."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Test that specifying dimensions fails for non-Matryoshka models
        with self.assertRaises(openai.BadRequestError) as cm:
            client.embeddings.create(
                model=self.model, input="Hello world", dimensions=512
            )

        self.assertEqual(cm.exception.status_code, 400)


class TestMatryoshkaEmbeddingModel(CustomTestCase):
    """Test class for Model that supports Matryoshka embedding functionality, using OpenAI API."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.matryoshka_dims = [128, 256, 512, 768, 1024]

        # Configure embedding-specific args with Matryoshka support via json_model_override_args
        matryoshka_config = {
            "is_matryoshka": True,
            "matryoshka_dimensions": cls.matryoshka_dims,
        }
        other_args = [
            "--is-embedding",
            "--enable-metrics",
            "--json-model-override-args",
            json.dumps(matryoshka_config),
        ]
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
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_matryoshka_embedding_valid_dimensions(self):
        """Test Matryoshka embedding with valid dimensions."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Test with various valid dimensions
        for dimensions in self.matryoshka_dims:
            with self.subTest(dimensions=dimensions):
                response = client.embeddings.create(
                    model=self.model, input="Hello world", dimensions=dimensions
                )
                self.assertEqual(len(response.data), 1)
                self.assertEqual(len(response.data[0].embedding), dimensions)

    def test_matryoshka_embedding_batch_same_dimensions(self):
        """Test Matryoshka embedding with batch input and same dimensions."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.embeddings.create(
            model=self.model,
            input=["Hello world", "Test text", "Another example"],
            dimensions=256,
        )

        self.assertEqual(len(response.data), 3)
        for embedding_data in response.data:
            self.assertEqual(len(embedding_data.embedding), 256)

    def test_matryoshka_embedding_no_dimensions(self):
        """Test embedding without specifying dimensions (should use full size)."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.embeddings.create(model=self.model, input="Hello world")

        self.assertEqual(len(response.data), 1)

        # Should return full embedding size when no dimensions specified
        self.assertEqual(len(response.data[0].embedding), 1536)

    def test_matryoshka_embedding_invalid_dimensions(self):
        """Test Matryoshka embedding with invalid dimensions."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        for dimensions in [100, 0, -1, 10000]:
            with self.assertRaises(openai.BadRequestError) as cm:
                client.embeddings.create(
                    model=self.model,
                    input="Hello world",
                    dimensions=dimensions,
                )
            self.assertEqual(cm.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
