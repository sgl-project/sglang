import json
import os
import unittest

import numpy as np
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


# Path to the local embedding model
LOCAL_EMBEDDING_MODEL_PATH = "/shared/public/elr-models/intfloat/e5-large-v2/f169b11e22de13617baa190a028a32f3493550b6/"


@unittest.skipUnless(
    os.path.exists(LOCAL_EMBEDDING_MODEL_PATH),
    f"Local model path {LOCAL_EMBEDDING_MODEL_PATH} not found",
)
class TestMatryoshkaEmbedding(CustomTestCase):
    """Test class for Matryoshka embedding functionality using local model."""

    @classmethod
    def setUpClass(cls):
        cls.model = LOCAL_EMBEDDING_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Configure embedding-specific args with Matryoshka support via json_model_override_args
        matryoshka_config = {
            "is_matryoshka": True,
            "matryoshka_dimensions": [128, 256, 512, 768, 1024],
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
        for dimensions in [128, 256, 512, 768, 1024]:
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
        # For e5-large-v2, the embedding size is 1024
        self.assertEqual(len(response.data[0].embedding), 1024)

    def test_matryoshka_embedding_invalid_dimensions(self):
        """Test Matryoshka embedding with invalid dimensions."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Test dimensions not in supported list
        with self.assertRaises(openai.BadRequestError):
            client.embeddings.create(
                model=self.model,
                input="Hello world",
                dimensions=100,  # Not in supported dimensions list
            )

    def test_matryoshka_embedding_zero_dimensions(self):
        """Test Matryoshka embedding with zero or negative dimensions."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Test zero dimensions
        with self.assertRaises(openai.BadRequestError):
            client.embeddings.create(
                model=self.model, input="Hello world", dimensions=0
            )

        # Test negative dimensions
        with self.assertRaises(openai.BadRequestError):
            client.embeddings.create(
                model=self.model, input="Hello world", dimensions=-1
            )

    def test_matryoshka_embedding_too_large_dimensions(self):
        """Test Matryoshka embedding with dimensions larger than model size."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Test dimensions larger than model's hidden size
        with self.assertRaises(openai.BadRequestError):
            client.embeddings.create(
                model=self.model,
                input="Hello world",
                dimensions=10000,  # Much larger than typical embedding size
            )

    def test_matryoshka_embedding_batch_mixed_dimensions_not_supported(self):
        """Test that mixed dimensions in batch requests are handled appropriately."""
        # Note: The current implementation may not support different dimensions
        # per item in a batch. This test documents the expected behavior.
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # This test just ensures we can handle batch requests with dimensions
        # The API currently doesn't support per-item dimensions in a batch
        response = client.embeddings.create(
            model=self.model, input=["Text 1", "Text 2"], dimensions=512
        )

        self.assertEqual(len(response.data), 2)
        for embedding_data in response.data:
            self.assertEqual(len(embedding_data.embedding), 512)


@unittest.skipUnless(
    os.path.exists(LOCAL_EMBEDDING_MODEL_PATH),
    f"Local model path {LOCAL_EMBEDDING_MODEL_PATH} not found",
)
class TestNonMatryoshkaEmbedding(CustomTestCase):
    """Test class for embedding functionality with models that don't support Matryoshka."""

    @classmethod
    def setUpClass(cls):
        cls.model = LOCAL_EMBEDDING_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Configure embedding-specific args WITHOUT Matryoshka support
        # This simulates a model that doesn't support Matryoshka embeddings
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
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_non_matryoshka_embedding_with_dimensions_parameter(self):
        """Test that non-Matryoshka models reject dimensions parameter."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Test that specifying dimensions fails for non-Matryoshka models
        with self.assertRaises(openai.BadRequestError) as cm:
            client.embeddings.create(
                model=self.model, input="Hello world", dimensions=512
            )

        # Verify it's a 400 error (Bad Request)
        self.assertEqual(cm.exception.status_code, 400)

    def test_non_matryoshka_embedding_without_dimensions(self):
        """Test that non-Matryoshka models work normally without dimensions parameter."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.embeddings.create(model=self.model, input="Hello world")

        self.assertEqual(len(response.data), 1)
        # Should return full embedding size (1024 for e5-large-v2)
        self.assertEqual(len(response.data[0].embedding), 1024)

    def test_non_matryoshka_embedding_batch_with_dimensions(self):
        """Test that non-Matryoshka models reject dimensions parameter in batch requests."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Test that specifying dimensions fails for batch requests too
        with self.assertRaises(openai.BadRequestError) as cm:
            client.embeddings.create(
                model=self.model, input=["Text 1", "Text 2"], dimensions=256
            )

        # Verify it's a 400 error (Bad Request)
        self.assertEqual(cm.exception.status_code, 400)

    def test_non_matryoshka_embedding_batch_without_dimensions(self):
        """Test that non-Matryoshka models work normally for batch requests without dimensions."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.embeddings.create(
            model=self.model, input=["Hello world", "Test text", "Another example"]
        )

        self.assertEqual(len(response.data), 3)
        for embedding_data in response.data:
            # Should return full embedding size (1024 for e5-large-v2)
            self.assertEqual(len(embedding_data.embedding), 1024)


if __name__ == "__main__":
    unittest.main()
