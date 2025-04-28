"""
Test the tokenizer API endpoints.

Run with:
python3 -m unittest test_tokenizer_api.TestTokenizerAPI
"""

import json
import unittest
import requests

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestTokenizerAPI(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_tokenize_simple(self):
        """Test tokenizing a simple string."""
        text = "Hello, world!"
        response = requests.post(
            f"{self.base_url}/tokenize",
            json={"text": text}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Compare with local tokenizer
        expected_tokens = self.tokenizer.encode(text)
        
        self.assertIn("tokens", data)
        self.assertIsInstance(data["tokens"], list)
        self.assertEqual(data["tokens"], expected_tokens)
        
        self.assertIn("count", data)
        self.assertEqual(data["count"], len(expected_tokens))

    def test_tokenize_empty(self):
        """Test tokenizing an empty string."""
        text = ""
        response = requests.post(
            f"{self.base_url}/tokenize",
            json={"text": text}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        expected_tokens = self.tokenizer.encode(text)
        
        self.assertIn("tokens", data)
        self.assertEqual(data["tokens"], expected_tokens)
        self.assertEqual(data["count"], len(expected_tokens))

    def test_tokenize_long(self):
        """Test tokenizing a longer text."""
        text = "This is a longer text that should be tokenized properly. " * 10
        response = requests.post(
            f"{self.base_url}/tokenize",
            json={"text": text}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        expected_tokens = self.tokenizer.encode(text)
        
        self.assertIn("tokens", data)
        self.assertEqual(data["tokens"], expected_tokens)
        self.assertEqual(data["count"], len(expected_tokens))

    def test_tokenize_invalid(self):
        """Test tokenizing with invalid input."""
        response = requests.post(
            f"{self.base_url}/tokenize",
            json={"text": 123}  # Not a string
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_detokenize_simple(self):
        """Test detokenizing a simple token list."""
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)
        
        response = requests.post(
            f"{self.base_url}/detokenize",
            json={"tokens": tokens}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("text", data)
        self.assertEqual(data["text"], text)

    def test_detokenize_empty(self):
        """Test detokenizing an empty token list."""
        tokens = []
        
        response = requests.post(
            f"{self.base_url}/detokenize",
            json={"tokens": tokens}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("text", data)
        self.assertEqual(data["text"], "")

    def test_detokenize_invalid_format(self):
        """Test detokenizing with invalid input format."""
        response = requests.post(
            f"{self.base_url}/detokenize",
            json={"tokens": "not_a_list"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_detokenize_invalid_token(self):
        """Test detokenizing with invalid token type."""
        response = requests.post(
            f"{self.base_url}/detokenize",
            json={"tokens": [1, 2, "not_an_int", 4]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_roundtrip(self):
        """Test tokenize followed by detokenize roundtrip."""
        original_text = "This is a test of the tokenizer API roundtrip functionality."
        
        # First tokenize
        tokenize_response = requests.post(
            f"{self.base_url}/tokenize",
            json={"text": original_text}
        )
        self.assertEqual(tokenize_response.status_code, 200)
        tokens = tokenize_response.json()["tokens"]
        
        # Then detokenize
        detokenize_response = requests.post(
            f"{self.base_url}/detokenize",
            json={"tokens": tokens}
        )
        self.assertEqual(detokenize_response.status_code, 200)
        reconstructed_text = detokenize_response.json()["text"]
        
        # Compare original and reconstructed text
        self.assertEqual(reconstructed_text, original_text)


if __name__ == "__main__":
    unittest.main() 