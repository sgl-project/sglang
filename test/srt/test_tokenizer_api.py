"""
Test the tokenizer API endpoints.

Run with:
python3 -m unittest sglang/test/srt/test_tokenizer_api.py
or directly:
python3 sglang/test/srt/test_tokenizer_api.py
"""

import json
import os
import sys
import unittest

import requests

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
        cls.api_key = None  # 如果需要API密钥，这里设置
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
        )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_tokenize_simple(self):
        """Test tokenizing a simple string."""
        text = "Hello, world!"
        response = requests.post(f"{self.base_url}/tokenize", json={"text": text})

        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
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
        response = requests.post(f"{self.base_url}/tokenize", json={"text": text})
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()

        expected_tokens = self.tokenizer.encode(text)

        self.assertIn("tokens", data)
        self.assertEqual(data["tokens"], expected_tokens)
        self.assertEqual(data["count"], len(expected_tokens))

    def test_tokenize_long(self):
        """Test tokenizing a longer text."""
        text = "This is a longer text that should be tokenized properly. " * 10
        response = requests.post(f"{self.base_url}/tokenize", json={"text": text})
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()

        expected_tokens = self.tokenizer.encode(text)

        self.assertIn("tokens", data)
        self.assertEqual(data["tokens"], expected_tokens)
        self.assertEqual(data["count"], len(expected_tokens))

    def test_tokenize_invalid(self):
        """Test tokenizing with invalid input."""
        response = requests.post(
            f"{self.base_url}/tokenize", json={"text": 123}  # Not a string
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_detokenize_simple(self):
        """Test detokenizing a simple token list."""
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)

        response = requests.post(f"{self.base_url}/detokenize", json={"tokens": tokens})
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()

        self.assertIn("text", data)
        # 比较时忽略开头的特殊标记
        self.assertEqual(data["text"].strip(), text.strip())

    def test_detokenize_empty(self):
        """Test detokenizing an empty token list."""
        tokens = []

        response = requests.post(f"{self.base_url}/detokenize", json={"tokens": tokens})
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()

        self.assertIn("text", data)
        self.assertEqual(data["text"], "")

    def test_detokenize_invalid_format(self):
        """Test detokenizing with invalid input format."""
        response = requests.post(
            f"{self.base_url}/detokenize", json={"tokens": "not_a_list"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_detokenize_invalid_token(self):
        """Test detokenizing with invalid token type."""
        response = requests.post(
            f"{self.base_url}/detokenize", json={"tokens": [1, 2, "not_an_int", 4]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_detokenize_keep_special_tokens(self):
        """Test detokenizing with the option to keep special tokens."""
        # 先获取一个包含特殊标记的示例
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)

        # 正常情况下，特殊标记会被移除
        response = requests.post(f"{self.base_url}/detokenize", json={"tokens": tokens})
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data_without_special = response.json()

        # 使用keep_special_tokens=True选项
        response = requests.post(
            f"{self.base_url}/detokenize",
            json={"tokens": tokens, "keep_special_tokens": True},
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data_with_special = response.json()

        # 对于某些模型，这两者可能会有所不同
        # 这里我们只检查响应格式是否正确
        self.assertIn("text", data_with_special)
        self.assertIsInstance(data_with_special["text"], str)

        # 如果模型确实添加了特殊标记，两个结果应该不同
        # 但由于测试可能使用不同的模型，我们只在有区别时断言
        if data_with_special["text"] != data_without_special["text"]:
            # 验证保留特殊标记的版本比不保留的版本更长或包含更多内容
            special_tokens = [
                "<|begin_of_text|>",
                "<|endoftext|>",
                "<s>",
                "</s>",
                "<pad>",
                "[CLS]",
                "[SEP]",
                "[PAD]",
                "[MASK]",
                "<bos>",
                "<eos>",
            ]
            has_special_token = any(
                token in data_with_special["text"] for token in special_tokens
            )
            self.assertTrue(
                has_special_token,
                f"Expected special tokens in: {data_with_special['text']}",
            )

    def test_roundtrip(self):
        """Test tokenize followed by detokenize roundtrip."""
        original_text = "This is a test of the tokenizer API roundtrip functionality."

        # First tokenize
        tokenize_response = requests.post(
            f"{self.base_url}/tokenize", json={"text": original_text}
        )
        self.assertEqual(
            tokenize_response.status_code, 200, f"Failed with: {tokenize_response.text}"
        )
        tokens = tokenize_response.json()["tokens"]

        # Then detokenize
        detokenize_response = requests.post(
            f"{self.base_url}/detokenize", json={"tokens": tokens}
        )
        self.assertEqual(
            detokenize_response.status_code,
            200,
            f"Failed with: {detokenize_response.text}",
        )
        reconstructed_text = detokenize_response.json()["text"]

        # Compare original and reconstructed text (ignore any special tokens)
        self.assertEqual(reconstructed_text.strip(), original_text.strip())


if __name__ == "__main__":
    unittest.main()
