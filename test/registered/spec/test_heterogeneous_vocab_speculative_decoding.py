"""
Integration tests for Heterogeneous Vocabulary Speculative Decoding.

This tests the TLI (Token-Level Intersection) algorithm from the paper:
"Lossless Speculative Decoding for Heterogeneous Vocabularies" (ICML 2025 Oral)
https://arxiv.org/abs/2502.05202

The key feature is enabling speculative decoding with draft and target models
that have different vocabularies.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Use small models with different vocabularies for testing
# Qwen uses a different tokenizer than LLaMA-based models
TARGET_MODEL_HETERO_VOCAB = "Qwen/Qwen2.5-0.5B-Instruct"
DRAFT_MODEL_HETERO_VOCAB = "HuggingFaceTB/SmolLM-135M"


class TestHeterogeneousVocabSpeculativeDecoding(CustomTestCase):
    """Test speculative decoding with heterogeneous vocabularies."""

    model = TARGET_MODEL_HETERO_VOCAB
    draft_model = DRAFT_MODEL_HETERO_VOCAB
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        """Launch server with heterogeneous vocab enabled."""
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "STANDALONE",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--enable-heterogeneous-vocab",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--mem-fraction-static",
                "0.3",
                "--disable-cuda-graph",  # Disable for testing stability
                "--max-running-requests",
                "4",
            ],
            env_override={
                "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up server process."""
        kill_process_tree(cls.process.pid)

    def test_server_health(self):
        """Test that server is running and healthy."""
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)

    def test_basic_completion(self):
        """Test basic text completion with heterogeneous vocab."""
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": "default",
                "prompt": "Hello, how are you?",
                "max_tokens": 50,
                "temperature": 0,
            },
        )
        self.assertEqual(response.status_code, 200)

        result = response.json()
        text = result["choices"][0]["text"]

        # Verify output is readable text (not garbled)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

        # Check that output contains mostly printable characters
        printable_ratio = sum(c.isprintable() or c.isspace() for c in text) / len(text)
        self.assertGreater(
            printable_ratio, 0.9, f"Output seems garbled: {text[:100]}..."
        )

    def test_speculative_acceptance(self):
        """Test that speculative decoding is working (some tokens accepted)."""
        # First, flush cache for clean state
        requests.get(f"{self.base_url}/flush_cache")

        # Generate some text
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": "default",
                "prompt": "Write a short poem about the ocean:",
                "max_tokens": 100,
                "temperature": 0,
            },
        )
        self.assertEqual(response.status_code, 200)

        # Check server info for speculative decoding stats
        server_info = requests.get(f"{self.base_url}/get_server_info")
        self.assertEqual(server_info.status_code, 200)

        info = server_info.json()
        if "internal_states" in info and len(info["internal_states"]) > 0:
            internal_state = info["internal_states"][0]
            if "avg_spec_accept_length" in internal_state:
                avg_accept_length = internal_state["avg_spec_accept_length"]
                print(f"Average speculative accept length: {avg_accept_length}")
                # With heterogeneous vocab, accept rate may be lower but should be > 1
                self.assertGreater(
                    avg_accept_length,
                    1.0,
                    "Speculative decoding should accept at least some tokens",
                )

    def test_output_consistency(self):
        """Test that output is consistent with temperature=0."""
        prompt = "The capital of France is"

        # Generate twice with same settings
        responses = []
        for _ in range(2):
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": "default",
                    "prompt": prompt,
                    "max_tokens": 20,
                    "temperature": 0,
                },
            )
            self.assertEqual(response.status_code, 200)
            responses.append(response.json()["choices"][0]["text"])

        # With temperature=0, outputs should be identical
        self.assertEqual(
            responses[0],
            responses[1],
            "Deterministic outputs should match with temperature=0",
        )

    def test_chat_completion(self):
        """Test chat completion endpoint."""
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
                "max_tokens": 50,
                "temperature": 0,
            },
        )
        self.assertEqual(response.status_code, 200)

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Should mention "4" somewhere in the response
        self.assertIn("4", content, f"Expected '4' in response: {content}")


class TestHeterogeneousVocabCorrectness(CustomTestCase):
    """Test output correctness compared to non-speculative decoding."""

    model = TARGET_MODEL_HETERO_VOCAB
    draft_model = DRAFT_MODEL_HETERO_VOCAB
    base_url = DEFAULT_URL_FOR_TEST
    base_url_reference = "http://127.0.0.1:30001"

    @classmethod
    def setUpClass(cls):
        """Launch two servers: one with spec decoding, one without."""
        # Server with heterogeneous vocab speculative decoding
        cls.process_spec = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "STANDALONE",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--enable-heterogeneous-vocab",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--mem-fraction-static",
                "0.15",
                "--disable-cuda-graph",
                "--max-running-requests",
                "4",
            ],
            env_override={
                "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            },
        )

        # Reference server without speculative decoding
        cls.process_ref = popen_launch_server(
            cls.model,
            cls.base_url_reference,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--mem-fraction-static",
                "0.15",
                "--disable-cuda-graph",
                "--max-running-requests",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up both server processes."""
        kill_process_tree(cls.process_spec.pid)
        kill_process_tree(cls.process_ref.pid)

    def test_output_matches_reference(self):
        """Test that speculative decoding output matches non-speculative output."""
        prompt = "Explain briefly what machine learning is:"

        # Get output from speculative decoding server
        response_spec = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": "default",
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0,
            },
        )
        self.assertEqual(response_spec.status_code, 200)
        text_spec = response_spec.json()["choices"][0]["text"]

        # Get output from reference server
        response_ref = requests.post(
            f"{self.base_url_reference}/v1/completions",
            json={
                "model": "default",
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0,
            },
        )
        self.assertEqual(response_ref.status_code, 200)
        text_ref = response_ref.json()["choices"][0]["text"]

        # Outputs should be identical (lossless speculative decoding)
        self.assertEqual(
            text_spec,
            text_ref,
            f"Speculative output should match reference.\n"
            f"Spec: {text_spec}\n"
            f"Ref: {text_ref}",
        )


if __name__ == "__main__":
    unittest.main()
