"""
Integration tests for top-k attention token visualization feature.

Tests the end-to-end flow of capturing and returning attention tokens
through the OpenAI-compatible API (both completions and chat completions).

Usage:
    python -m pytest test/srt/test_attention_tokens.py -v
    python -m unittest test.srt.test_attention_tokens.TestAttentionTokens
"""

import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAttentionTokens(CustomTestCase):
    """Test attention token capture and return through OpenAI API."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Launch server with attention token capture enabled
        # Use triton backend which has attention token capture support
        # Disable CUDA graphs to avoid capture issues with attention token extraction
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--return-attention-tokens",
                "--attention-tokens-top-k",
                "5",
                "--attention-backend",
                "triton",
                "--disable-cuda-graph",
            ],
        )
        cls.base_url_v1 = cls.base_url + "/v1"
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url_v1)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_completion_attention_tokens(self):
        """Test attention tokens in completions API (non-streaming)."""
        response = self.client.completions.create(
            model=self.model,
            prompt="The capital of France is",
            max_tokens=10,
            temperature=0,
            extra_body={"return_attention_tokens": True},
        )

        self.assertEqual(len(response.choices), 1)
        choice = response.choices[0]

        # Check that attention_tokens is present and has expected structure
        self.assertIsNotNone(
            choice.attention_tokens,
            "attention_tokens should be present in response",
        )
        self.assertIsInstance(choice.attention_tokens, list)

        # Each element should have token_positions, attention_scores, and layer_id
        if len(choice.attention_tokens) > 0:
            token_info = choice.attention_tokens[0]
            self.assertIn("token_positions", token_info)
            self.assertIn("attention_scores", token_info)
            self.assertIn("layer_id", token_info)
            self.assertIsInstance(token_info["token_positions"], list)
            self.assertIsInstance(token_info["attention_scores"], list)
            # Should have top_k entries (or fewer if sequence is shorter)
            # Default top_k_attention is 10 per the protocol
            self.assertLessEqual(len(token_info["token_positions"]), 10)

    def test_completion_attention_tokens_disabled(self):
        """Test that attention_tokens is not returned when not requested."""
        response = self.client.completions.create(
            model=self.model,
            prompt="Hello world",
            max_tokens=5,
            temperature=0,
        )

        self.assertEqual(len(response.choices), 1)
        # attention_tokens should not be in response when not requested
        self.assertIsNone(
            getattr(response.choices[0], "attention_tokens", None),
            "attention_tokens should not be present when not requested",
        )

    def test_chat_completion_attention_tokens(self):
        """Test attention tokens in chat completions API (non-streaming)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=10,
            temperature=0,
            extra_body={"return_attention_tokens": True},
        )

        self.assertEqual(len(response.choices), 1)
        choice = response.choices[0]

        # Check that attention_tokens is present
        self.assertIsNotNone(
            choice.attention_tokens,
            "attention_tokens should be present in chat completion response",
        )
        self.assertIsInstance(choice.attention_tokens, list)

    def test_completion_stream_attention_tokens(self):
        """Test attention tokens in completions API (streaming)."""
        stream = self.client.completions.create(
            model=self.model,
            prompt="The capital of France is",
            max_tokens=10,
            temperature=0,
            stream=True,
            extra_body={"return_attention_tokens": True},
        )

        chunks = list(stream)
        self.assertGreater(len(chunks), 0)

        # Find the chunk with attention_tokens (should be near the end)
        attention_chunks = [
            c
            for c in chunks
            if c.choices
            and len(c.choices) > 0
            and getattr(c.choices[0], "attention_tokens", None) is not None
        ]

        # At least one chunk should have attention tokens
        self.assertGreater(
            len(attention_chunks),
            0,
            "At least one streaming chunk should contain attention_tokens",
        )

        # Verify structure
        attn_chunk = attention_chunks[0]
        self.assertIsInstance(attn_chunk.choices[0].attention_tokens, list)

    def test_chat_completion_stream_attention_tokens(self):
        """Test attention tokens in chat completions API (streaming)."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=10,
            temperature=0,
            stream=True,
            extra_body={"return_attention_tokens": True},
        )

        chunks = list(stream)
        self.assertGreater(len(chunks), 0)

        # Find chunks with attention_tokens in delta
        attention_chunks = [
            c
            for c in chunks
            if c.choices
            and len(c.choices) > 0
            and getattr(c.choices[0].delta, "attention_tokens", None) is not None
        ]

        # At least one chunk should have attention tokens
        self.assertGreater(
            len(attention_chunks),
            0,
            "At least one streaming chunk should contain attention_tokens in delta",
        )

    def test_attention_tokens_multiple_tokens(self):
        """Test that attention tokens are captured for multiple generated tokens."""
        response = self.client.completions.create(
            model=self.model,
            prompt="Count: 1, 2, 3,",
            max_tokens=20,
            temperature=0,
            extra_body={"return_attention_tokens": True},
        )

        choice = response.choices[0]
        self.assertIsNotNone(choice.attention_tokens)

        # Should have attention info for each generated token
        # (number of attention_tokens entries = number of decode steps)
        self.assertGreater(
            len(choice.attention_tokens),
            1,
            "Should have attention info for multiple tokens",
        )


if __name__ == "__main__":
    unittest.main()
