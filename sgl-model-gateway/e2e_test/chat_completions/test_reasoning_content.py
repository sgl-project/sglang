"""Reasoning Content E2E Tests.

Tests for chat completions with reasoning content (DeepSeek R1 reasoning parser).

Source: Migrated from e2e_grpc/features/test_reasoning_content.py
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Reasoning Content API Tests (DeepSeek 7B)
# =============================================================================


@pytest.mark.model("deepseek-7b")
@pytest.mark.gateway(
    extra_args=["--reasoning-parser", "deepseek_r1", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestReasoningContentAPI:
    """Tests for reasoning content API with DeepSeek R1 reasoning parser."""

    def test_streaming_separate_reasoning_false(self, setup_backend):
        """Test streaming with separate_reasoning=False, reasoning_content should be empty."""
        _, model, client, gateway = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            max_tokens=100,
            stream=True,
            extra_body={"separate_reasoning": False},
        )

        reasoning_content = ""
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content

        assert len(reasoning_content) == 0
        assert len(content) > 0

    def test_streaming_separate_reasoning_true(self, setup_backend):
        """Test streaming with separate_reasoning=True, reasoning_content should not be empty."""
        _, model, client, gateway = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            max_tokens=100,
            stream=True,
            extra_body={"separate_reasoning": True},
        )

        reasoning_content = ""
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content

        assert len(reasoning_content) > 0
        assert len(content) > 0

    def test_streaming_separate_reasoning_true_stream_reasoning_false(
        self, setup_backend
    ):
        """Test streaming with separate_reasoning=True and stream_reasoning=False."""
        _, model, client, gateway = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            max_tokens=100,
            stream=True,
            extra_body={"separate_reasoning": True, "stream_reasoning": False},
        )

        reasoning_content = ""
        content = ""
        first_chunk = False
        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                if not first_chunk:
                    reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if not first_chunk:
                assert (
                    not chunk.choices[0].delta.reasoning_content
                    or len(chunk.choices[0].delta.reasoning_content) == 0
                )

        assert len(reasoning_content) > 0
        assert len(content) > 0

    def test_nonstreaming_separate_reasoning_false(self, setup_backend):
        """Test non-streaming with separate_reasoning=False, reasoning_content should be empty."""
        _, model, client, gateway = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            max_tokens=100,
            extra_body={"separate_reasoning": False},
        )

        assert (
            not response.choices[0].message.reasoning_content
            or len(response.choices[0].message.reasoning_content) == 0
        )
        assert len(response.choices[0].message.content) > 0

    def test_nonstreaming_separate_reasoning_true(self, setup_backend):
        """Test non-streaming with separate_reasoning=True, reasoning_content should not be empty."""
        _, model, client, gateway = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            max_tokens=100,
            extra_body={"separate_reasoning": True},
        )

        assert len(response.choices[0].message.reasoning_content) > 0
        assert len(response.choices[0].message.content) > 0
