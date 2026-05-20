"""Enable Thinking E2E Tests.

Tests for chat completions with enable_thinking feature (Qwen3 reasoning).

Source: Migrated from e2e_grpc/features/test_enable_thinking.py
"""

from __future__ import annotations

import json
import logging

import pytest
import requests

logger = logging.getLogger(__name__)

# API key is not validated by the gateway, but required for OpenAI-compatible headers
API_KEY = "not-used"


# =============================================================================
# Enable Thinking Tests (Qwen 30B)
# =============================================================================


@pytest.mark.model("qwen-30b")
@pytest.mark.gateway(
    extra_args=["--reasoning-parser", "qwen3", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestEnableThinking:
    """Tests for enable_thinking feature with Qwen3 reasoning parser."""

    def test_chat_completion_with_reasoning(self, setup_backend):
        """Test non-streaming with enable_thinking=True, reasoning_content should not be empty."""
        _, model, client, gateway = setup_backend

        response = requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )

        assert response.status_code == 200, f"Failed with: {response.text}"
        data = response.json()

        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "reasoning_content" in data["choices"][0]["message"]
        assert data["choices"][0]["message"]["reasoning_content"] is not None

    def test_chat_completion_without_reasoning(self, setup_backend):
        """Test non-streaming with enable_thinking=False, reasoning_content should be empty."""
        _, model, client, gateway = setup_backend

        response = requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        assert response.status_code == 200, f"Failed with: {response.text}"
        data = response.json()

        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]

        if "reasoning_content" in data["choices"][0]["message"]:
            assert data["choices"][0]["message"]["reasoning_content"] is None

    def test_stream_chat_completion_with_reasoning(self, setup_backend):
        """Test streaming with enable_thinking=True, reasoning_content should not be empty."""
        _, model, client, gateway = setup_backend

        response = requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "separate_reasoning": True,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            stream=True,
        )

        assert response.status_code == 200, f"Failed with: {response.text}"

        has_reasoning = False
        has_content = False

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})

                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            has_reasoning = True

                        if "content" in delta and delta["content"]:
                            has_content = True

        assert (
            has_reasoning
        ), "The reasoning content is not included in the stream response"
        assert has_content, "The stream response does not contain normal content"

    def test_stream_chat_completion_without_reasoning(self, setup_backend):
        """Test streaming with enable_thinking=False, reasoning_content should be empty."""
        _, model, client, gateway = setup_backend

        response = requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "separate_reasoning": True,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stream=True,
        )

        assert response.status_code == 200, f"Failed with: {response.text}"

        has_reasoning = False
        has_content = False

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})

                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            has_reasoning = True

                        if "content" in delta and delta["content"]:
                            has_content = True

        assert (
            not has_reasoning
        ), "The reasoning content should not be included in the stream response"
        assert has_content, "The stream response does not contain normal content"
