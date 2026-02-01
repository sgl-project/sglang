"""
Tests for Anthropic tool_choice variations.

Tests different tool_choice modes: auto, any, and specific tool.
"""

import os

import anthropic
import httpx
import pytest


def get_base_url() -> str:
    """Get base URL from environment, normalizing /v1 suffix."""
    url = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:8000")
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


@pytest.fixture
def client():
    """Create Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy-key")
    http_client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"})
    return anthropic.Anthropic(
        base_url=get_base_url(),
        api_key=api_key,
        http_client=http_client,
    )


@pytest.fixture
def model_name():
    """Model name for tests."""
    return os.environ.get("TEST_MODEL_NAME", "default")


@pytest.fixture
def weather_tool():
    """Weather tool definition."""
    return {
        "name": "get_weather",
        "description": "Get the current weather for a specific location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g. 'Paris, France'",
                }
            },
            "required": ["location"],
        },
    }


@pytest.fixture
def calculator_tool():
    """Calculator tool definition."""
    return {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    }


class TestToolChoice:
    """Test tool_choice variations."""

    def test_messages_create__forces_tool_use_when_tool_choice_any(
        self, client, model_name, weather_tool, calculator_tool
    ):
        """Test that tool_choice='any' forces the model to use a tool."""
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool, calculator_tool],
            tool_choice={"type": "any"},
            messages=[
                {
                    "role": "user",
                    "content": "Hello, how are you today?",
                }
            ],
        )

        tool_uses = [c for c in response.content if c.type == "tool_use"]
        assert len(tool_uses) > 0, "tool_choice='any' should force tool use"
        assert response.stop_reason == "tool_use"

    def test_messages_create__uses_specified_tool_when_tool_choice_tool(
        self, client, model_name, weather_tool, calculator_tool
    ):
        """Test that tool_choice with specific tool name forces that tool."""
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool, calculator_tool],
            tool_choice={"type": "tool", "name": "calculate"},
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Paris?",
                }
            ],
        )

        tool_uses = [c for c in response.content if c.type == "tool_use"]
        assert (
            len(tool_uses) > 0
        ), "tool_choice with specific tool should force tool use"
        assert tool_uses[0].name == "calculate", "Should use the specified tool"
        assert response.stop_reason == "tool_use"

    def test_messages_create__allows_text_response_when_tool_choice_auto(
        self, client, model_name, weather_tool
    ):
        """Test that tool_choice='auto' allows model to not use tools."""
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool],
            tool_choice={"type": "auto"},
            messages=[
                {
                    "role": "user",
                    "content": "What is 2 + 2?",
                }
            ],
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__explicit_auto_same_as_default(
        self, client, model_name, weather_tool
    ):
        """Test explicit tool_choice={'type': 'auto'} works same as default."""
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": "What's the weather in London?"}],
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__uses_only_available_tool_when_tool_choice_any(
        self, client, model_name, weather_tool
    ):
        """Test tool_choice='any' with single tool available."""
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": "Say hello"}],
        )

        tool_uses = [c for c in response.content if c.type == "tool_use"]
        assert len(tool_uses) > 0, "Should use the only available tool"
        assert tool_uses[0].name == "get_weather"
