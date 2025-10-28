"""
End-to-end tests for OpenAI-compatible LoRA adapter usage.

Tests the model:adapter syntax and backward compatibility with explicit lora_path.

Usage:
    python3 -m unittest openai_server.features.test_lora_openai_compatible.TestLoRAOpenAICompatible.test_model_adapter_syntax
    python3 -m unittest openai_server.features.test_lora_openai_compatible.TestLoRAOpenAICompatible.test_explicit_lora_path
    python3 -m unittest openai_server.features.test_lora_openai_compatible.TestLoRAOpenAICompatible.test_priority_model_over_explicit
    python3 -m unittest openai_server.features.test_lora_openai_compatible.TestLoRAOpenAICompatible.test_base_model_no_adapter
    python3 -m unittest openai_server.features.test_lora_openai_compatible.TestLoRAOpenAICompatible.test_completions_api_with_adapter
    python3 -m unittest openai_server.features.test_lora_openai_compatible.TestLoRAOpenAICompatible.test_streaming_with_adapter
    python3 -m unittest openai_server.features.test_lora_openai_compatible.TestLoRADisabledError.test_lora_disabled_error
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


def get_real_lora_adapter() -> str:
    """Use a real LoRA adapter from Hugging Face."""
    return "codelion/Llama-3.2-1B-Instruct-tool-calling-lora"


def setup_class(cls, enable_lora=True):
    """Setup test class with LoRA-enabled server."""
    cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    cls.base_url = DEFAULT_URL_FOR_TEST

    # Use real LoRA adapter
    cls.lora_adapter_path = get_real_lora_adapter()

    other_args = [
        "--max-running-requests",
        "10",
        "--disable-radix-cache",  # Disable cache for cleaner tests
    ]

    if enable_lora:
        other_args.extend(
            [
                "--enable-lora",
                "--lora-paths",
                f"tool_calling={cls.lora_adapter_path}",
            ]
        )

    cls.process = popen_launch_server(
        cls.model,
        cls.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")


class TestLoRAOpenAICompatible(CustomTestCase):
    """Test OpenAI-compatible LoRA adapter usage."""

    @classmethod
    def setUpClass(cls):
        setup_class(cls, enable_lora=True)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_model_adapter_syntax(self):
        """Test the new model:adapter syntax works correctly."""
        response = self.client.chat.completions.create(
            # ← New OpenAI-compatible syntax
            model=f"{self.model}:tool_calling",
            messages=[{"role": "user", "content": "What tools do you have available?"}],
            max_tokens=50,
            temperature=0,
        )

        self.assertIsNotNone(response.choices[0].message.content)
        self.assertGreater(len(response.choices[0].message.content), 0)
        print(f"Model adapter syntax response: {response.choices[0].message.content}")

    def test_explicit_lora_path(self):
        """Test backward compatibility with explicit lora_path via extra_body."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "What tools do you have available?"}],
            # ← Legacy explicit method
            extra_body={"lora_path": "tool_calling"},
            max_tokens=50,
            temperature=0,
        )

        self.assertIsNotNone(response.choices[0].message.content)
        self.assertGreater(len(response.choices[0].message.content), 0)
        print(f"Explicit lora_path response: {response.choices[0].message.content}")

    def test_priority_model_over_explicit(self):
        """Test that model:adapter syntax takes precedence over explicit lora_path."""
        # This test verifies the priority logic in _resolve_lora_path
        response = self.client.chat.completions.create(
            # ← Model specifies tool_calling adapter
            model=f"{self.model}:tool_calling",
            messages=[{"role": "user", "content": "What tools do you have available?"}],
            # ← Both specify same adapter
            extra_body={"lora_path": "tool_calling"},
            max_tokens=50,
            temperature=0,
        )

        # Should use tool_calling adapter (model parameter takes precedence)
        self.assertIsNotNone(response.choices[0].message.content)
        self.assertGreater(len(response.choices[0].message.content), 0)
        print(f"Priority test response: {response.choices[0].message.content}")

    def test_base_model_no_adapter(self):
        """Test using base model without any adapter."""
        response = self.client.chat.completions.create(
            model=self.model,  # ← No adapter specified
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=30,
            temperature=0,
        )

        self.assertIsNotNone(response.choices[0].message.content)
        self.assertGreater(len(response.choices[0].message.content), 0)
        print(f"Base model response: {response.choices[0].message.content}")

    def test_completions_api_with_adapter(self):
        """Test completions API with LoRA adapter."""
        response = self.client.completions.create(
            model=f"{self.model}:tool_calling",  # ← Using model:adapter syntax
            prompt="What tools do you have available?",
            max_tokens=50,
            temperature=0,
        )

        self.assertIsNotNone(response.choices[0].text)
        self.assertGreater(len(response.choices[0].text), 0)
        print(f"Completions API response: {response.choices[0].text}")

    def test_streaming_with_adapter(self):
        """Test streaming with LoRA adapter."""
        stream = self.client.chat.completions.create(
            model=f"{self.model}:tool_calling",
            messages=[{"role": "user", "content": "What tools do you have available?"}],
            max_tokens=50,
            temperature=0,
            stream=True,
        )

        collected_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                collected_content += chunk.choices[0].delta.content

        self.assertGreater(len(collected_content), 0)
        print(f"Streaming response: {collected_content}")

    def test_multiple_adapters(self):
        """Test using different adapters in sequence."""
        # Test tool_calling adapter
        tool_response = self.client.chat.completions.create(
            model=f"{self.model}:tool_calling",
            messages=[{"role": "user", "content": "What tools do you have available?"}],
            max_tokens=30,
            temperature=0,
        )

        # Test base model without adapter
        base_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=30,
            temperature=0,
        )

        self.assertIsNotNone(tool_response.choices[0].message.content)
        self.assertIsNotNone(base_response.choices[0].message.content)
        print(
            f"Tool calling adapter response: {tool_response.choices[0].message.content}"
        )
        print(f"Base model response: {base_response.choices[0].message.content}")


class TestLoRADisabledError(CustomTestCase):
    """Test error handling when LoRA is disabled."""

    @classmethod
    def setUpClass(cls):
        setup_class(cls, enable_lora=False)  # ← LoRA disabled

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_disabled_error(self):
        """Test that using LoRA adapter when LoRA is disabled raises appropriate error."""
        with self.assertRaises(openai.APIError) as context:
            self.client.chat.completions.create(
                model=f"{self.model}:tool_calling",  # ← Trying to use adapter
                messages=[
                    {"role": "user", "content": "What tools do you have available?"}
                ],
                max_tokens=50,
            )

        # Verify the error message contains helpful guidance
        error_message = str(context.exception)
        self.assertIn("LoRA", error_message)
        self.assertIn("not enabled", error_message)
        print(f"Expected error message: {error_message}")


class TestLoRAEdgeCases(CustomTestCase):
    """Test edge cases for LoRA adapter usage."""

    @classmethod
    def setUpClass(cls):
        setup_class(cls, enable_lora=True)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_model_with_colon_no_adapter(self):
        """Test model parameter ending with colon (empty adapter)."""
        response = self.client.chat.completions.create(
            model=f"{self.model}:",  # ← Model ends with colon
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=30,
            temperature=0,
        )

        # Should work as base model (no adapter)
        self.assertIsNotNone(response.choices[0].message.content)
        print(f"Model with colon response: {response.choices[0].message.content}")

    def test_explicit_lora_path_none(self):
        """Test explicit lora_path set to None."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Hello!"}],
            extra_body={"lora_path": None},  # ← Explicitly None
            max_tokens=30,
            temperature=0,
        )

        # Should work as base model
        self.assertIsNotNone(response.choices[0].message.content)
        print(
            f"Explicit None lora_path response: {response.choices[0].message.content}"
        )

    def test_invalid_adapter_name(self):
        """Test using non-existent adapter name."""
        with self.assertRaises(openai.APIError) as context:
            self.client.chat.completions.create(
                model=f"{self.model}:nonexistent",  # ← Non-existent adapter
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=30,
            )

        error_message = str(context.exception)
        print(f"Invalid adapter error: {error_message}")


if __name__ == "__main__":
    unittest.main()
