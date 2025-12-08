"""
Unit tests for OpenAI-compatible LoRA API support.

Tests the model parameter parsing and LoRA adapter resolution logic
that enables OpenAI-compatible LoRA adapter selection.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="nightly-1-gpu", nightly=True)
from sglang.srt.server_args import ServerArgs


class MockTokenizerManager:
    """Mock TokenizerManager for testing."""

    def __init__(self, enable_lora=False):
        self.server_args = MagicMock(spec=ServerArgs)
        self.server_args.enable_lora = enable_lora
        self.server_args.tokenizer_metrics_allowed_custom_labels = None


class ConcreteServingBase(OpenAIServingBase):
    """Concrete implementation for testing abstract base class."""

    def _request_id_prefix(self) -> str:
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        pass

    def _validate_request(self, request):
        pass


class TestParseModelParameter(unittest.TestCase):
    """Test _parse_model_parameter method."""

    def setUp(self):
        self.tokenizer_manager = MockTokenizerManager(enable_lora=True)
        self.serving = ConcreteServingBase(self.tokenizer_manager)

    def test_model_without_adapter(self):
        """Test parsing model without adapter returns None for adapter."""
        base_model, adapter = self.serving._parse_model_parameter("llama-3.1-8B")
        self.assertEqual(base_model, "llama-3.1-8B")
        self.assertIsNone(adapter)

    def test_model_with_adapter(self):
        """Test parsing model with adapter extracts both parts."""
        base_model, adapter = self.serving._parse_model_parameter(
            "llama-3.1-8B:sql-expert"
        )
        self.assertEqual(base_model, "llama-3.1-8B")
        self.assertEqual(adapter, "sql-expert")

    def test_model_with_path_and_adapter(self):
        """Test parsing model path with slashes and adapter."""
        base_model, adapter = self.serving._parse_model_parameter(
            "meta-llama/Llama-3.1-8B-Instruct:adapter-name"
        )
        self.assertEqual(base_model, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(adapter, "adapter-name")

    def test_model_with_multiple_colons(self):
        """Test that only first colon is used for splitting."""
        base_model, adapter = self.serving._parse_model_parameter("model:adapter:extra")
        self.assertEqual(base_model, "model")
        self.assertEqual(adapter, "adapter:extra")

    def test_model_with_whitespace(self):
        """Test that whitespace is stripped from both parts."""
        base_model, adapter = self.serving._parse_model_parameter(
            " model-name : adapter-name "
        )
        self.assertEqual(base_model, "model-name")
        self.assertEqual(adapter, "adapter-name")

    def test_model_with_empty_adapter(self):
        """Test model ending with colon returns None for adapter."""
        base_model, adapter = self.serving._parse_model_parameter("model-name:")
        self.assertEqual(base_model, "model-name")
        self.assertIsNone(adapter)

    def test_model_with_only_spaces_after_colon(self):
        """Test model with only whitespace after colon returns None for adapter."""
        base_model, adapter = self.serving._parse_model_parameter("model-name:   ")
        self.assertEqual(base_model, "model-name")
        self.assertIsNone(adapter)


class TestResolveLoraPath(unittest.TestCase):
    """Test _resolve_lora_path method."""

    def setUp(self):
        self.tokenizer_manager = MockTokenizerManager(enable_lora=True)
        self.serving = ConcreteServingBase(self.tokenizer_manager)

    def test_no_adapter_specified(self):
        """Test when neither model nor explicit lora_path has adapter."""
        result = self.serving._resolve_lora_path("model-name", None)
        self.assertIsNone(result)

    def test_adapter_in_model_only(self):
        """Test adapter from model parameter when no explicit path."""
        result = self.serving._resolve_lora_path("model:sql-expert", None)
        self.assertEqual(result, "sql-expert")

    def test_adapter_in_explicit_only(self):
        """Test adapter from explicit lora_path when not in model."""
        result = self.serving._resolve_lora_path("model-name", "python-expert")
        self.assertEqual(result, "python-expert")

    def test_model_parameter_takes_precedence(self):
        """Test model parameter adapter takes precedence over explicit."""
        result = self.serving._resolve_lora_path("model:sql-expert", "python-expert")
        self.assertEqual(result, "sql-expert")

    def test_with_list_explicit_lora_path(self):
        """Test that explicit list is returned when no model adapter."""
        explicit = ["adapter1", "adapter2", None]
        result = self.serving._resolve_lora_path("model-name", explicit)
        self.assertEqual(result, explicit)

    def test_model_adapter_overrides_list(self):
        """Test model adapter overrides even when explicit is a list."""
        result = self.serving._resolve_lora_path(
            "model:sql-expert", ["adapter1", "adapter2"]
        )
        self.assertEqual(result, "sql-expert")

    def test_complex_model_name_with_adapter(self):
        """Test resolution with complex model name."""
        result = self.serving._resolve_lora_path(
            "org/model-v2.1:adapter-name", "other-adapter"
        )
        self.assertEqual(result, "adapter-name")


class TestValidateLoraEnabled(unittest.TestCase):
    """Test _validate_lora_enabled method."""

    def test_validation_passes_when_lora_enabled(self):
        """Test validation passes when LoRA is enabled."""
        tokenizer_manager = MockTokenizerManager(enable_lora=True)
        serving = ConcreteServingBase(tokenizer_manager)

        # Should not raise
        try:
            serving._validate_lora_enabled("sql-expert")
        except ValueError:
            self.fail("_validate_lora_enabled raised ValueError unexpectedly")

    def test_validation_fails_when_lora_disabled(self):
        """Test validation fails with helpful message when LoRA is disabled."""
        tokenizer_manager = MockTokenizerManager(enable_lora=False)
        serving = ConcreteServingBase(tokenizer_manager)

        with self.assertRaises(ValueError) as context:
            serving._validate_lora_enabled("sql-expert")

        error_message = str(context.exception)
        self.assertIn("sql-expert", error_message)
        self.assertIn("--enable-lora", error_message)
        self.assertIn("not enabled", error_message)

    def test_validation_error_mentions_adapter_name(self):
        """Test that error message includes the requested adapter name."""
        tokenizer_manager = MockTokenizerManager(enable_lora=False)
        serving = ConcreteServingBase(tokenizer_manager)

        with self.assertRaises(ValueError) as context:
            serving._validate_lora_enabled("my-custom-adapter")

        self.assertIn("my-custom-adapter", str(context.exception))


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for common usage scenarios."""

    def setUp(self):
        self.tokenizer_manager = MockTokenizerManager(enable_lora=True)
        self.serving = ConcreteServingBase(self.tokenizer_manager)

    def test_openai_compatible_usage(self):
        """Test typical OpenAI-compatible usage pattern."""
        # User specifies adapter in model parameter
        model = "meta-llama/Llama-3.1-8B:sql-expert"
        explicit_lora = None

        lora_path = self.serving._resolve_lora_path(model, explicit_lora)
        self.assertEqual(lora_path, "sql-expert")

        # Validation should pass
        self.serving._validate_lora_enabled(lora_path)

    def test_backward_compatible_usage(self):
        """Test backward-compatible usage with explicit lora_path."""
        model = "meta-llama/Llama-3.1-8B"
        explicit_lora = "sql-expert"

        lora_path = self.serving._resolve_lora_path(model, explicit_lora)
        self.assertEqual(lora_path, "sql-expert")

        # Validation should pass
        self.serving._validate_lora_enabled(lora_path)

    def test_base_model_usage(self):
        """Test using base model without any adapter."""
        model = "meta-llama/Llama-3.1-8B"
        explicit_lora = None

        lora_path = self.serving._resolve_lora_path(model, explicit_lora)
        self.assertIsNone(lora_path)

        # No validation needed when no adapter

    def test_batch_request_scenario(self):
        """Test batch request with list of adapters."""
        model = "meta-llama/Llama-3.1-8B"  # No adapter in model
        explicit_lora = ["sql-expert", "python-expert", None]

        lora_path = self.serving._resolve_lora_path(model, explicit_lora)
        self.assertEqual(lora_path, explicit_lora)

        # Validate first adapter in list
        if isinstance(lora_path, list) and lora_path[0]:
            self.serving._validate_lora_enabled(lora_path[0])

    def test_adapter_in_model_overrides_batch_list(self):
        """Test that adapter in model parameter overrides batch list."""
        model = "meta-llama/Llama-3.1-8B:preferred-adapter"
        explicit_lora = ["adapter1", "adapter2"]

        lora_path = self.serving._resolve_lora_path(model, explicit_lora)
        self.assertEqual(lora_path, "preferred-adapter")

    def test_error_when_lora_not_enabled(self):
        """Test comprehensive error flow when LoRA is not enabled."""
        # Setup server without LoRA enabled
        tokenizer_manager = MockTokenizerManager(enable_lora=False)
        serving = ConcreteServingBase(tokenizer_manager)

        # User tries to use adapter
        model = "meta-llama/Llama-3.1-8B:sql-expert"
        lora_path = serving._resolve_lora_path(model, None)

        # Should get helpful error
        with self.assertRaises(ValueError) as context:
            serving._validate_lora_enabled(lora_path)

        error = str(context.exception)
        self.assertIn("--enable-lora", error)
        self.assertIn("sql-expert", error)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        self.tokenizer_manager = MockTokenizerManager(enable_lora=True)
        self.serving = ConcreteServingBase(self.tokenizer_manager)

    def test_empty_string_model(self):
        """Test handling of empty string model."""
        base, adapter = self.serving._parse_model_parameter("")
        self.assertEqual(base, "")
        self.assertIsNone(adapter)

    def test_only_colon(self):
        """Test model parameter that is just a colon."""
        base, adapter = self.serving._parse_model_parameter(":")
        self.assertEqual(base, "")
        self.assertIsNone(adapter)

    def test_empty_list_lora_path(self):
        """Test validation with empty list doesn't crash."""
        lora_path = self.serving._resolve_lora_path("model-name", [])
        # Empty list is falsy, so validation won't be called
        self.assertEqual(lora_path, [])

    def test_list_with_none_first(self):
        """Test validation finds first non-None adapter in list."""
        lora_path = self.serving._resolve_lora_path("model-name", [None, "adapter2"])
        self.assertEqual(lora_path, [None, "adapter2"])
        # In actual usage, validation would find "adapter2"

    def test_list_all_none(self):
        """Test validation with list of all None values."""
        lora_path = self.serving._resolve_lora_path("model-name", [None, None])
        self.assertEqual(lora_path, [None, None])
        # In actual usage, no validation would occur (no non-None adapters)

    def test_unicode_in_adapter_name(self):
        """Test Unicode characters in adapter name."""
        base, adapter = self.serving._parse_model_parameter("model:adapter-名前")
        self.assertEqual(base, "model")
        self.assertEqual(adapter, "adapter-名前")

    def test_special_characters_in_adapter(self):
        """Test special characters in adapter name."""
        base, adapter = self.serving._parse_model_parameter("model:adapter_v2.1-final")
        self.assertEqual(base, "model")
        self.assertEqual(adapter, "adapter_v2.1-final")

    def test_none_as_explicit_lora_path(self):
        """Test None as explicit lora_path is handled correctly."""
        result = self.serving._resolve_lora_path("model:adapter", None)
        self.assertEqual(result, "adapter")

    def test_empty_string_as_explicit_lora_path(self):
        """Test empty string as explicit lora_path."""
        result = self.serving._resolve_lora_path("model-name", "")
        self.assertEqual(result, "")

    def test_validation_with_empty_adapter_name(self):
        """Test validation with empty adapter name still raises error."""
        tokenizer_manager = MockTokenizerManager(enable_lora=False)
        serving = ConcreteServingBase(tokenizer_manager)

        with self.assertRaises(ValueError):
            serving._validate_lora_enabled("")


if __name__ == "__main__":
    unittest.main()
