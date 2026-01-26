"""
Unit tests for LoRA support in embedding models.

Tests the EmbeddingReqInput LoRA fields, normalization logic,
and the integration with the serving layer.
"""

import copy
import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    TokenizedEmbeddingReqInput,
)
from sglang.srt.entrypoints.openai.protocol import EmbeddingRequest
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.server_args import ServerArgs


class TestEmbeddingReqInputLoraFields(unittest.TestCase):
    """Test that EmbeddingReqInput has the required LoRA fields."""

    def test_lora_path_field_exists(self):
        """Test that lora_path field exists and defaults to None."""
        req = EmbeddingReqInput(text="Hello")
        self.assertIsNone(req.lora_path)

    def test_lora_id_field_exists(self):
        """Test that lora_id field exists and defaults to None."""
        req = EmbeddingReqInput(text="Hello")
        self.assertIsNone(req.lora_id)

    def test_lora_path_can_be_set_as_string(self):
        """Test that lora_path can be set as a string."""
        req = EmbeddingReqInput(text="Hello", lora_path="my-adapter")
        self.assertEqual(req.lora_path, "my-adapter")

    def test_lora_path_can_be_set_as_list(self):
        """Test that lora_path can be set as a list."""
        req = EmbeddingReqInput(
            text=["Hello", "World"],
            lora_path=["adapter1", "adapter2"]
        )
        self.assertEqual(req.lora_path, ["adapter1", "adapter2"])

    def test_lora_id_can_be_set(self):
        """Test that lora_id can be set."""
        req = EmbeddingReqInput(text="Hello", lora_id="lora-123")
        self.assertEqual(req.lora_id, "lora-123")


class TestEmbeddingReqInputLoraPathNormalization(unittest.TestCase):
    """Test the normalization of lora_path in EmbeddingReqInput for batch processing."""

    def setUp(self):
        """Set up common test fixtures."""
        self.base_req = EmbeddingReqInput(
            text=["Hello", "World"],
            sampling_params=[{}, {}],
            rid=["id1", "id2"],
        )

    def test_single_lora_path_expanded_to_batch(self):
        """Test that a single lora_path string is expanded to match batch size."""
        req = copy.deepcopy(self.base_req)
        req.lora_path = "my-adapter"

        req.normalize_batch_and_arguments()

        self.assertEqual(req.lora_path, ["my-adapter", "my-adapter"])

    def test_list_lora_path_preserved(self):
        """Test that a list of lora_paths is preserved."""
        req = copy.deepcopy(self.base_req)
        req.lora_path = ["adapter1", "adapter2"]

        req.normalize_batch_and_arguments()

        self.assertEqual(req.lora_path, ["adapter1", "adapter2"])

    def test_lora_path_list_length_mismatch_raises_error(self):
        """Test that mismatched lora_path list length raises ValueError."""
        req = copy.deepcopy(self.base_req)
        req.lora_path = ["adapter1"]  # Length 1, but batch size is 2

        with self.assertRaises(ValueError) as context:
            req.normalize_batch_and_arguments()

        self.assertIn("lora_path list length", str(context.exception))
        self.assertIn("must match batch size", str(context.exception))

    def test_none_lora_path_stays_none(self):
        """Test that None lora_path remains None after normalization."""
        req = copy.deepcopy(self.base_req)
        req.lora_path = None

        req.normalize_batch_and_arguments()

        self.assertIsNone(req.lora_path)

    def test_single_request_lora_path_not_normalized(self):
        """Test that single (non-batch) requests don't normalize lora_path."""
        req = EmbeddingReqInput(text="Hello", lora_path="my-adapter")

        req.normalize_batch_and_arguments()

        # For single requests, lora_path should remain as-is
        self.assertEqual(req.lora_path, "my-adapter")

    def test_lora_path_with_none_values_in_list(self):
        """Test that list with None values is handled correctly."""
        req = copy.deepcopy(self.base_req)
        req.lora_path = [None, "adapter2"]

        req.normalize_batch_and_arguments()

        self.assertEqual(req.lora_path, [None, "adapter2"])


class TestEmbeddingReqInputGetitemWithLora(unittest.TestCase):
    """Test the __getitem__ method includes LoRA fields."""

    def test_getitem_includes_lora_path(self):
        """Test that __getitem__ extracts lora_path correctly."""
        req = EmbeddingReqInput(
            text=["Hello", "World"],
            lora_path=["adapter1", "adapter2"],
        )
        req.normalize_batch_and_arguments()

        item0 = req[0]
        item1 = req[1]

        self.assertEqual(item0.lora_path, "adapter1")
        self.assertEqual(item1.lora_path, "adapter2")

    def test_getitem_includes_lora_id(self):
        """Test that __getitem__ extracts lora_id correctly."""
        req = EmbeddingReqInput(
            text=["Hello", "World"],
            lora_id=["id1", "id2"],
        )
        req.normalize_batch_and_arguments()

        item0 = req[0]
        item1 = req[1]

        self.assertEqual(item0.lora_id, "id1")
        self.assertEqual(item1.lora_id, "id2")

    def test_getitem_with_none_lora_fields(self):
        """Test that __getitem__ handles None lora fields correctly."""
        req = EmbeddingReqInput(
            text=["Hello", "World"],
            lora_path=None,
            lora_id=None,
        )
        req.normalize_batch_and_arguments()

        item0 = req[0]

        self.assertIsNone(item0.lora_path)
        self.assertIsNone(item0.lora_id)

    def test_getitem_cross_encoder_includes_lora_fields(self):
        """Test that __getitem__ for cross-encoder requests includes LoRA fields."""
        req = EmbeddingReqInput(
            text=[["query1", "doc1"], ["query2", "doc2"]],
            lora_path=["adapter1", "adapter2"],
            is_cross_encoder_request=True,
        )
        req.normalize_batch_and_arguments()

        item0 = req[0]
        item1 = req[1]

        self.assertEqual(item0.lora_path, "adapter1")
        self.assertEqual(item1.lora_path, "adapter2")
        self.assertTrue(item0.is_cross_encoder_request)
        self.assertTrue(item1.is_cross_encoder_request)


class TestTokenizedEmbeddingReqInputLoraField(unittest.TestCase):
    """Test that TokenizedEmbeddingReqInput has the lora_id field."""

    def test_lora_id_field_exists(self):
        """Test that lora_id field exists and defaults to None."""
        tokenized = TokenizedEmbeddingReqInput(
            input_text="Hello",
            input_ids=[1, 2, 3],
            image_inputs={},
            token_type_ids=[],
            sampling_params=MagicMock(),
        )
        self.assertIsNone(tokenized.lora_id)

    def test_lora_id_can_be_set(self):
        """Test that lora_id can be set during construction."""
        tokenized = TokenizedEmbeddingReqInput(
            input_text="Hello",
            input_ids=[1, 2, 3],
            image_inputs={},
            token_type_ids=[],
            sampling_params=MagicMock(),
            lora_id="my-lora-id",
        )
        self.assertEqual(tokenized.lora_id, "my-lora-id")


class TestEmbeddingRequestProtocolLoraField(unittest.TestCase):
    """Test that EmbeddingRequest protocol has lora_path field."""

    def test_lora_path_field_exists(self):
        """Test that lora_path field exists in EmbeddingRequest."""
        request = EmbeddingRequest(
            input="Hello world",
            model="test-model",
        )
        self.assertIsNone(request.lora_path)

    def test_lora_path_can_be_set_as_string(self):
        """Test that lora_path can be set as a string."""
        request = EmbeddingRequest(
            input="Hello world",
            model="test-model",
            lora_path="my-adapter",
        )
        self.assertEqual(request.lora_path, "my-adapter")

    def test_lora_path_can_be_set_as_list(self):
        """Test that lora_path can be set as a list."""
        request = EmbeddingRequest(
            input=["Hello", "World"],
            model="test-model",
            lora_path=["adapter1", "adapter2"],
        )
        self.assertEqual(request.lora_path, ["adapter1", "adapter2"])


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


class TestEmbeddingLoraResolution(unittest.TestCase):
    """Test LoRA resolution for embedding requests using the serving base class."""

    def setUp(self):
        self.tokenizer_manager = MockTokenizerManager(enable_lora=True)
        self.serving = ConcreteServingBase(self.tokenizer_manager)

    def test_resolve_lora_from_model_parameter(self):
        """Test LoRA resolution from model:adapter syntax."""
        lora_path = self.serving._resolve_lora_path(
            "embedding-model:my-adapter", None
        )
        self.assertEqual(lora_path, "my-adapter")

    def test_resolve_lora_from_explicit_path(self):
        """Test LoRA resolution from explicit lora_path."""
        lora_path = self.serving._resolve_lora_path(
            "embedding-model", "my-adapter"
        )
        self.assertEqual(lora_path, "my-adapter")

    def test_model_parameter_takes_precedence(self):
        """Test that model:adapter takes precedence over explicit lora_path."""
        lora_path = self.serving._resolve_lora_path(
            "embedding-model:adapter1", "adapter2"
        )
        self.assertEqual(lora_path, "adapter1")

    def test_no_lora_specified(self):
        """Test that None is returned when no LoRA is specified."""
        lora_path = self.serving._resolve_lora_path(
            "embedding-model", None
        )
        self.assertIsNone(lora_path)

    def test_list_lora_paths_preserved(self):
        """Test that list of lora_paths is preserved."""
        lora_path = self.serving._resolve_lora_path(
            "embedding-model", ["adapter1", "adapter2", None]
        )
        self.assertEqual(lora_path, ["adapter1", "adapter2", None])


class TestEmbeddingLoraValidation(unittest.TestCase):
    """Test LoRA validation for embedding requests."""

    def test_validation_passes_when_lora_enabled(self):
        """Test validation passes when LoRA is enabled."""
        tokenizer_manager = MockTokenizerManager(enable_lora=True)
        serving = ConcreteServingBase(tokenizer_manager)

        # Should not raise
        try:
            serving._validate_lora_enabled("my-embedding-adapter")
        except ValueError:
            self.fail("_validate_lora_enabled raised ValueError unexpectedly")

    def test_validation_fails_when_lora_disabled(self):
        """Test validation fails when LoRA is disabled."""
        tokenizer_manager = MockTokenizerManager(enable_lora=False)
        serving = ConcreteServingBase(tokenizer_manager)

        with self.assertRaises(ValueError) as context:
            serving._validate_lora_enabled("my-embedding-adapter")

        error_message = str(context.exception)
        self.assertIn("my-embedding-adapter", error_message)
        self.assertIn("--enable-lora", error_message)


class TestEmbeddingLoraIntegrationScenarios(unittest.TestCase):
    """Integration tests for common embedding LoRA usage scenarios."""

    def setUp(self):
        self.tokenizer_manager = MockTokenizerManager(enable_lora=True)
        self.serving = ConcreteServingBase(self.tokenizer_manager)

    def test_embedding_with_lora_path_parameter(self):
        """Test typical usage with explicit lora_path parameter."""
        # Simulates: {"model": "embedding-model", "input": "Hello", "lora_path": "retrieval"}
        model = "embedding-model"
        explicit_lora = "retrieval"

        lora_path = self.serving._resolve_lora_path(model, explicit_lora)
        self.assertEqual(lora_path, "retrieval")

        # Validation should pass
        self.serving._validate_lora_enabled(lora_path)

    def test_embedding_with_model_adapter_syntax(self):
        """Test usage with model:adapter syntax."""
        # Simulates: {"model": "embedding-model:retrieval", "input": "Hello"}
        model = "embedding-model:retrieval"
        explicit_lora = None

        lora_path = self.serving._resolve_lora_path(model, explicit_lora)
        self.assertEqual(lora_path, "retrieval")

        # Validation should pass
        self.serving._validate_lora_enabled(lora_path)

    def test_embedding_without_lora(self):
        """Test base embedding without LoRA."""
        # Simulates: {"model": "embedding-model", "input": "Hello"}
        model = "embedding-model"
        explicit_lora = None

        lora_path = self.serving._resolve_lora_path(model, explicit_lora)
        self.assertIsNone(lora_path)

    def test_batch_embedding_with_lora(self):
        """Test batch embedding with LoRA."""
        # Simulates batch request with same adapter for all
        model = "embedding-model"
        explicit_lora = "retrieval"

        lora_path = self.serving._resolve_lora_path(model, explicit_lora)
        self.assertEqual(lora_path, "retrieval")

        # Create embedding request with batch
        req = EmbeddingReqInput(
            text=["Hello", "World", "Test"],
            lora_path=lora_path,
        )
        req.normalize_batch_and_arguments()

        # lora_path should be expanded to batch size
        self.assertEqual(req.lora_path, ["retrieval", "retrieval", "retrieval"])

    def test_batch_embedding_with_mixed_lora(self):
        """Test batch embedding with different adapters per item."""
        req = EmbeddingReqInput(
            text=["Hello", "World", "Test"],
            lora_path=["adapter1", None, "adapter2"],
        )
        req.normalize_batch_and_arguments()

        self.assertEqual(req.lora_path, ["adapter1", None, "adapter2"])

        # Verify __getitem__ works correctly
        self.assertEqual(req[0].lora_path, "adapter1")
        self.assertIsNone(req[1].lora_path)
        self.assertEqual(req[2].lora_path, "adapter2")


class TestEmbeddingLoraEdgeCases(unittest.TestCase):
    """Test edge cases for embedding LoRA support."""

    def test_empty_string_lora_path(self):
        """Test handling of empty string lora_path."""
        req = EmbeddingReqInput(text="Hello", lora_path="")
        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, "")

    def test_lora_path_with_special_characters(self):
        """Test lora_path with special characters."""
        req = EmbeddingReqInput(
            text="Hello",
            lora_path="adapter-v2.1_final"
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, "adapter-v2.1_final")

    def test_lora_path_with_path_separators(self):
        """Test lora_path that looks like a file path."""
        req = EmbeddingReqInput(
            text="Hello",
            lora_path="/path/to/my/adapter"
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, "/path/to/my/adapter")

    def test_invalid_lora_path_type_raises_error(self):
        """Test that invalid lora_path type raises error."""
        req = EmbeddingReqInput(
            text=["Hello", "World"],
            lora_path=123,  # Invalid type
        )

        with self.assertRaises(ValueError) as context:
            req.normalize_batch_and_arguments()

        self.assertIn("lora_path should be a list or a string", str(context.exception))


if __name__ == "__main__":
    unittest.main()
