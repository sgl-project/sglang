"""Unit tests for OpenAIServingBase — no server, no model loading."""

from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.test.test_utils import CustomTestCase


class _MockTokenizerManager:
    def __init__(self):
        self.model_config = Mock()
        self.server_args = Mock()
        self.server_args.enable_cache_report = False
        self.server_args.tokenizer_metrics_allowed_custom_labels = None
        self.model_path = "test-model"
        self.tokenizer = Mock()
        self.tokenizer.chat_template = None


class _ConcreteServingBase(OpenAIServingBase):
    """Minimal concrete subclass so we can test base-class methods."""

    _request_id_prefix = "test"

    def _convert_to_internal_request(self, request):
        pass


def _make_serving():
    return _ConcreteServingBase(_MockTokenizerManager())


class TestParseModelParameter(CustomTestCase):

    def setUp(self):
        self.serving = _make_serving()

    def test_no_colon(self):
        base, adapter = self.serving._parse_model_parameter("my-model")
        self.assertEqual(base, "my-model")
        self.assertIsNone(adapter)

    def test_single_colon(self):
        base, adapter = self.serving._parse_model_parameter("base-model:adapter-v1")
        self.assertEqual(base, "base-model")
        self.assertEqual(adapter, "adapter-v1")

    def test_multiple_colons_splits_on_first(self):
        """Model paths like 'org:model:adapter' should split on the first colon only."""
        base, adapter = self.serving._parse_model_parameter("org:model:adapter")
        self.assertEqual(base, "org")
        self.assertEqual(adapter, "model:adapter")

    def test_empty_adapter_becomes_none(self):
        base, adapter = self.serving._parse_model_parameter("base-model:")
        self.assertEqual(base, "base-model")
        self.assertIsNone(adapter)

    def test_whitespace_trimmed(self):
        base, adapter = self.serving._parse_model_parameter("  base : adapter  ")
        self.assertEqual(base, "base")
        self.assertEqual(adapter, "adapter")

    def test_whitespace_only_adapter_becomes_none(self):
        base, adapter = self.serving._parse_model_parameter("base:   ")
        self.assertEqual(base, "base")
        self.assertIsNone(adapter)


class TestResolveLoraPath(CustomTestCase):

    def setUp(self):
        self.serving = _make_serving()

    def test_adapter_from_model_takes_precedence(self):
        result = self.serving._resolve_lora_path("base:adapter-A", "adapter-B")
        self.assertEqual(result, "adapter-A")

    def test_fallback_to_explicit_lora_path(self):
        result = self.serving._resolve_lora_path("base-model", "explicit-adapter")
        self.assertEqual(result, "explicit-adapter")

    def test_no_adapter_anywhere(self):
        result = self.serving._resolve_lora_path("base-model", None)
        self.assertIsNone(result)

    def test_list_lora_path_passthrough(self):
        """Batch requests pass a list of lora_paths; should be returned as-is."""
        result = self.serving._resolve_lora_path("base-model", ["a", "b"])
        self.assertEqual(result, ["a", "b"])


class TestCreateErrorResponse(CustomTestCase):

    def setUp(self):
        self.serving = _make_serving()

    def test_default_status_code(self):
        resp = self.serving.create_error_response("Something went wrong")
        self.assertEqual(resp.status_code, 400)

    def test_custom_status_code(self):
        resp = self.serving.create_error_response(
            "Not found", err_type="NotFoundError", status_code=404
        )
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
