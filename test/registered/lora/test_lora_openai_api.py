"""
Unit tests for OpenAI-compatible LoRA adapter selection.

``_parse_model_parameter`` and ``_resolve_lora_path`` in ``serving_base.py``
decide which adapter a ``model="base:adapter"`` request runs on; a wrong split
silently serves the base model or another adapter.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="nightly-amd-1-gpu", nightly=True)
register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="stage-b-test-cpu-intel")

SERVED_MODEL_NAME = "test-model"
# An exact served name that itself contains a colon.
URI_MODEL_NAME = "s3://bucket/model"


class MockTokenizerManager:
    """The manager attributes the resolver reads."""

    def __init__(self, enable_lora=False):
        self.server_args = MagicMock(spec=ServerArgs)
        self.server_args.enable_lora = enable_lora
        self.served_model_name = SERVED_MODEL_NAME


class ConcreteServingBase(OpenAIServingBase):
    """Concrete implementation for testing abstract base class."""

    def _request_id_prefix(self) -> str:
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        pass

    def _validate_request(self, request):
        pass


class TestLoRAModelParameter(CustomTestCase):
    def setUp(self):
        # `OpenAIServingBase.__init__` reads `get_observability()`, so a case
        # needs a published config: `MagicMock(spec=ServerArgs)` passes the
        # `isinstance` guard, and there is no bag to answer from otherwise.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.tokenizer_manager = MockTokenizerManager(enable_lora=True)
        self.serving = ConcreteServingBase(self.tokenizer_manager)

    def test_model_without_adapter(self):
        self.assertEqual(
            self.serving._parse_model_parameter("llama-3.1-8B"), ("llama-3.1-8B", None)
        )
        self.assertIsNone(self.serving._resolve_lora_path("llama-3.1-8B", None))

    def test_explicit_lora_path_string(self):
        self.assertEqual(
            self.serving._resolve_lora_path("llama-3.1-8B", "python-expert"),
            "python-expert",
        )

    def test_explicit_lora_path_list(self):
        explicit = ["adapter1", None, "adapter2"]
        self.assertIs(
            self.serving._resolve_lora_path("llama-3.1-8B", explicit), explicit
        )

    def test_model_suffix_overrides_explicit_string(self):
        self.assertEqual(
            self.serving._resolve_lora_path("model:sql-expert", "python-expert"),
            "sql-expert",
        )

    def test_model_suffix_overrides_explicit_list(self):
        self.assertEqual(
            self.serving._resolve_lora_path(
                "model:sql-expert", ["adapter1", "adapter2"]
            ),
            "sql-expert",
        )

    def test_first_colon_splits_base_from_adapter(self):
        self.assertEqual(self.serving._parse_model_parameter("a:b:c"), ("a", "b:c"))
        self.assertEqual(
            self.serving._parse_model_parameter(
                "meta-llama/Llama-3.1-8B-Instruct:adapter-name"
            ),
            ("meta-llama/Llama-3.1-8B-Instruct", "adapter-name"),
        )

    def test_whitespace_stripped_and_empty_suffix_is_none(self):
        self.assertEqual(
            self.serving._parse_model_parameter(" model-name : adapter-name "),
            ("model-name", "adapter-name"),
        )
        for model in ("model-name:", "model-name:   "):
            with self.subTest(model=model):
                self.assertEqual(
                    self.serving._parse_model_parameter(model), ("model-name", None)
                )

    def test_served_model_name_with_colon_is_the_base_model(self):
        self.tokenizer_manager.served_model_name = URI_MODEL_NAME
        self.assertEqual(
            self.serving._parse_model_parameter(URI_MODEL_NAME), (URI_MODEL_NAME, None)
        )
        self.assertIsNone(self.serving._resolve_lora_path(URI_MODEL_NAME, None))

    def test_served_model_name_with_colon_keeps_adapter_suffix(self):
        self.tokenizer_manager.served_model_name = URI_MODEL_NAME
        self.assertEqual(
            self.serving._parse_model_parameter(f"{URI_MODEL_NAME}:adapter"),
            (URI_MODEL_NAME, "adapter"),
        )
        self.assertEqual(
            self.serving._resolve_lora_path(f"{URI_MODEL_NAME}:adapter", "other"),
            "adapter",
        )


if __name__ == "__main__":
    unittest.main()
