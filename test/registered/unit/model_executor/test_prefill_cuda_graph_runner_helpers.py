"""Unit tests for prefill CUDA graph wrapper helpers."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _resolve_transformer_layer_model,
)
from sglang.srt.model_loader.utils import resolve_language_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _LayerModel:
    def __init__(self):
        self.layers = [object()]

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return input_embeds


class TestPrefillCudaGraphRunnerHelpers(CustomTestCase):
    def test_input_embeds_hidden_size_uses_explicit_value(self):
        """An explicit buffer width must take precedence over the default."""
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.model_runner = SimpleNamespace(
            model=SimpleNamespace(input_embeds_hidden_size=7),
            model_config=SimpleNamespace(hidden_size=5),
        )

        self.assertEqual(runner._input_embeds_hidden_size(), 7)

    def test_input_embeds_hidden_size_uses_default_value(self):
        """The configured hidden size remains the default buffer width."""
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.model_runner = SimpleNamespace(
            model=SimpleNamespace(),
            model_config=SimpleNamespace(hidden_size=5),
        )

        self.assertEqual(runner._input_embeds_hidden_size(), 5)

    def test_resolve_layer_model_from_language_model_wrapper(self):
        layer_model = _LayerModel()
        model = SimpleNamespace(language_model=SimpleNamespace(model=layer_model))

        self.assertIs(_resolve_transformer_layer_model(model), layer_model)

    def test_resolve_layer_model_from_nested_model_wrapper(self):
        layer_model = _LayerModel()
        model = SimpleNamespace(model=SimpleNamespace(model=layer_model))

        self.assertIs(_resolve_transformer_layer_model(model), layer_model)

    def test_resolve_layer_model_rejects_wrapper_without_layers(self):
        model = SimpleNamespace()
        model.model = model

        with self.assertRaisesRegex(RuntimeError, "without layers"):
            _resolve_transformer_layer_model(model)

    def test_resolve_language_model_accepts_asr_style_wrapper(self):
        language_model = object()
        self.assertIs(
            resolve_language_model(SimpleNamespace(language_model=language_model)),
            language_model,
        )

    def test_resolve_language_model_accepts_omni_style_wrapper(self):
        language_model = object()
        omni_model = type("Qwen3OmniMoeForConditionalGeneration", (), {})()
        omni_model.thinker = SimpleNamespace(model=language_model)
        self.assertIs(resolve_language_model(omni_model), language_model)

    def test_resolve_language_model_rejects_non_language_wrapper(self):
        with self.assertRaises(AttributeError):
            resolve_language_model(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
