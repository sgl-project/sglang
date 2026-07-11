"""Unit tests for prefill CUDA graph runner helpers."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    _prefill_input_embeds_slot,
    _resolve_transformer_layer_model,
)
from sglang.srt.model_executor.model_runner import _has_resolvable_language_model
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class _LayerModel:
    def __init__(self):
        self.layers = [object()]

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return input_embeds


class _FakeSlot:
    def __init__(self, tensor):
        self.tensor = tensor
        self.calls = []

    def slice_for(self, bs, num_tokens):
        self.calls.append((bs, num_tokens))
        return self.tensor[:num_tokens]


class _FakeRegistry:
    def __init__(self, slot=None):
        self.slot = slot

    def has_slot(self, name):
        return name == "input_embeds" and self.slot is not None

    def get_slot(self, name):
        assert name == "input_embeds"
        return self.slot


class TestPrefillCudaGraphRunnerHelpers(CustomTestCase):
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

    def test_language_model_gate_accepts_asr_style_wrapper(self):
        self.assertTrue(
            _has_resolvable_language_model(SimpleNamespace(language_model=object()))
        )

    def test_language_model_gate_accepts_omni_style_wrapper(self):
        omni_model = type("Qwen3OmniMoeForConditionalGeneration", (), {})()
        self.assertTrue(_has_resolvable_language_model(omni_model))

    def test_language_model_gate_rejects_non_language_wrapper(self):
        self.assertFalse(_has_resolvable_language_model(SimpleNamespace()))

    def test_prefill_input_embeds_slot_returns_stable_slot_slice(self):
        slot = _FakeSlot(torch.arange(12).view(3, 4))
        registry = _FakeRegistry(slot)

        embeds = _prefill_input_embeds_slot(registry, bs=2, num_tokens=2)

        self.assertTrue(torch.equal(embeds, slot.tensor[:2]))
        self.assertEqual(slot.calls, [(2, 2)])

    def test_prefill_input_embeds_slot_returns_none_when_absent(self):
        self.assertIsNone(
            _prefill_input_embeds_slot(_FakeRegistry(), bs=1, num_tokens=4)
        )


if __name__ == "__main__":
    unittest.main()
