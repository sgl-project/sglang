"""Unit tests for prefill CUDA graph wrapper helpers."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _resolve_transformer_layer_model,
)
from sglang.srt.model_executor.runner_utils.buffers import PrefillInputBuffers
from sglang.srt.model_loader.utils import resolve_language_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _LayerModel:
    def __init__(self):
        self.layers = [object()]

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return input_embeds


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

    def test_prefill_buffers_allocate_pipeline_proxy_token_rows(self):
        buffers = PrefillInputBuffers.create(
            device=torch.device("cpu"),
            max_bs=4,
            max_num_tokens=16,
            cache_loc_dtype=torch.int64,
            is_multimodal=False,
            hidden_size=8,
            dtype=torch.bfloat16,
            enable_mamba_track=False,
            pp_size=2,
            pp_proxy_residual_num_blocks=3,
        )

        self.assertEqual(
            {
                key: tuple(value.shape)
                for key, value in buffers.pp_proxy_tensors.items()
            },
            {"hidden_states": (16, 8), "residual": (16, 3, 8)},
        )

    def test_pipeline_proxy_output_is_supported(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.raw_num_tokens = 3
        output = PPProxyTensors({"hidden_states": torch.zeros((8, 8))})

        finalized = runner._finalize_execute_output(output)
        self.assertEqual(finalized["hidden_states"].shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
