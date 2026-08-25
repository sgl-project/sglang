"""Unit tests for prefill CUDA graph wrapper helpers."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_prefill_registry,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _build_layer_model_forward_kwargs,
    _resolve_transformer_layer_model,
)
from sglang.srt.model_executor.runner_utils.buffers import PrefillInputBuffers
from sglang.srt.model_loader.utils import resolve_language_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _LayerModel:
    def __init__(self):
        self.layers = [object()]

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return input_embeds


class _ProxyBeforeEmbedsLayerModel:
    def __init__(self):
        self.layers = [object()]

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        pp_proxy_tensors=None,
        inputs_embeds=None,
    ):
        return pp_proxy_tensors, inputs_embeds


def _make_pp_buffers_and_registry():
    buffers = PrefillInputBuffers.create(
        device=torch.device("cpu"),
        max_bs=2,
        max_num_tokens=8,
        cache_loc_dtype=torch.int64,
        is_multimodal=False,
        hidden_size=4,
        dtype=torch.float32,
        enable_mamba_track=False,
        pp_size=4,
    )
    registry = build_prefill_registry(
        device=torch.device("cpu"),
        max_bs=2,
        max_num_token=8,
        cache_loc_dtype=torch.int64,
        share_pool=False,
        source=buffers,
    )
    return buffers, registry


class TestPrefillCudaGraphRunnerHelpers(CustomTestCase):
    def test_pp_proxy_uses_stable_capture_buffer_and_live_replay_values(self):
        buffers, registry = _make_pp_buffers_and_registry()
        live_proxy = PPProxyTensors(
            {
                "hidden_states": torch.full((3, 4), 2.0),
                "residual": torch.full((3, 4), 3.0),
            }
        )
        forward_batch = SimpleNamespace(
            input_ids=torch.arange(3),
            positions=torch.arange(3),
            out_cache_loc=torch.arange(3),
        )

        registry.fill_from(
            forward_batch,
            raw_bs=1,
            padded_bs=1,
            raw_num_tokens=3,
            padded_num_tokens=3,
            pp_proxy_tensors=live_proxy,
        )

        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.buffers = buffers
        runner.model_runner = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=False)
        )
        capture_proxy = runner._capture_pp_proxy_tensors(3)
        self.assertEqual(capture_proxy["hidden_states"].tolist(), [[2.0] * 4] * 3)
        self.assertEqual(capture_proxy["residual"].tolist(), [[3.0] * 4] * 3)
        self.assertEqual(
            capture_proxy["hidden_states"].data_ptr(),
            buffers.pp_proxy_tensors["hidden_states"].data_ptr(),
        )

    def test_pp_proxy_allows_hidden_only_live_contract(self):
        buffers, registry = _make_pp_buffers_and_registry()
        live_proxy = PPProxyTensors({"hidden_states": torch.full((3, 4), 2.0)})
        forward_batch = SimpleNamespace(
            input_ids=torch.arange(3),
            positions=torch.arange(3),
            out_cache_loc=torch.arange(3),
        )

        registry.fill_from(
            forward_batch,
            raw_bs=1,
            padded_bs=1,
            raw_num_tokens=3,
            padded_num_tokens=3,
            pp_proxy_tensors=live_proxy,
        )

        self.assertEqual(
            buffers.pp_proxy_tensors["hidden_states"][:3].tolist(),
            [[2.0] * 4] * 3,
        )
        self.assertEqual(
            buffers.pp_proxy_tensors["residual"][:3].tolist(),
            [[0.0] * 4] * 3,
        )

    def test_layer_model_kwargs_bind_proxy_before_embeds_by_name(self):
        layer_model = _ProxyBeforeEmbedsLayerModel()
        input_embeds = object()
        pp_proxy_tensors = object()
        forward_batch = SimpleNamespace(input_embeds=input_embeds)

        kwargs = _build_layer_model_forward_kwargs(
            layer_model, forward_batch, pp_proxy_tensors
        )
        actual_proxy, actual_embeds = layer_model.forward(
            None, None, forward_batch, **kwargs
        )

        self.assertIs(actual_proxy, pp_proxy_tensors)
        self.assertIs(actual_embeds, input_embeds)

    def test_layer_model_kwargs_keep_input_embeds_alias(self):
        layer_model = _LayerModel()
        input_embeds = object()
        forward_batch = SimpleNamespace(input_embeds=input_embeds)

        kwargs = _build_layer_model_forward_kwargs(layer_model, forward_batch, None)

        self.assertIs(
            layer_model.forward(None, None, forward_batch, **kwargs), input_embeds
        )

    def test_finalize_pp_proxy_trims_padded_token_rows(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.raw_num_tokens = 3
        output = PPProxyTensors(
            {
                "hidden_states": torch.arange(20).reshape(5, 4),
                "residual": torch.arange(20, 40).reshape(5, 4),
            }
        )

        trimmed = runner._finalize_execute_output(output)

        self.assertIsInstance(trimmed, PPProxyTensors)
        self.assertEqual(tuple(trimmed["hidden_states"].shape), (3, 4))
        self.assertEqual(tuple(trimmed["residual"].shape), (3, 4))
        self.assertEqual(
            trimmed["hidden_states"].tolist(),
            output["hidden_states"][:3].tolist(),
        )
        self.assertEqual(trimmed["residual"].tolist(), output["residual"][:3].tolist())

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
