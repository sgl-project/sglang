"""Unit tests for prefill CUDA graph wrapper helpers."""

import unittest
from contextlib import nullcontext
from functools import partial
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

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _LayerModel:
    def __init__(self):
        self.layers = [object()]

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return input_embeds


def _make_pp_buffers_and_registry():
    base = torch.zeros(3, dtype=torch.int64)
    buffers = SimpleNamespace(
        **{name: base.clone() for name in ("input_ids", "positions", "out_cache_loc")},
        pp_proxy_tensors={
            key: torch.zeros((3, 2)) for key in ("hidden_states", "residual")
        },
    )
    registry = build_prefill_registry(
        device=base.device,
        max_bs=1,
        max_num_token=len(base),
        cache_loc_dtype=torch.int64,
        share_pool=False,
        source=buffers,
    )
    return buffers, registry


class TestPrefillCudaGraphRunnerHelpers(CustomTestCase):
    def test_pp_proxy_stable_buffers_accept_full_and_hidden_only_contracts(self):
        buffers, registry = _make_pp_buffers_and_registry()
        full_proxy = PPProxyTensors(
            {
                "hidden_states": torch.full((3, 2), 2.0),
                "residual": torch.full((3, 2), 3.0),
            }
        )
        values = torch.arange(3)
        fill = partial(
            registry.fill_from,
            SimpleNamespace(input_ids=values, positions=values, out_cache_loc=values),
            raw_bs=1,
            padded_bs=1,
            raw_num_tokens=3,
            padded_num_tokens=3,
        )
        fill(pp_proxy_tensors=full_proxy)

        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.buffers = buffers
        runner.model_runner = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=False)
        )
        capture_proxy = runner._capture_pp_proxy_tensors(3)
        torch.testing.assert_close(capture_proxy.tensors, full_proxy.tensors)
        self.assertEqual(
            capture_proxy["hidden_states"].data_ptr(),
            buffers.pp_proxy_tensors["hidden_states"].data_ptr(),
        )

        hidden_only_proxy = PPProxyTensors({"hidden_states": torch.full((3, 2), 4.0)})
        fill(pp_proxy_tensors=hidden_only_proxy)
        torch.testing.assert_close(
            buffers.pp_proxy_tensors["hidden_states"][:3],
            hidden_only_proxy["hidden_states"],
        )

    def test_layer_model_kwargs_bind_optional_inputs_by_signature(self):
        def proxy_before_embeds(a, b, c, pp_proxy_tensors=None, inputs_embeds=None):
            pass

        cases = (
            (_LayerModel(), {"input_embeds": "embeds"}),
            (
                SimpleNamespace(forward=proxy_before_embeds),
                {"inputs_embeds": "embeds", "pp_proxy_tensors": "proxy"},
            ),
        )
        forward_batch = SimpleNamespace(input_embeds="embeds")
        for layer_model, expected in cases:
            with self.subTest(signature=layer_model.forward.__name__):
                kwargs = _build_layer_model_forward_kwargs(
                    layer_model, forward_batch, "proxy"
                )
                self.assertEqual(kwargs, expected)
                layer_model.forward(None, None, forward_batch, **kwargs)

    def test_finalize_pp_proxy_trims_padded_token_rows(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.raw_num_tokens = 3
        output = PPProxyTensors({"hidden_states": torch.arange(10).reshape(5, 2)})
        trimmed = runner._finalize_execute_output(output)
        self.assertIsInstance(trimmed, PPProxyTensors)
        self.assertEqual(tuple(trimmed["hidden_states"].shape), (3, 2))
        torch.testing.assert_close(
            trimmed["hidden_states"][-1], output["hidden_states"][2]
        )

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
            is_first_pp_rank=False,
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

    def test_bcg_eager_tail_uses_live_multimodal_embeddings(self):
        live_embeds = object()
        live_batch = SimpleNamespace(mm_input_embeds=live_embeds)
        static_batch = SimpleNamespace(
            input_ids=None,
            positions=None,
            mm_input_embeds=None,
        )

        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner._input_embeds_arg_idx = None
        runner.buffer_registry = SimpleNamespace(has_slot=lambda _name: False)
        runner.backend = SimpleNamespace(replay=lambda *_args, **_kwargs: None)
        runner.layer_model = SimpleNamespace(forward=lambda *_args, **_kwargs: None)
        runner.model_runner = SimpleNamespace(
            model=SimpleNamespace(
                forward=lambda _ids, _positions, batch, **_kwargs: batch.mm_input_embeds
            )
        )
        runner._prefill_forward_context = lambda *_args, **_kwargs: nullcontext()

        output = runner._execute_body_capture(
            live_batch,
            static_batch,
            static_num_tokens=1,
            raw_num_tokens=1,
            shape_key=object(),
        )

        self.assertIs(output, live_embeds)


if __name__ == "__main__":
    unittest.main()
