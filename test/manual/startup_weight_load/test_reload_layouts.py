"""Reload-safe weight layouts and captured scales; no fused TRT-LLM kernel."""

import unittest

import torch
from test_mla_reload import _MLAModel
from torch import nn

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
    ModelStorageManifest,
)
from sglang.srt.runtime_context import get_flags
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestReloadLayouts(CustomTestCase):
    @torch.no_grad()
    def test_mla_non_128_scale_refreshes_captured_storage(self):
        def make_model():
            model = _MLAModel(torch.float8_e4m3fn)
            model.quant_config.weight_block_size = [64, 64]
            model.attention.kv_b_proj.weight_scale = nn.Parameter(
                torch.ones((8, 2), device="cuda"), requires_grad=False
            )
            model.attention.kv_b_proj.weight_scale.format_ue8m0 = False
            model.attention.w_scale = 1.0
            return model

        def forward(model, query, latent):
            attn = model.attention
            return (
                torch.bmm(query, attn.w_kc.float()) * attn.w_scale,
                torch.bmm(latent, attn.w_vc.float()) * attn.w_scale,
            )

        candidate = make_model()
        candidate.post_load_weights()
        manifest = ModelStorageManifest.capture(candidate)
        old_scale = candidate.attention.w_scale.clone()
        query = torch.randn((2, 3, 128), device="cuda")
        latent = torch.randn_like(query)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward(candidate, query, latent)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward(candidate, query, latent)

        serial = make_model()
        weight = torch.randn((512, 128), device="cuda").to(torch.float8_e4m3fn)
        scales = torch.arange(1, 17, device="cuda").reshape(8, 2).float()
        for model in (candidate, serial):
            model.attention.kv_b_proj.weight.copy_(weight)
            model.attention.kv_b_proj.weight_scale.copy_(scales)
            model.post_load_weights()

        self.assertEqual(manifest.changed_names(candidate), ())
        self.assertFalse(torch.equal(candidate.attention.w_scale, old_scale))
        graph.replay()
        for actual, expected in zip(captured, forward(serial, query, latent)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.no_grad()
    def test_trtllm_reload_matches_fresh_postprocess(self):
        def make_layer():
            layer = nn.Module()
            layer.num_local_experts = 2
            layer.hidden_size = 256
            layer.intermediate_size_per_partition = 256
            layer.moe_runner_config = MoeRunnerConfig(is_gated=True)
            for name, shape in (
                ("w13_weight", (2, 512, 256)),
                ("w2_weight", (2, 256, 256)),
            ):
                layer.register_parameter(
                    name,
                    nn.Parameter(
                        torch.zeros(shape, device="cuda", dtype=torch.bfloat16),
                        requires_grad=False,
                    ),
                )
            return layer

        for backend in (
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
        ):
            with (
                self.subTest(backend=backend),
                get_flags().moe.override(runner_backend=backend),
            ):
                method = UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True)
                candidate = make_layer()
                serial = make_layer()
                method.process_weights_after_loading(candidate)
                manifest = ModelStorageManifest.capture(candidate)
                for name, param in candidate.named_parameters():
                    method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
                        candidate, param, f"model.layers.0.mlp.experts.{name}"
                    )
                    weight = torch.randn_like(getattr(serial, name))
                    param.copy_(weight)
                    getattr(serial, name).copy_(weight)
                method.process_weights_after_loading(candidate)
                method.process_weights_after_loading(serial)
                self.assertEqual(manifest.changed_names(candidate), ())
                for name, param in candidate.named_parameters():
                    torch.testing.assert_close(
                        param, getattr(serial, name), rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
