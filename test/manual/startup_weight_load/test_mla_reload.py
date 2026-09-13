"""CUDA graph replay after refreshing MLA's weight-derived tensors."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
    ModelStorageManifest,
)
from sglang.srt.model_loader.weight_utils import CAPTURE_SAFE_WEIGHT_SENTINEL
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.test.test_utils import CustomTestCase


class _MLAAttention(nn.Module):
    named_startup_weight_load_derived_tensors = (
        DeepseekV2AttentionMLA.named_startup_weight_load_derived_tensors
    )

    def __init__(self, dtype):
        super().__init__()
        self.qk_nope_head_dim = 128
        self.v_head_dim = 128
        self.kv_b_proj = nn.Module()
        self.kv_b_proj.weight = nn.Parameter(
            torch.full(
                (512, 128),
                CAPTURE_SAFE_WEIGHT_SENTINEL,
                dtype=dtype,
                device="cuda",
            ),
            requires_grad=False,
        )
        if dtype == torch.float8_e4m3fn:
            self.kv_b_proj.weight_scale = nn.Parameter(
                torch.ones((4, 1), device="cuda"), requires_grad=False
            )
            self.kv_b_proj.weight_scale.format_ue8m0 = False
        self.w_kc = self.w_vc = None
        self.w_scale = self.w_scale_k = self.w_scale_v = None
        self.use_deep_gemm_bmm = False


class _MLAModel(DeepseekV2WeightLoaderMixin, nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.quant_config = (
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
            )
            if dtype == torch.float8_e4m3fn
            else None
        )
        layer = nn.Module()
        layer.self_attn = _MLAAttention(dtype)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])
        self.model.start_layer = 0
        self.model.end_layer = 1

    @property
    def attention(self):
        return self.model.layers[0].self_attn

    def forward(self, query, latent):
        return (
            torch.bmm(query, self.attention.w_kc),
            torch.bmm(latent, self.attention.w_vc),
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestStartupWeightLoadMLA(CustomTestCase):
    @torch.no_grad()
    def test_refresh_preserves_captured_mla_weights(self):
        for dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float8_e4m3fn,
        ):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)
                candidate = _MLAModel(dtype)
                candidate.post_load_weights()
                self.assertFalse(candidate.attention.use_deep_gemm_bmm)
                manifest = ModelStorageManifest.capture(candidate)
                compute_dtype = candidate.attention.w_kc.dtype
                query = torch.randn((2, 3, 128), device="cuda", dtype=compute_dtype)
                latent = torch.randn_like(query)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        candidate(query, latent)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = candidate(query, latent)
                graph.replay()
                dummy = tuple(output.clone() for output in captured)

                weight = torch.randn((512, 128), device="cuda").to(dtype)
                candidate.attention.kv_b_proj.weight.copy_(weight)
                serial = _MLAModel(dtype)
                serial.attention.kv_b_proj.weight.copy_(weight)
                if dtype == torch.float8_e4m3fn:
                    scales = torch.tensor([[0.5], [1.0], [1.5], [2.0]], device="cuda")
                    candidate.attention.kv_b_proj.weight_scale.copy_(scales)
                    serial.attention.kv_b_proj.weight_scale.copy_(scales)
                    weight = (weight.float() * scales.repeat_interleave(128, dim=0)).to(
                        compute_dtype
                    )

                candidate.post_load_weights()
                serial.post_load_weights()
                self.assertEqual(manifest.changed_names(candidate), ())
                expected_k, expected_v = weight.reshape(2, 256, 128).split(128, dim=1)
                torch.testing.assert_close(candidate.attention.w_kc, expected_k)
                torch.testing.assert_close(
                    candidate.attention.w_vc, expected_v.transpose(1, 2)
                )

                graph.replay()
                reference = serial(query, latent)
                for actual, expected, placeholder in zip(captured, reference, dummy):
                    self.assertTrue(torch.isfinite(actual).all().item())
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertFalse(torch.equal(actual, placeholder))


if __name__ == "__main__":
    unittest.main()
