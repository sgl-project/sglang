"""Speculative captures must retain the eager decoder's complete layer outputs."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.layernorm import Gemma3RMSNorm, RMSNorm
from sglang.srt.models.gemma3_causal import Gemma3DecoderLayer, Gemma3TextModel
from sglang.srt.models.gemma4_causal import Gemma4DecoderLayer, Gemma4TextModel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class ToyAttention(nn.Module):
    is_sliding = False

    def forward(self, hidden_states, **kwargs):
        return hidden_states.roll(1, dims=-1) * 0.25


def make_model(version, device, dtype):
    """Use the real decoder and norm paths, with small deterministic sublayers."""
    model_cls, layer_cls, norm_cls = (
        (Gemma3TextModel, Gemma3DecoderLayer, Gemma3RMSNorm)
        if version == 3
        else (Gemma4TextModel, Gemma4DecoderLayer, RMSNorm)
    )

    def norm():
        module = norm_cls(128, eps=1e-6).to(device=device, dtype=dtype)
        offset = 0.0 if version == 3 else 1.0
        module.weight.data.copy_(
            torch.linspace(-0.2, 0.2, 128, device=device, dtype=dtype) + offset
        )
        if device == "cpu":
            module._forward_method = module.forward_native
        return module

    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    layers = []
    for _ in range(3):
        layer = layer_cls.__new__(layer_cls)
        nn.Module.__init__(layer)
        layer.self_attn = ToyAttention()
        layer.mlp = nn.Tanh()
        layer.input_layernorm = norm()
        layer.post_attention_layernorm = norm()
        layer.pre_feedforward_layernorm = norm()
        layer.post_feedforward_layernorm = norm()
        if version == 4:
            layer.enable_moe_block = False
            layer.has_ple = False
            layer.moe = None
            layer.layer_scalar = torch.tensor([0.75], device=device, dtype=dtype)
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    model.norm = norm()
    model.layers_to_capture = [0, 1, 3]
    if version == 3:
        model.rotary_emb = lambda *args: None
        model.rotary_emb_local = lambda *args: None
    else:
        model.config = SimpleNamespace(num_hidden_layers=3)
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.start_layer, model.end_layer = 0, 3
        model.per_layer_model_projection = None
    return model


def eager_reference(model, inputs, version):
    """Unfused residual additions, without a separately carried residual stream."""

    def norm(x, module):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        weight = module.weight.float() + (1.0 if version == 3 else 0.0)
        return (value * weight).to(x.dtype)

    hidden = inputs.clone()
    outputs = [hidden]
    for layer in model.layers:
        attn = layer.self_attn(hidden_states=norm(hidden, layer.input_layernorm))
        hidden = hidden + norm(attn, layer.post_attention_layernorm)
        mlp = layer.mlp(norm(hidden, layer.pre_feedforward_layernorm))
        hidden = hidden + norm(mlp, layer.post_feedforward_layernorm)
        if version == 4:
            hidden = hidden * layer.layer_scalar
        outputs.append(hidden)
    return norm(hidden, model.norm), [outputs[i] for i in model.layers_to_capture]


class TestGemmaAuxHiddenStates(CustomTestCase):
    def check_captures(self, version, device, dtype, tokens, graph=False):
        model = make_model(version, device, dtype)
        generator = torch.Generator(device=device).manual_seed(42)
        inputs = torch.randn(tokens, 128, generator=generator, device=device).to(dtype)
        positions = torch.arange(tokens, device=device)

        def forward():
            return model(None, positions, None, input_embeds=inputs.clone())

        expected_final, expected_captures = eager_reference(model, inputs, version)
        final, captured = forward()
        if graph:
            for _ in range(3):
                forward()
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                final, captured = forward()
            cuda_graph.replay()
            cuda_graph.replay()
            torch.cuda.synchronize()
        atol, rtol = (0.08, 0.02) if dtype == torch.bfloat16 else (0.005, 0.005)
        self.assertEqual(len(captured), len(expected_captures))
        for actual, expected in zip(captured, expected_captures):
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(final, expected_final, atol=atol, rtol=rtol)
        model.layers_to_capture = []
        torch.testing.assert_close(final, forward(), atol=0, rtol=0)

    @torch.no_grad()
    def test_native_captures_match_eager_residual_additions(self):
        for version in (3, 4):
            for cpu_amx in (False, True):
                with (
                    self.subTest(version=version, cpu_amx=cpu_amx),
                    patch("sglang.srt.models.gemma3_causal._is_cpu", cpu_amx),
                    patch(
                        "sglang.srt.models.gemma3_causal._is_cpu_amx_available",
                        cpu_amx,
                        create=True,
                    ),
                ):
                    self.check_captures(version, "cpu", torch.float32, 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.no_grad()
    def test_fused_captures_survive_inplace_norms_and_graph_replay(self):
        for version in (3, 4):
            for dtype in (torch.float16, torch.bfloat16):
                for tokens in (1, 8):
                    with self.subTest(version=version, dtype=dtype, tokens=tokens):
                        self.check_captures(version, "cuda", dtype, tokens, graph=True)


if __name__ == "__main__":
    unittest.main()
