"""Lossless LingBot modulation sites against their native Torch/CuTe paths."""

import unittest

import torch

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    try_fused_fp32_layernorm_bf16,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    ScaleResidualLayerNormScaleShift,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    CausalLingBotWorldTransformerBlock,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def make_sites(channels):
    # These methods need only the existing normalization and MulAdd modules;
    # avoid allocating the unrelated attention/FFN weights in kernel tests.
    block = CausalLingBotWorldTransformerBlock.__new__(
        CausalLingBotWorldTransformerBlock
    )
    torch.nn.Module.__init__(block)
    with torch.inference_mode(False):
        block.norm1 = FP32LayerNorm(channels, eps=1e-6, elementwise_affine=False).cuda()
        block.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            channels, eps=1e-6, elementwise_affine=True, dtype=torch.float32
        ).cuda()
    block._norm1_modulation_gate = BitExactFusionGate("test norm1", per_signature=True)
    block._cross_norm_gate = BitExactFusionGate("test cross norm", per_signature=True)
    block._self_residual_gate = BitExactFusionGate("test residual", per_signature=True)
    return block


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "NVIDIA CUDA required"
)
class TestLingBotModulation(CustomTestCase):
    def assert_bits(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )

    def assert_norm_dispatch(self, gate, exact=True):
        # Only Hopper's native FP32 reduction order is supported. Other
        # devices retain the reference without entering the exactness gate.
        if torch.cuda.get_device_capability() == (9, 0):
            self.assertEqual(gate.verified, exact)
            self.assertEqual(gate.disabled, not exact)
        else:
            self.assertFalse(gate.verified)
            self.assertFalse(gate.disabled)

    @torch.inference_mode()
    def test_norm1_per_frame_and_replay(self):
        torch.manual_seed(42)
        block = make_sites(256)
        x = torch.randn(2, 63, 256, device="cuda", dtype=torch.bfloat16)
        table = torch.randn(2, 3, 6, 256, device="cuda", dtype=torch.float32)
        shift, scale = table.chunk(6, dim=2)[:2]
        normalized = block.norm1(x.float())
        expected = (
            (normalized.unflatten(1, (3, 21)) * (1 + scale) + shift)
            .flatten(1, 2)
            .bfloat16()
        )
        raw = try_fused_fp32_layernorm_bf16(
            x.view(6, 21, 256),
            scale.reshape(6, 256).contiguous(),
            shift.reshape(6, 256).contiguous(),
            1e-6,
        ).view_as(x)
        exact = torch.equal(raw.view(torch.int16), expected.view(torch.int16))
        self.assert_bits(block._fp32_norm(x, scale, shift), expected)
        self.assert_norm_dispatch(block._norm1_modulation_gate, exact)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = block._fp32_norm(x, scale, shift)
        x.normal_()
        table.normal_()
        normalized = block.norm1(x.float())
        graph.replay()
        expected = (
            (normalized.unflatten(1, (3, 21)) * (1 + scale) + shift)
            .flatten(1, 2)
            .bfloat16()
        )
        self.assert_bits(output, expected)
        self.assert_norm_dispatch(block._norm1_modulation_gate)

    @torch.inference_mode()
    def test_residual_matches_native_cute_and_replay(self):
        for channels, frames, tokens in [(256, 1, 255), (5120, 3, 63), (5120, 1, 4680)]:
            with self.subTest(channels=channels, frames=frames):
                block = make_sites(channels)
                residual = torch.randn(
                    1, tokens, channels, device="cuda", dtype=torch.bfloat16
                )
                update = torch.randn_like(residual)
                gate = torch.randn(
                    1, frames, 1, channels, device="cuda", dtype=torch.float32
                )
                zero = residual.new_zeros((1,))
                expected = block.self_attn_residual_norm(
                    residual, update, gate, zero, zero
                )[1]
                self.assert_bits(
                    block._self_attn_residual(residual, update, gate), expected
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = block._self_attn_residual(residual, update, gate)
                residual.normal_()
                update.normal_()
                gate.normal_()
                graph.replay()
                self.assert_bits(
                    output,
                    block.self_attn_residual_norm(residual, update, gate, zero, zero)[
                        1
                    ],
                )
                self.assertFalse(block._self_residual_gate.disabled)
                # Reuse the verified signature with signed-zero gates: a +0
                # inserted before the multiply would flip all residual bits.
                residual.zero_().neg_()
                update.fill_(1)
                gate.zero_().neg_()
                self.assert_bits(
                    block._self_attn_residual(residual, update, gate),
                    block.self_attn_residual_norm(residual, update, gate, zero, zero)[
                        1
                    ],
                )
        block = make_sites(256)
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        values = bits.view(torch.bfloat16)
        update = values[torch.isfinite(values)].view(1, 255, 256)
        residual = torch.zeros_like(update)
        gate = torch.linspace(-2, 2, 256, device="cuda").view(1, 1, 1, 256)
        zero = residual.new_zeros((1,))
        self.assert_bits(
            block._self_attn_residual(residual, update, gate),
            block.self_attn_residual_norm(residual, update, gate, zero, zero)[1],
        )
        self.assertFalse(block._self_residual_gate.disabled)

    @torch.no_grad()
    def test_affine_norm_and_live_parameters(self):
        block = make_sites(5120)
        x = torch.randn(1, 63, 5120, device="cuda", dtype=torch.bfloat16)
        norm = block.self_attn_residual_norm.norm
        norm.weight.normal_()
        norm.bias.normal_()
        self.assert_bits(block._fp32_norm(x), norm(x))
        self.assert_norm_dispatch(block._cross_norm_gate)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = block._fp32_norm(x)
        x.normal_()
        norm.weight.normal_()
        norm.bias.normal_()
        graph.replay()
        self.assert_bits(out, norm(x))
        # Native FP32 parameter caching must also follow a BF16 conversion.
        norm.bfloat16()
        self.assert_bits(block._fp32_norm(x), norm(x))
        norm.weight.normal_()
        self.assert_bits(block._fp32_norm(x), norm(x))
        self.assert_norm_dispatch(block._cross_norm_gate)


if __name__ == "__main__":
    unittest.main()
