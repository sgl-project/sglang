"""Lossless LingBot modulation sites against their native Torch/CuTe paths."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    modulate_scale_shift,
    try_fused_fp32_layernorm_bf16,
)
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    ScaleResidualLayerNormScaleShift,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    CausalLingBotWorldTransformerBlock,
    LingBotWorldCamConditioner,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


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
    block.mlp_residual = MulAdd()
    block._norm1_modulation_gate = BitExactFusionGate("test norm1", per_signature=True)
    block._cross_norm_gate = BitExactFusionGate("test cross norm", per_signature=True)
    block._self_residual_gate = BitExactFusionGate("test residual", per_signature=True)
    return block


def native_modulation(x, scale, shift):
    return (
        F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6) * (1 + scale[:, None])
        + shift[:, None]
    ).bfloat16()


def make_camera_site():
    # scale_shift is supplied by the caller; no parallel MLP is needed here.
    camera = LingBotWorldCamConditioner.__new__(LingBotWorldCamConditioner)
    torch.nn.Module.__init__(camera)
    return camera


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "NVIDIA CUDA required"
)
class TestLingBotModulation(CustomTestCase):
    def assert_bits(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )

    def assert_norm_dispatch(self, gate):
        # A live Torch reduction can differ on another device. The site must
        # either verify exact fusion or permanently choose its native path.
        self.assertNotEqual(gate.verified, gate.disabled)
        if torch.cuda.get_device_capability() == (9, 0):
            self.assertTrue(gate.verified)
            self.assertFalse(gate.disabled)

    @torch.inference_mode()
    def test_fp32_norm_shapes_and_rounding(self):
        torch.manual_seed(42)
        for shape in [(1, 17, 256), (2, 63, 2240), (1, 4680, 5120)]:
            for values in ("random", "constant", "small_variance", "signed_zero"):
                with self.subTest(shape=shape, values=values):
                    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                    if values == "constant":
                        x.fill_(1)
                    elif values == "small_variance":
                        x.fill_(1)
                        x[:, ::2, ::7] = 1.0078125
                    elif values == "signed_zero":
                        x.zero_().neg_()
                    scale = torch.randn((shape[0], shape[2]), device="cuda")
                    shift = torch.randn_like(scale)
                    raw = try_fused_fp32_layernorm_bf16(x, scale, shift, 1e-6)
                    reference = native_modulation(x, scale, shift)
                    block = make_sites(shape[-1])
                    self.assert_bits(
                        block._fp32_norm(x, scale[:, None, None], shift[:, None, None]),
                        reference,
                    )
                    exact = torch.equal(
                        raw.view(torch.int16), reference.view(torch.int16)
                    )
                    self.assertEqual(block._norm1_modulation_gate.disabled, not exact)
                    self.assertEqual(block._norm1_modulation_gate.verified, exact)
                    # Raw reduction equivalence was validated on Hopper. Other
                    # Torch/device dispatches may differ (B200, D=2240); the
                    # production site must exercise its live exactness gate.
                    if torch.cuda.get_device_capability() == (9, 0):
                        self.assert_bits(raw, reference)
                    weight, bias = (
                        scale[:1].expand(shape[0], -1),
                        shift[:1].expand(shape[0], -1),
                    )
                    raw = try_fused_fp32_layernorm_bf16(
                        x, weight, bias, 1e-6, affine=True
                    )
                    reference = F.layer_norm(
                        x.float(), (shape[-1],), weight[0], bias[0], 1e-6
                    ).bfloat16()
                    norm = block.self_attn_residual_norm.norm
                    norm.weight.copy_(weight[0])
                    norm.bias.copy_(bias[0])
                    self.assert_bits(block._fp32_norm(x), reference)
                    exact = torch.equal(
                        raw.view(torch.int16), reference.view(torch.int16)
                    )
                    self.assertEqual(block._cross_norm_gate.disabled, not exact)
                    self.assertEqual(block._cross_norm_gate.verified, exact)
                    if torch.cuda.get_device_capability() == (9, 0):
                        self.assert_bits(raw, reference)

    @torch.inference_mode()
    def test_camera_rounding_finite_bf16_and_replay(self):
        camera = make_camera_site()
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        values = bits.view(torch.bfloat16)
        x = values[torch.isfinite(values)].view(1, 1020, 64)
        scale = (
            torch.linspace(-1.5, 1.5, x.numel(), device="cuda").bfloat16().view_as(x)
        )
        shift = torch.zeros_like(x)
        expected = (1 + scale) * x + shift
        with patch(
            "sglang.multimodal_gen.runtime.models.dits.lingbot_world.modulate_scale_shift",
            wraps=modulate_scale_shift,
        ) as fused:
            self.assert_bits(camera(x, x, (scale, shift)), expected)
            self.assertEqual(fused.call_count, 1)
        # Same verified layout, but a different activation and affine state.
        x = torch.randn_like(x)
        scale = torch.randn_like(x)
        shift = torch.randn_like(x)
        camera(x, x, (scale, shift))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = camera(x, x, (scale, shift))
        x.normal_()
        scale.normal_()
        shift.normal_()
        graph.replay()
        self.assert_bits(output, (1 + scale) * x + shift)

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
        self.assertEqual(block._norm1_modulation_gate.disabled, not exact)
        self.assertEqual(block._norm1_modulation_gate.verified, exact)
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

    @torch.inference_mode()
    def test_unverified_capture_and_mismatch_fallback(self):
        block = make_sites(256)
        normalized = torch.randn(1, 17, 256, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(1, 1, 1, 256, device="cuda")
        shift = torch.randn_like(scale)
        ref = native_modulation(normalized, scale.view(1, 256), shift.view(1, 256))
        module = "sglang.multimodal_gen.runtime.models.dits.lingbot_world"
        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            patch(
                module + ".try_fused_fp32_layernorm_bf16",
                side_effect=AssertionError("unverified fusion ran during capture"),
            ),
        ):
            self.assert_bits(block._fp32_norm(normalized, scale, shift), ref)
        with patch(
            module + ".try_fused_fp32_layernorm_bf16",
            return_value=torch.zeros_like(ref),
        ):
            self.assert_bits(block._fp32_norm(normalized, scale, shift), ref)
        self.assertTrue(block._norm1_modulation_gate.disabled)
        with patch(
            module + ".try_fused_fp32_layernorm_bf16",
            side_effect=AssertionError("disabled fusion ran"),
        ):
            self.assert_bits(block._fp32_norm(normalized, scale, shift), ref)

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

    @torch.inference_mode()
    def test_causal_block_forward_and_cache_update(self):
        block = make_sites(256)
        block.tp_rmsnorm = False
        block.local_num_heads = 2
        block.dim_head = 128
        block.norm_q = torch.nn.Identity()
        block.norm_k = torch.nn.Identity()
        block.ffn = torch.nn.Identity()
        block.cam_conditioner = make_camera_site()
        with torch.inference_mode(False):
            block.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
                256, eps=1e-6, elementwise_affine=False, dtype=torch.float32
            ).cuda()
        block.scale_shift_table = torch.randn(1, 6, 256, device="cuda")
        x = torch.randn(1, 63, 256, device="cuda", dtype=torch.bfloat16)
        temb = torch.randn(1, 3, 6, 256, device="cuda")
        camera = (torch.randn_like(x), torch.randn_like(x))
        # Keep the full block's norm/residual/camera order, replacing only
        # unrelated attention and projections with deterministic outputs.
        with (
            patch.object(block, "_project_qkv", side_effect=lambda x: (x, x, x)),
            patch.object(
                block, "attn1", create=True, side_effect=lambda q, k, v, *a, **kw: v
            ),
            patch.object(block, "to_out", create=True, side_effect=lambda x: (x, None)),
            patch.object(block, "_cross_attn_with_cache", side_effect=lambda x, *a: x),
            patch.object(block, "_fp32_norm", wraps=block._fp32_norm) as norm,
        ):
            kwargs = dict(cam_conditioner_scale_shift=camera)
            actual = block(x, x, temb, (), None, **kwargs)
            self.assertEqual(norm.call_count, 2)
            self.assert_norm_dispatch(block._norm1_modulation_gate)
            self.assert_norm_dispatch(block._cross_norm_gate)
            self.assertTrue(block._self_residual_gate.verified)
            block._norm1_modulation_gate.disable()
            block._cross_norm_gate.disable()
            block._self_residual_gate.disable()
            self.assert_bits(actual, block(x, x, temb, (), None, **kwargs))
            norm.reset_mock()
            self.assert_bits(
                block(x, x, temb, (), None, update_cache_only=True, **kwargs), x
            )
            self.assertEqual(norm.call_count, 1)

    def test_unsupported_and_gradients(self):
        x = torch.randn(1, 17, 64, device="cuda", requires_grad=True)
        row = torch.randn(1, 64, device="cuda")
        self.assertIsNone(try_fused_fp32_layernorm_bf16(x, row, row, 1e-6))
        self.assertIsNone(try_fused_fp32_layernorm_bf16(x.bfloat16(), row, row, 1e-6))
        camera = make_camera_site()
        scale = torch.randn_like(x, requires_grad=True)
        shift = torch.randn_like(x, requires_grad=True)
        result = camera(x, x, (scale, shift))
        result.sum().backward()
        torch.testing.assert_close(x.grad, 1 + scale)
        torch.testing.assert_close(scale.grad, x)
        torch.testing.assert_close(shift.grad, torch.ones_like(shift))
        with torch.inference_mode():
            self.assertIsNone(
                try_fused_fp32_layernorm_bf16(x.transpose(1, 2), row, row, 1e-6)
            )
            self.assertIsNone(
                try_fused_fp32_layernorm_bf16(
                    x.bfloat16(), row.bfloat16(), row.bfloat16(), 1e-6
                )
            )
            with patch("torch.compiler.is_compiling", return_value=True):
                self.assertIsNone(try_fused_fp32_layernorm_bf16(x, row, row, 1e-6))


if __name__ == "__main__":
    unittest.main()
