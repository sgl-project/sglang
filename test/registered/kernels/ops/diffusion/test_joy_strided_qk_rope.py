import os
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.diffusion.sites.bitexact_gate import BitExactFusionGate
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm_with_optional_rope,
)
from sglang.multimodal_gen.runtime.models.dits import joy_image
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def make_inputs(batch=1, tokens=4096):
    packed = torch.randn(batch, tokens, 3, 32, 128, device="cuda", dtype=torch.bfloat16)
    q, k, _ = packed.unbind(2)
    norms = [RMSNorm(128, eps=1e-6).to(device="cuda", dtype=q.dtype) for _ in range(2)]
    for norm in norms:
        norm.weight.data.copy_(torch.rand_like(norm.weight) + 0.5)
    angles = torch.randn(tokens, 64, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return packed, (q, k, *norms, cache, None)


def reference(inputs):
    q, k, q_norm, k_norm, cache, complex_freqs = inputs
    return apply_qk_norm_with_optional_rope(
        q.contiguous(),
        k.contiguous(),
        q_norm,
        k_norm,
        128,
        cos_sin_cache=cache,
        freqs_complex=complex_freqs,
        is_neox=False,
    )


class TestJoyStridedQKRoPE(CustomTestCase):
    def assert_bits(self, actual, expected):
        for a, b in zip(actual, expected, strict=True):
            self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))

    @torch.inference_mode()
    def test_production_batch_shapes_and_pristine_input(self):
        for shape in [(1, 8048), (2, 2049)]:
            with self.subTest(shape=shape):
                packed, inputs = make_inputs(*shape)
                gate = BitExactFusionGate("test", per_signature=True)
                with patch.object(joy_image, "_JOY_IMAGE_QK_ROPE", gate):
                    for scale in [0.001, 1.0, 30.0]:
                        packed.copy_(torch.randn_like(packed) * scale)
                        packed.view(torch.int16)[0, 0, 0, 0, :2] = torch.tensor(
                            [0, -32768], device="cuda", dtype=torch.int16
                        )
                        before = packed.clone()
                        self.assert_bits(
                            joy_image._joy_image_qk_rope(*inputs), reference(inputs)
                        )
                        self.assert_bits((packed,), (before,))
                    self.assertEqual(
                        gate.verified, torch.cuda.get_device_capability() == (9, 0)
                    )

    @torch.inference_mode()
    def test_changed_inputs_weights_cache_graph_replay(self):
        packed, inputs = make_inputs()
        gate = BitExactFusionGate("test", per_signature=True)
        with patch.object(joy_image, "_JOY_IMAGE_QK_ROPE", gate):
            self.assert_bits(joy_image._joy_image_qk_rope(*inputs), reference(inputs))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    joy_image._joy_image_qk_rope(*inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = joy_image._joy_image_qk_rope(*inputs)
            for seed in range(8):
                torch.manual_seed(seed)
                packed.copy_(torch.randn_like(packed))
                for norm in inputs[2:4]:
                    norm.weight.copy_(torch.rand_like(norm.weight) + 0.5)
                angles = torch.randn_like(inputs[4][:, :64])
                inputs[4].copy_(torch.cat((angles.cos(), angles.sin()), dim=-1))
                before = packed.clone()
                graph.replay()
                self.assert_bits(out, reference(inputs))
                self.assert_bits((packed,), (before,))

    @torch.inference_mode()
    def test_small_shape_and_backend_compile_switch_fallback(self):
        _, small = make_inputs(tokens=257)
        _, large = make_inputs()
        with patch.object(
            joy_image.diffusion_kernels, "fused_qknorm_rope_out_of_place"
        ) as fused:
            self.assert_bits(joy_image._joy_image_qk_rope(*small), reference(small))
            with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
                self.assert_bits(joy_image._joy_image_qk_rope(*large), reference(large))
            with patch("torch.compiler.is_compiling", return_value=True):
                self.assert_bits(joy_image._joy_image_qk_rope(*large), reference(large))
            with patch.dict(os.environ, {"SGLANG_ENABLE_FUSED_QKNORM_ROPE": "0"}):
                self.assert_bits(joy_image._joy_image_qk_rope(*large), reference(large))
            fused.assert_not_called()

    @torch.inference_mode()
    def test_partial_output_exception_mismatch_and_unverified_capture(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("Other architectures retain the native path")
        packed, inputs = make_inputs()
        expected = reference(inputs)
        before = packed.clone()

        def partial_output_then_raise(q, k, q_out, k_out, *args, **kwargs):
            q_out.zero_()
            raise RuntimeError("output write failed")

        def wrong_output(q, k, q_out, k_out, *args, **kwargs):
            q_out.zero_()
            k_out.zero_()

        for failure in [partial_output_then_raise, wrong_output]:
            gate = BitExactFusionGate("test", per_signature=True)
            with (
                patch.object(joy_image, "_JOY_IMAGE_QK_ROPE", gate),
                patch.object(
                    joy_image.diffusion_kernels,
                    "fused_qknorm_rope_out_of_place",
                    side_effect=failure,
                ) as fused,
            ):
                self.assert_bits(joy_image._joy_image_qk_rope(*inputs), expected)
                self.assertTrue(gate.disabled)
                self.assert_bits(joy_image._joy_image_qk_rope(*inputs), expected)
                fused.assert_called_once()
                self.assert_bits((packed,), (before,))
        with (
            patch.object(
                joy_image,
                "_JOY_IMAGE_QK_ROPE",
                BitExactFusionGate("test", per_signature=True),
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            patch.object(
                joy_image.diffusion_kernels, "fused_qknorm_rope_out_of_place"
            ) as fused,
        ):
            self.assert_bits(joy_image._joy_image_qk_rope(*inputs), expected)
            fused.assert_not_called()


if __name__ == "__main__":
    unittest.main()
