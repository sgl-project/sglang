"""Klein packed Q/K fusion against the actual native RMSNorm/RoPE dispatch."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    can_use_flux2_strided_qknorm_rope,
    flux2_strided_qknorm_rope,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm_with_optional_rope,
)
from sglang.multimodal_gen.runtime.models.dits import flux_2
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def inputs(batch, tokens):
    packed = torch.randn(batch, tokens, 9 * 3072, device="cuda", dtype=torch.bfloat16)
    q, k, _ = packed[:, :, : 3 * 3072].chunk(3, dim=-1)
    q, k = (x.unflatten(-1, (24, 128)) for x in (q, k))
    qn, kn = (RMSNorm(128, eps=1e-6).to("cuda", torch.bfloat16) for _ in range(2))
    qn.weight.normal_()
    kn.weight.normal_()
    angles = torch.randn(tokens, 64, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return packed, q, k, qn, kn, cache


def reference(q, k, qn, kn, cache):
    return apply_qk_norm_with_optional_rope(
        q,
        k,
        qn,
        kn,
        128,
        cache,
        is_neox=False,
        allow_inplace=True,
        allow_strided_qk=False,
    )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "NVIDIA CUDA required"
)
class TestFlux2StridedQKNormRoPE(CustomTestCase):
    def assert_bits(self, actual, expected):
        for a, b in zip(actual, expected, strict=True):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))

    @torch.inference_mode()
    def test_native_shapes_magnitudes_and_untouched_packed_input(self):
        torch.manual_seed(42)
        for batch, tokens in [(1, 1), (1, 17), (2, 257), (1, 4608)]:
            packed, q, k, qn, kn, cache = inputs(batch, tokens)
            for magnitude in (0.005, 1.0, 200.0):
                with self.subTest(batch=batch, tokens=tokens, magnitude=magnitude):
                    packed.normal_().mul_(magnitude)
                    original = packed.clone()
                    gate = BitExactFusionGate("test Klein QK", per_signature=True)
                    with patch.object(flux_2, "_FLUX2_STRIDED_QK_ROPE", gate):
                        actual = flux_2._flux2_single_qk_rope(
                            q, k, qn, kn, 128, cache, None
                        )
                    self.assert_bits(actual, reference(q, k, qn, kn, cache))
                    self.assertTrue(
                        torch.equal(
                            packed.view(torch.int16), original.view(torch.int16)
                        )
                    )
                    if can_use_flux2_strided_qknorm_rope(q, k, qn.weight, kn.weight, cache):
                        self.assertTrue(gate.verified)
                        self.assertFalse(gate.disabled)
                        self.assert_bits(
                            flux2_strided_qknorm_rope(
                                q, k, qn.weight, kn.weight, cache, qn.variance_epsilon
                            ),
                            reference(q, k, qn, kn, cache),
                        )
                    else:
                        self.assertFalse(gate.verified)
                        self.assertFalse(gate.disabled)

    @torch.inference_mode()
    def test_changed_inputs_weights_cache_and_graph_replay(self):
        packed, q, k, qn, kn, cache = inputs(1, 513)
        gate = BitExactFusionGate("test Klein replay", per_signature=True)
        with patch.object(flux_2, "_FLUX2_STRIDED_QK_ROPE", gate):
            flux_2._flux2_single_qk_rope(q, k, qn, kn, 128, cache, None)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = flux_2._flux2_single_qk_rope(q, k, qn, kn, 128, cache, None)
            for seed in range(8):
                torch.manual_seed(seed)
                packed.normal_()
                qn.weight.normal_()
                kn.weight.normal_()
                cache.normal_()
                graph.replay()
                self.assert_bits(out, reference(q, k, qn, kn, cache))
            packed.zero_().neg_()
            graph.replay()
            self.assert_bits(out, reference(q, k, qn, kn, cache))

    @torch.inference_mode()
    def test_mismatch_unverified_capture_and_native_dispatch(self):
        _, q, k, qn, kn, cache = inputs(1, 17)
        expected = reference(q, k, qn, kn, cache)
        gate = BitExactFusionGate("test mismatch", per_signature=True)
        with (
            patch.object(flux_2, "_FLUX2_STRIDED_QK_ROPE", gate),
            patch.object(
                flux_2, "can_use_flux2_strided_qknorm_rope", return_value=True
            ),
        ):
            with (
                patch("torch.cuda.is_current_stream_capturing", return_value=True),
                patch.object(flux_2, "flux2_strided_qknorm_rope") as raw,
            ):
                self.assert_bits(
                    flux_2._flux2_single_qk_rope(q, k, qn, kn, 128, cache, None),
                    expected,
                )
                raw.assert_not_called()
            with patch.object(
                flux_2,
                "flux2_strided_qknorm_rope",
                return_value=(torch.zeros_like(q), torch.zeros_like(k)),
            ):
                self.assert_bits(
                    flux_2._flux2_single_qk_rope(q, k, qn, kn, 128, cache, None),
                    expected,
                )
            self.assertTrue(gate.disabled)
            with patch.object(flux_2, "flux2_strided_qknorm_rope") as raw:
                self.assert_bits(
                    flux_2._flux2_single_qk_rope(q, k, qn, kn, 128, cache, None),
                    expected,
                )
                raw.assert_not_called()
        qn._forward_method = qn.forward_native
        kn._forward_method = kn.forward_native
        with (
            patch.object(
                flux_2, "_FLUX2_STRIDED_QK_ROPE", BitExactFusionGate("native")
            ),
            patch.object(flux_2, "flux2_strided_qknorm_rope") as raw,
        ):
            self.assert_bits(
                flux_2._flux2_single_qk_rope(q, k, qn, kn, 128, cache, None),
                reference(q, k, qn, kn, cache),
            )
            raw.assert_not_called()

    @torch.inference_mode()
    def test_unsupported_layout_dtype_device_and_compile(self):
        _, q, k, qn, kn, cache = inputs(2, 17)
        args = (q, k, qn.weight, kn.weight, cache)
        with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
            self.assertFalse(can_use_flux2_strided_qknorm_rope(*args))
        with patch("torch.compiler.is_compiling", return_value=True):
            self.assertFalse(can_use_flux2_strided_qknorm_rope(*args))
        with torch.enable_grad():
            self.assertFalse(can_use_flux2_strided_qknorm_rope(*args))
        for bad_q, bad_k, bad_w, bad_cache in [
            (q.cpu(), k, qn.weight, cache),
            (q.float(), k.float(), qn.weight, cache),
            (q.contiguous(), k.contiguous(), qn.weight, cache),
            (q[:, ::2], k[:, ::2], qn.weight, cache),
            (q[..., ::2], k[..., ::2], qn.weight, cache),
            (q, k, qn.weight.float(), cache),
            (q, k, qn.weight, cache.bfloat16()),
            (q, k, qn.weight, cache[:1]),
        ]:
            self.assertFalse(
                can_use_flux2_strided_qknorm_rope(
                    bad_q, bad_k, bad_w, kn.weight, bad_cache
                )
            )


if __name__ == "__main__":
    unittest.main()
