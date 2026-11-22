"""Klein packed Q/K fusion against the actual native RMSNorm/RoPE dispatch."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    can_use_flux2_strided_qknorm_rope,
    flux2_strided_qknorm_rope,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm_with_optional_rope,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding.utils import (
    flashinfer_apply_rope_inplace,
)
from sglang.multimodal_gen.runtime.models.dits import flux_2
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def inputs(batch, tokens, heads=24):
    packed = torch.randn(
        batch, tokens, 9 * heads * 128, device="cuda", dtype=torch.bfloat16
    )
    q, k, _ = packed[:, :, : 3 * heads * 128].chunk(3, dim=-1)
    q, k = (x.unflatten(-1, (heads, 128)) for x in (q, k))
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


def _documented_arithmetic(x, weight, cache, eps):
    """Emulate the kernel's steps per row: fp32 RMSNorm, cast to bf16, then an
    interleaved rotation in fp32, cast back to bf16."""
    v = x.float()
    rstd = torch.rsqrt((v * v).sum(-1, keepdim=True) / 128.0 + eps)
    normed = (v * rstd * weight.float()).to(torch.bfloat16).float()
    positions = torch.arange(x.shape[1], device=x.device)
    cos = cache[positions, :64].view(1, x.shape[1], 1, 64)
    sin = cache[positions, 64:].view(1, x.shape[1], 1, 64)
    even, odd = normed[..., 0::2], normed[..., 1::2]
    out = torch.empty_like(normed)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = odd * cos + even * sin
    return out.to(torch.bfloat16)


def _packed_qk_fusion_supported() -> bool:
    """The dispatch this test covers mounts on NVIDIA SM90 and newer."""
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and current_arch().is_sm_at_least(9, 0)
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
        for batch, tokens in [(1, 17), (2, 257), (1, 4608)]:
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
                    if _packed_qk_fusion_supported():
                        self.assertTrue(
                            can_use_flux2_strided_qknorm_rope(
                                q, k, qn.weight, kn.weight, cache
                            )
                        )
                        if flashinfer_apply_rope_inplace is not None:
                            # Bit-exactness is contracted against
                            # FlashInfer's interleaved RoPE, the dispatch
                            # this kernel replaces. Without FlashInfer the
                            # reference takes a different rounding path, so
                            # the gate correctly returns the reference Q/K
                            # and stays unverified; the output check above
                            # still covers that case unconditionally.
                            self.assertTrue(gate.verified)
                            self.assertFalse(gate.disabled)
                            self.assert_bits(
                                flux2_strided_qknorm_rope(
                                    q,
                                    k,
                                    qn.weight,
                                    kn.weight,
                                    cache,
                                    qn.variance_epsilon,
                                ),
                                reference(q, k, qn, kn, cache),
                            )
                    else:
                        self.assertFalse(gate.verified)
                        self.assertFalse(gate.disabled)

    @torch.inference_mode()
    def test_eligibility_does_not_depend_on_the_head_count(self):
        """The kernel indexes rows by ``(token, head)``; ``HEADS`` is a
        ``constexpr``. Neither the fusion's math nor its cost model has a
        24-head term, so the gate must accept any head count the packed
        layout can express (FLUX.2-dev is 48 heads; a TP split makes
        ``heads / tp_size``, which must not have to land on 24).
        """

        class _Hopper:
            def is_sm(self, major, minor=None):
                return True

            def is_sm_at_least(self, major, minor=0):
                return True

        with patch(
            "sglang.kernels.ops.diffusion.rope.flux2_qknorm_rope_triton.current_arch",
            _Hopper,
        ):
            for heads in (8, 12, 24, 32, 48):
                with self.subTest(heads=heads):
                    _, q, k, qn, kn, cache = inputs(1, 64, heads)
                    self.assertTrue(
                        can_use_flux2_strided_qknorm_rope(
                            q, k, qn.weight, kn.weight, cache
                        ),
                        f"head count {heads} must not be rejected",
                    )

    @torch.inference_mode()
    def test_head_generic_kernel_matches_its_documented_arithmetic(self):
        """Run the fused kernel at a non-24 head count and compare against a
        per-row emulation of the arithmetic the kernel documents. Differences
        are bf16 ULP-scale rounding-order only, which is what proves the
        row -> (token, head) mapping is right at any head count."""
        for heads in (24, 48):
            with self.subTest(heads=heads):
                _, q, k, qn, kn, cache = inputs(1, 257, heads)
                for magnitude in (0.005, 1.0):
                    q.normal_().mul_(magnitude)
                    k.normal_().mul_(magnitude)
                    fused_q, fused_k = flux2_strided_qknorm_rope(
                        q, k, qn.weight, kn.weight, cache, qn.variance_epsilon
                    )
                    for fused, src, norm in ((fused_q, q, qn), (fused_k, k, kn)):
                        expected = _documented_arithmetic(
                            src, norm.weight, cache, norm.variance_epsilon
                        )
                        differing = (
                            fused.view(torch.int16) != expected.view(torch.int16)
                        ).sum()
                        # A wrong row mapping moves whole rows (hundreds of
                        # thousands of elements), not a handful of roundings.
                        self.assertLess(differing.item() / fused.numel(), 1e-4)

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


if __name__ == "__main__":
    unittest.main()
