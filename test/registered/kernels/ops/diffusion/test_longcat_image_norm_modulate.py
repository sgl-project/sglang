"""LongCat normalization parity and graph-safe fusion dispatch."""

import unittest
from unittest.mock import patch

import torch
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle

import sglang.multimodal_gen.runtime.models.dits.longcat_image as longcat
from sglang.kernels.ops.diffusion import BitExactFusionGate, modulate_scale_shift
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestLongCatNormModulation(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.original_gate = longcat._LONGCAT_LN_MOD
        longcat._LONGCAT_LN_MOD = BitExactFusionGate("test", per_signature=True)
        torch.manual_seed(42)

    def tearDown(self):
        longcat._LONGCAT_LN_MOD = self.original_gate
        super().tearDown()

    def require_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    @torch.inference_mode()
    def test_adaln_checkpoint_and_output_parity(self):
        for device, dtype, dim, seq in [
            ("cpu", torch.float32, 64, 17),
            ("cuda", torch.bfloat16, 3072, 512),
            ("cuda", torch.bfloat16, 3072, 4608),
        ]:
            if device == "cuda" and not torch.cuda.is_available():
                continue
            for reference_cls, candidate_cls in [
                (AdaLayerNormZero, longcat._LongCatAdaLayerNormZero),
                (AdaLayerNormZeroSingle, longcat._LongCatAdaLayerNormZeroSingle),
            ]:
                with self.subTest(device=device, seq=seq, cls=reference_cls.__name__):
                    reference = reference_cls(dim).to(device=device, dtype=dtype)
                    candidate = candidate_cls(dim).to(device=device, dtype=dtype)
                    candidate.load_state_dict(reference.state_dict(), strict=True)
                    x = torch.randn(1, seq, dim, device=device, dtype=dtype)
                    emb = torch.randn(1, dim, device=device, dtype=dtype)
                    expected, actual = reference(x, emb=emb), candidate(x, emb=emb)
                    for a, b in zip(expected, actual, strict=True):
                        self.assertTrue(torch.equal(a, b))
                    if device == "cuda":
                        self.assertTrue(longcat._LONGCAT_LN_MOD.verified)
                        self.assertFalse(longcat._LONGCAT_LN_MOD.disabled)

    def inputs(self, seq=4096):
        self.require_cuda()
        x = torch.randn(1, seq, 3072, device="cuda", dtype=torch.bfloat16)
        modulation = torch.randn(1, 6 * 3072, device="cuda", dtype=torch.bfloat16)
        shift, scale, *_ = modulation.chunk(6, dim=-1)
        norm = torch.nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).cuda()
        return norm, x, scale, shift

    def test_grad_enabled_uses_differentiable_reference(self):
        norm, x, scale, shift = self.inputs(seq=17)
        with torch.inference_mode():
            longcat._longcat_norm_modulate(norm, x, scale, shift)
        self.assertTrue(longcat._LONGCAT_LN_MOD.verified)
        leaves = [t.detach().clone().requires_grad_() for t in (x, scale, shift)]
        refs = [t.detach().clone().requires_grad_() for t in leaves]
        with patch.object(longcat.diffusion_ops, "fused_layernorm_modulate") as fused:
            actual = longcat._longcat_norm_modulate(norm, *leaves)
            actual.float().sum().backward()
            fused.assert_not_called()
        expected = norm(refs[0]) * (1 + refs[1][:, None]) + refs[2][:, None]
        expected.float().sum().backward()
        self.assertTrue(torch.equal(actual, expected))
        for a, b in zip(leaves, refs, strict=True):
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.equal(a.grad, b.grad))

    @torch.inference_mode()
    def test_changed_inputs_are_used_by_graph_replay(self):
        norm, x, scale, shift = self.inputs()
        longcat._longcat_norm_modulate(norm, x, scale, shift)
        self.assertTrue(longcat._LONGCAT_LN_MOD.verified)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = longcat._longcat_norm_modulate(norm, x, scale, shift)
        x.add_(0.25)
        scale.neg_()
        shift.mul_(0.5)
        graph.replay()
        expected = norm(x) * (1 + scale[:, None]) + shift[:, None]
        self.assertTrue(torch.equal(actual, expected))

    @torch.inference_mode()
    def test_unverified_capture_uses_eager_reference(self):
        norm, x, scale, shift = self.inputs(seq=17)
        expected = modulate_scale_shift(norm(x), scale, shift)
        with patch.object(longcat.diffusion_ops, "fused_layernorm_modulate") as fused:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = longcat._longcat_norm_modulate(norm, x, scale, shift)
            graph.replay()
            fused.assert_not_called()
        self.assertFalse(longcat._LONGCAT_LN_MOD.verified)
        self.assertTrue(torch.equal(actual, expected))

    @torch.inference_mode()
    def test_mismatch_disables_fusion_and_returns_reference(self):
        norm, x, scale, shift = self.inputs(seq=17)
        expected = norm(x) * (1 + scale[:, None]) + shift[:, None]
        with patch.object(
            longcat.diffusion_ops,
            "fused_layernorm_modulate",
            return_value=torch.zeros_like(x),
        ):
            actual = longcat._longcat_norm_modulate(norm, x, scale, shift)
        self.assertTrue(longcat._LONGCAT_LN_MOD.disabled)
        self.assertTrue(torch.equal(actual, expected))
        with patch.object(longcat.diffusion_ops, "fused_layernorm_modulate") as fused:
            actual = longcat._longcat_norm_modulate(norm, x, scale, shift)
            fused.assert_not_called()
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
