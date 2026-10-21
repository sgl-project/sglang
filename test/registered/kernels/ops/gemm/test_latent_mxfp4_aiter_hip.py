"""Numerical and graph-replay tests for Kimi-K3 latent MXFP4 GEMMs."""

import unittest

import torch

from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestKimiK3LatentMXFP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not latent_mxfp4_aiter_hip.supported():
            raise unittest.SkipTest("Kimi-K3 latent MXFP4 needs AITER on gfx950")
        torch.manual_seed(0)
        cls.device = torch.device("cuda", 0)
        cls.projections = []
        for out_features, in_features in ((3584, 7168), (7168, 3584)):
            weight = (
                torch.randn(
                    out_features, in_features, dtype=torch.bfloat16, device=cls.device
                )
                * 0.02
            )
            packed, scale = latent_mxfp4_aiter_hip.pack(weight=weight)
            cls.projections.append((weight, packed, scale))

    def _assert_matches_bf16(self, *, actual, x, weight):
        self.assertEqual(actual.shape, (x.shape[0], weight.shape[0]))
        self.assertEqual(actual.dtype, x.dtype)
        expected = torch.nn.functional.linear(x, weight)
        rel_l2 = (
            (actual.float() - expected.float()).norm() / expected.float().norm()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item()
        # MXFP4 rounding is lossy; these bounds detect layout/scale corruption.
        self.assertLess(rel_l2, 0.20)
        self.assertGreater(cosine, 0.98)

    def test_run_matches_bf16_reference(self):
        for weight, packed, scale in self.projections:
            for tokens in (1, 8, 2047, 2048, 2049):
                with self.subTest(shape=tuple(weight.shape), tokens=tokens):
                    x = (
                        torch.randn(
                            tokens,
                            weight.shape[1],
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                        * 0.1
                    )
                    actual = latent_mxfp4_aiter_hip.run(x=x, weight=packed, scale=scale)
                    self._assert_matches_bf16(actual=actual, x=x, weight=weight)

    def test_graph_replay_uses_changed_inputs(self):
        for weight, packed, scale in self.projections:
            with self.subTest(shape=tuple(weight.shape)):
                x = (
                    torch.randn(
                        8, weight.shape[1], dtype=torch.bfloat16, device=self.device
                    )
                    * 0.1
                )
                latent_mxfp4_aiter_hip.run(x=x, weight=packed, scale=scale)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = latent_mxfp4_aiter_hip.run(x=x, weight=packed, scale=scale)
                graph.replay()
                first = out.clone()
                x.copy_(torch.randn_like(x) * 0.1)
                graph.replay()
                self.assertFalse(torch.equal(first, out))
                self._assert_matches_bf16(actual=out, x=x, weight=weight)

    def test_rejects_invalid_weight_layout(self):
        for shape in ((7168,), (17, 64), (16, 32), (16, 0)):
            with self.subTest(shape=shape):
                weight = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
                with self.assertRaisesRegex(ValueError, "aligned to 16/64"):
                    latent_mxfp4_aiter_hip.pack(weight=weight)

    def test_rejects_incompatible_activations(self):
        _, packed, scale = self.projections[0]
        for shape, dtype in (
            ((8, 7168), torch.float16),
            ((8, 3584), torch.bfloat16),
            ((8, 1, 7168), torch.bfloat16),
            ((0, 7168), torch.bfloat16),
        ):
            with self.subTest(shape=shape, dtype=dtype):
                x = torch.empty(shape, dtype=dtype, device=self.device)
                with self.assertRaises(ValueError):
                    latent_mxfp4_aiter_hip.run(x=x, weight=packed, scale=scale)

    def test_rejects_missing_or_host_scales(self):
        _, packed, scale = self.projections[0]
        x = torch.ones((1, 7168), dtype=torch.bfloat16, device=self.device)
        for invalid_scale in (scale.cpu(), scale.flatten()[:0]):
            with self.subTest(device=invalid_scale.device, size=invalid_scale.numel()):
                with self.assertRaisesRegex(ValueError, "scales must match"):
                    latent_mxfp4_aiter_hip.run(x=x, weight=packed, scale=invalid_scale)


if __name__ == "__main__":
    unittest.main()
