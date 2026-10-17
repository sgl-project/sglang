"""Kimi-K3's prefill MoE with MXFP4 activations on the gate/up-interleaved a8w4 weights."""

import os
import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: aiter's gfx950 a4w4 / a8w4 SiTU MoE kernels.
register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")

# the activation modes the K3 serving recipe sets for aiter's SiTU MoE
os.environ.setdefault("AITER_SITUV2_A8W4", "1")
os.environ.setdefault("AITER_FLYDSL_FORCE", "1")

# Kimi-K3 TP8 latent MoE: 896 experts, 3584 latent, 384 intermediate, top-16, SiTUv2.
E, D, I, TOPK = 896, 3584, 384, 16
BETA, LINEAR_BETA = 4.0, 25.0


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


@unittest.skipUnless(is_gfx95_supported(), "gfx950 MXFP4 MoE")
class TestKimiK3MoeA4w4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import aiter
        from aiter import QuantType, dtypes
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        quant = aiter.get_torch_quant(QuantType.per_1x32)

        def mxfp4_weights(shape, seed, scale):
            g = torch.Generator(device="cuda").manual_seed(seed)
            w = (torch.randn(E, *shape, generator=g, device="cuda") * scale).bfloat16()
            q, s = quant(w, quant_dtype=dtypes.fp4x2)
            return q.view(E, shape[0], shape[1] // 2), s.view(E * shape[0], -1)

        cls.w1_q, cls.w1_s = mxfp4_weights((2 * I, D), 0, D**-0.5)
        cls.w2_q, cls.w2_s = mxfp4_weights((D, I), 1, I**-0.5)
        # the a8w4 layout mxfp4.py prepares: gate/up-interleaved w13, plain w2
        cls.shuffled = (
            shuffle_weight_a16w4(cls.w1_q, 16, True),
            shuffle_scale_a16w4(cls.w1_s, E, True),
            shuffle_weight_a16w4(cls.w2_q, 16, False),
            shuffle_scale_a16w4(cls.w2_s, E, False),
        )

    def _moe(self, x, weights, ids, fp4_activations):
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.moe_common import GateMode

        w1, w1_scale, w2, w2_scale = self.shuffled
        mode = (
            dict(quant_dtype_a=dtypes.fp4x2, gate_mode=GateMode.INTERLEAVE.value)
            if fp4_activations
            else dict(gate_mode=GateMode.SEPARATED.value)
        )
        return fused_moe(
            x, w1, w2, weights, ids,
            activation=ActivationType.Situv2, quant_type=QuantType.per_1x32,
            w1_scale=w1_scale, w2_scale=w2_scale, beta=BETA, linear_beta=LINEAR_BETA,
            **mode,
        )  # fmt: skip

    @torch.inference_mode()
    def test_interleaved_a4w4_matches_a8w4(self):
        """From the token count SGLANG_ROCM_K3_PREFILL_MOE_A4W4_MIN_TOKENS defaults to,
        aiter's a4w4 config resolves to the stage 1 that honors the interleaved gate/up
        layout, and its output tracks the a8w4 path over the same weights."""
        from aiter.fused_moe import get_padded_M

        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.moe_runner.aiter import (
            _situ_a4w4_interleave_resolves,
        )

        min_tokens = envs.SGLANG_ROCM_K3_PREFILL_MOE_A4W4_MIN_TOKENS.get()
        for tokens in (min_tokens, 2048):
            g = torch.Generator(device="cuda").manual_seed(tokens)
            x = torch.randn(tokens, D, generator=g, device="cuda").bfloat16()
            scores = torch.sigmoid(torch.randn(tokens, E, generator=g, device="cuda"))
            weights, ids = torch.topk(scores, TOPK, dim=-1)
            weights = (weights / weights.sum(-1, keepdim=True)).float()
            ids = ids.to(torch.int32)
            resolves = _situ_a4w4_interleave_resolves(
                get_padded_M(tokens), D, I, E, TOPK, BETA, LINEAR_BETA
            )
            with self.subTest(tokens=tokens):
                self.assertTrue(resolves)
                a8w4 = self._moe(x, weights, ids, fp4_activations=False)
                a4w4 = self._moe(x, weights, ids, fp4_activations=True)
                # MXFP4 activations cost ~0.03 of cosine against MXFP8 ones here; a
                # stage 1 that misreads the interleaved gate/up rows lands near 0.
                self.assertGreater(_cos(a4w4, a8w4), 0.95)


if __name__ == "__main__":
    unittest.main()
