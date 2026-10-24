"""aiter block-fp8 fused MoE at tiny M vs a float32 torch reference.

For K=2048, topk=8, inter=512 (Qwen3.5/3.6-FP8 MoE at TP1), aiter's default
heuristic splits stage1 eight ways at M=1, so each split owns a single 256-wide
K tile. The split-K accumulator is then never zeroed and the output is
garbage. The sglang aiter runner installs a guard that runs such calls unsplit
(ROCm/aiter#4032).
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


def _rocm_with_aiter():
    if not (torch.version.hip and torch.cuda.is_available()):
        return False
    try:
        import aiter  # noqa: F401
    except ImportError:
        return False
    return True


def _block_quant_128(w):
    # [E, N, K] -> fp8 [E, N, K] with one fp32 scale per 128x128 block
    from aiter import dtypes, pertoken_quant

    e, n, k = w.shape
    blocks = w.view(e, n // 128, 128, k // 128, 128).permute(0, 1, 3, 2, 4)
    q, s = pertoken_quant(
        blocks.reshape(e, -1, 128 * 128).contiguous(), quant_dtype=dtypes.fp8
    )
    q = q.view(e, n // 128, k // 128, 128, 128).permute(0, 1, 3, 2, 4)
    return q.reshape(e, n, k).contiguous(), s.view(e, n // 128, k // 128)


def _dequant_128(q, s):
    return q.float() * s.repeat_interleave(128, 1).repeat_interleave(128, 2)


@unittest.skipUnless(_rocm_with_aiter(), "ROCm with aiter only")
class TestAiterMoeSplitKKBatchGuard(CustomTestCase):
    E, INTER, DIM, TOPK = 16, 512, 2048, 8

    @classmethod
    def setUpClass(cls):
        from aiter.ops.shuffle import shuffle_weight

        from sglang.srt.layers.moe.moe_runner.aiter import (
            apply_aiter_splitk_kbatch_guard,
        )

        apply_aiter_splitk_kbatch_guard()
        torch.manual_seed(0)
        dev = torch.device("cuda", 0)
        w1 = torch.randn(cls.E, 2 * cls.INTER, cls.DIM, device=dev) * 0.05
        w2 = torch.randn(cls.E, cls.DIM, cls.INTER, device=dev) * 0.05
        w1_q, cls.w1_scale = _block_quant_128(w1.to(torch.bfloat16))
        w2_q, cls.w2_scale = _block_quant_128(w2.to(torch.bfloat16))
        cls.w1_ref = _dequant_128(w1_q, cls.w1_scale)
        cls.w2_ref = _dequant_128(w2_q, cls.w2_scale)
        cls.w1 = shuffle_weight(w1_q, (16, 16))
        cls.w2 = shuffle_weight(w2_q, (16, 16))
        # same layout flag the fp8 MoE method sets on its shuffled weights
        cls.w1.is_shuffled = True
        cls.w2.is_shuffled = True
        cls.dev = dev

    def _reference(self, x, topk_weights, topk_ids):
        from aiter import dtypes, pertoken_quant

        # aiter quantizes activations per 1x128 group before stage1
        x_q, x_s = pertoken_quant(x.view(x.shape[0], -1, 128), quant_dtype=dtypes.fp8)
        x_deq = (x_q.float() * x_s).view(x.shape[0], -1)
        out = torch.zeros(x.shape[0], self.DIM, device=self.dev)
        for t in range(x.shape[0]):
            for slot in range(self.TOPK):
                e = int(topk_ids[t, slot])
                h = x_deq[t] @ self.w1_ref[e].t()
                act = F.silu(h[: self.INTER]) * h[self.INTER :]
                out[t] += float(topk_weights[t, slot]) * (act @ self.w2_ref[e].t())
        return out

    def test_small_m_matches_reference(self):
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe, fused_topk

        for m in (1, 2, 4, 8):
            with self.subTest(m=m):
                x = torch.randn(m, self.DIM, device=self.dev).to(torch.bfloat16)
                logits = torch.randn(m, self.E, device=self.dev).to(torch.bfloat16)
                topk_weights, topk_ids = fused_topk(x, logits, self.TOPK, True)
                out = fused_moe(
                    x,
                    self.w1,
                    self.w2,
                    topk_weights,
                    topk_ids,
                    quant_type=QuantType.per_128x128,
                    activation=ActivationType.Silu,
                    w1_scale=self.w1_scale,
                    w2_scale=self.w2_scale,
                ).float()
                ref = self._reference(x, topk_weights, topk_ids)
                self.assertTrue(torch.isfinite(out).all())
                cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
                self.assertGreater(cos.item(), 0.99)


if __name__ == "__main__":
    unittest.main()
