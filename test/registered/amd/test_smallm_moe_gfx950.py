"""gfx950 small-M MXFP4 fused-MoE kernel (sglang.kernels.ops.moe.smallm_moe_gfx950) vs a bf16 torch reference.

Runs only on ROCm gfx950 with hipcc; everywhere else the module reports itself unavailable and the test is skipped.
"""

import os
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")


def _gfx950():
    return (
        bool(torch.version.hip)
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    )


@unittest.skipUnless(_gfx950(), "gfx950 (MI35x) only")
class TestSmallMMoeGfx950(CustomTestCase):
    E, INTER, DIM, TOPK = (
        16,
        256,
        4096,
        10,
    )  # 10 distinct experts per token, like Qwen3.5's routed topk

    @classmethod
    def setUpClass(cls):
        from aiter import QuantType, dtypes
        from aiter.ops.quant import get_torch_quant
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility.fp4_utils import e8m0_shuffle

        from sglang.kernels.ops.moe import smallm_moe_gfx950 as M

        cls.M = M
        torch.manual_seed(0)
        dev = torch.device("cuda", 0)
        E, INTER, DIM = cls.E, cls.INTER, cls.DIM
        w1 = torch.randn(E, 2 * INTER, DIM, device=dev, dtype=torch.bfloat16) * 0.05
        w2 = torch.randn(E, DIM, INTER, device=dev, dtype=torch.bfloat16) * 0.05
        tq = get_torch_quant(QuantType.per_1x32)
        w1_q, w1_s = tq(w1, quant_dtype=dtypes.fp4x2)
        w2_q, w2_s = tq(w2, quant_dtype=dtypes.fp4x2)
        w1_q = w1_q.view(E, 2 * INTER, DIM // 2)
        w2_q = w2_q.view(E, DIM, INTER // 2)
        cls.w13 = shuffle_weight(w1_q, layout=(16, 16)).contiguous()
        cls.w2 = shuffle_weight(w2_q, layout=(16, 16)).contiguous()
        cls.w13_scale = e8m0_shuffle(w1_s.view(E * 2 * INTER, -1)).contiguous()
        cls.w2_scale = e8m0_shuffle(w2_s.view(E * DIM, -1)).contiguous()
        # bf16 reference on the dequantized weights (what the kernel computes: bf16 activations, fp32 accumulation)
        cls.w1_deq = cls._dequant(w1_q, w1_s.view(E, 2 * INTER, -1))
        cls.w2_deq = cls._dequant(w2_q, w2_s.view(E, DIM, -1))
        cls.dev = dev

    @staticmethod
    def _dequant(packed, scale_e8m0):
        # fp4 e2m1 nibbles (low nibble first) times 2^(e8m0-127) per 32-element group -> float32
        E, N, KB = packed.shape
        u8 = packed.view(torch.uint8)
        lo, hi = u8 & 0xF, u8 >> 4
        nib = torch.stack([lo, hi], dim=-1).reshape(E, N, KB * 2).to(torch.int64)
        mag = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device
        )
        vals = mag[nib & 7] * torch.where(nib & 8 > 0, -1.0, 1.0)
        sc = torch.pow(2.0, scale_e8m0.view(torch.uint8).to(torch.float32) - 127.0)
        return vals.view(E, N, -1, 32) * sc.unsqueeze(-1)

    def _reference(self, x, ids, wts):
        tok = x.shape[0]
        out = torch.zeros(tok, self.DIM, device=self.dev, dtype=torch.float32)
        for t in range(tok):
            for j in range(ids.shape[1]):
                e = int(ids[t, j])
                g_u = self.w1_deq[e].reshape(2 * self.INTER, self.DIM) @ x[t].float()
                g, u = g_u[: self.INTER], g_u[self.INTER :]
                h = (torch.nn.functional.silu(g) * u).to(torch.bfloat16).float()
                out[t] += float(wts[t, j]) * (
                    self.w2_deq[e].reshape(self.DIM, self.INTER) @ h
                )
        return out.to(torch.bfloat16)

    def _routing(self, tok):
        # distinct experts per token (as topk produces); the kernel's per-expert token list holds up to 64 entries,
        # which tok <= 40 with distinct slots can never exceed
        ids = (
            torch.stack(
                [
                    torch.randperm(self.E, device=self.dev)[: self.TOPK]
                    for _ in range(tok)
                ]
            )
            .to(torch.int32)
            .contiguous()
        )
        wts = torch.softmax(
            torch.randn(tok, self.TOPK, device=self.dev), -1
        ).contiguous()
        return ids, wts

    def test_matches_reference_and_falls_back_above_cap(self):
        M = self.M
        self.assertTrue(
            M.smallm_moe_enabled(), "gfx950 + ROCm>=7.2 + hipcc expected here"
        )
        os.environ["SGLANG_ROCM_SMALLM_MOE"] = "0"
        try:
            self.assertFalse(M.smallm_moe_enabled())
        finally:
            os.environ.pop("SGLANG_ROCM_SMALLM_MOE", None)
        for tok in (1, 3, 16, 40):
            x = torch.randn(tok, self.DIM, device=self.dev, dtype=torch.bfloat16)
            ids, wts = self._routing(tok)
            self.assertTrue(
                M.smallm_moe_supported(
                    x, self.w13, self.w2, ids, None, False, True, False, None
                )
            )
            out = M.smallm_moe_fwd(
                x, self.w13, self.w2, wts, ids, self.w13_scale, self.w2_scale
            )
            torch.cuda.synchronize()
            ref = self._reference(x, ids, wts)
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            self.assertLess(rel, 5e-3, f"tok={tok}: rel_l2={rel:.3e}")
            self.assertFalse(out.isnan().any().item())
        # above the dispatch cap the caller must stay on aiter's fused_moe
        x = torch.randn(64, self.DIM, device=self.dev, dtype=torch.bfloat16)
        ids, _ = self._routing(64)
        self.assertFalse(
            M.smallm_moe_supported(
                x, self.w13, self.w2, ids, None, False, True, False, None
            )
        )
        # the TP2 shape (per-rank intermediate 512) is not dispatched yet
        w13_512 = torch.empty(
            self.E, 1024, self.DIM // 2, device=self.dev, dtype=torch.uint8
        )
        w2_512 = torch.empty(self.E, self.DIM, 256, device=self.dev, dtype=torch.uint8)
        x = torch.randn(4, self.DIM, device=self.dev, dtype=torch.bfloat16)
        ids, _ = self._routing(4)
        self.assertFalse(
            M.smallm_moe_supported(
                x, w13_512, w2_512, ids, None, False, True, False, None
            )
        )

    def test_graph_replay_matches_eager(self):
        M = self.M
        tok = 4
        x = torch.randn(tok, self.DIM, device=self.dev, dtype=torch.bfloat16)
        ids, wts = self._routing(tok)
        eager = M.smallm_moe_fwd(
            x, self.w13, self.w2, wts, ids, self.w13_scale, self.w2_scale
        ).clone()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gout = M.smallm_moe_fwd(
                x, self.w13, self.w2, wts, ids, self.w13_scale, self.w2_scale
            )
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(gout, eager))


if __name__ == "__main__":
    unittest.main()
