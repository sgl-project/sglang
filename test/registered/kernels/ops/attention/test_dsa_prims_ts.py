"""prims_ts DSA backend: planned FlashInfer Prims-TS sparse MLA launches over raw FP8
MLA rows (E4M3 q, natural-log LSE) against an FP32 reference, and the DCP rejection
that the natural-log LSE requires."""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.attention.dsa.prims_ts import (
    _EXACT_PLAN_MAX_ROWS,
    PrimsTSSparseMLA,
    check_prims_ts_sparse_mla_support,
)
from sglang.srt.layers.attention.dsa_backend import _validate_prims_ts_dcp
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    is_mla_dcp_lse_base_on_e,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

FP8 = torch.float8_e4m3fn
KV_LORA, ROPE, QK_NOPE, TOPK = 512, 64, 128, 2048
DIM = KV_LORA + ROPE
SM_SCALE = (QK_NOPE + ROPE) ** -0.5


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


class TestPrimsTsSparseMla(CustomTestCase):
    def _require_prims_ts(self, heads: int):
        if not _sm100():
            self.skipTest("Prims-TS sparse MLA needs SM100/SM103/SM107")
        try:
            check_prims_ts_sparse_mla_support(
                device=torch.device("cuda"),
                num_heads=heads,
                kv_lora_rank=KV_LORA,
                qk_rope_head_dim=ROPE,
            )
        except ValueError as exc:
            self.skipTest(str(exc))
        except RuntimeError as exc:
            # FlashInfer 0.7.2 lacks the D576 native-FP8 kernel.
            self.skipTest(f"Skipped until the FlashInfer bump: {exc}")

    def _check_matches_reference(self, rows: int):
        """Each row attends over its own top-k cache rows (-1 past its count); the
        output and the natural-log LSE match an FP32 reference over the E4M3 values."""
        heads, ctx = 96, 16384
        self._require_prims_ts(heads)
        gen = torch.Generator(device="cuda").manual_seed(3)
        kv = (torch.randn(ctx, DIM, generator=gen, device="cuda") * 0.5).to(FP8)
        q8 = torch.randn(rows, heads, DIM, generator=gen, device="cuda").to(FP8)
        slots = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
        counts = torch.randint(1, TOPK + 1, (rows,), generator=gen, device="cuda")
        counts[1] = TOPK
        for r in range(rows):
            n = int(counts[r])
            pick = torch.randperm(ctx, generator=gen, device="cuda")[:n]
            slots[r, :n] = pick.to(torch.int32)
        counts = counts.to(torch.int32)

        runner = PrimsTSSparseMLA(torch.device("cuda"))
        out, lse = runner.forward(
            q=q8,
            kv_cache=kv,
            indices=slots,
            lengths=counts,
            sm_scale=SM_SCALE,
            kv_scale=1.0,
        )
        # (rows, heads, width, packed) plan keys: rows past the exact limit pack
        packed = rows > _EXACT_PLAN_MAX_ROWS
        self.assertEqual({key[3] for key in runner._plans}, {packed})

        qf, kvf = q8.float(), kv.float()
        ref = torch.empty_like(out, dtype=torch.float32)
        ref_lse = torch.empty_like(lse)
        for r in range(rows):
            k = kvf[slots[r, : counts[r]].long()]
            s = (qf[r] @ k.T) * SM_SCALE
            ref_lse[r] = torch.logsumexp(s, dim=-1)
            ref[r] = torch.softmax(s, dim=-1) @ k[:, :KV_LORA]

        # The error comes from rounding the softmax probabilities to E4M3.
        self.assertLess(((out.float() - ref).norm() / ref.norm()).item(), 0.03)
        torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=2e-3)

    def test_exact_plan_matches_reference(self):
        self._check_matches_reference(rows=24)

    def test_packed_plan_matches_reference(self):
        # Past the exact-plan limit: power-of-two packed capacity with qo_indptr.
        self._check_matches_reference(rows=_EXACT_PLAN_MAX_ROWS + 1)


class TestPrimsTsDcpLseBase(CustomTestCase):
    def test_dsa_dcp_rejects_prims_ts(self):
        """Prims-TS returns a natural-log LSE while the DSA DCP combine merges base 2,
        so prims_ts must be rejected under DCP until the combine merges it base e."""
        self.assertFalse(is_mla_dcp_lse_base_on_e("dsa"))
        with self.assertRaisesRegex(ValueError, "natural-log LSE"):
            _validate_prims_ts_dcp(dcp_size=2)
        # Without DCP there is no merge.
        _validate_prims_ts_dcp(dcp_size=1)


if __name__ == "__main__":
    unittest.main()
