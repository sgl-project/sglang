"""CI gating test for Work-Centric (Lean) Attention decode kernel.

Lean is an opt-in, gated decode-attention kernel in the ROCm/AMD Triton backend
(``python/sglang/kernels/ops/attention/decode_attention.py``). Its core contract
is that it is **numerically identical** to the standard SplitK grouped kernel —
the auto-gate only decides *when* to use it for speed, never *whether* the output
is correct. This test locks in that contract so a future change to the kernel or
its launch/reduction path cannot silently regress correctness.

Two things are checked:
  1. **Parity** — Lean output matches the standard SplitK kernel (cosine sim ~1.0)
     across representative GQA head shapes / batches / contexts.
  2. **Gate logic** — ``lean_decode_seqlen_gate`` enables Lean in the long-context
     / low-batch regime and keeps it off for short context, and stays off for MLA.

Correctness is triton-version-independent (only performance varies with the triton
build), so this makes a robust per-commit gate on MI35x hardware.
"""

import unittest

import torch

from sglang.kernels.ops.attention.decode_attention import (
    _LEAN_BLOCK_M,
    _lean_decode_launch_params,
    decode_attention_fwd,
    decode_attention_fwd_grouped,
    lean_decode_seqlen_gate,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# Lean lives in the ROCm/AMD Triton backend and is tuned for gfx950 (MI35x),
# so gate it on the per-commit MI35x single-GPU suite. Correctness (not perf) is
# what this test asserts, which holds regardless of the triton build.
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

# (name, H_Q, H_KV, head_dim) — the two GQA shapes validated in the PR benchmarks.
GQA_SHAPES = [
    ("qwen2.5-7b", 28, 4, 128),
    ("llama3.1-8b", 32, 8, 128),
]
MAX_KV_SPLITS = 8


def _run_pair(H_Q, H_KV, D, B, S, dev="cuda", dt=torch.float16, seed=0):
    """Run standard SplitK and Lean on the same inputs; return (o_std, o_lean)."""
    torch.manual_seed(seed)
    D_V = D
    kv_group_num = H_Q // H_KV
    sm = 1.0 / (D**0.5)
    tot = B * S

    total_programs, _, _ = _lean_decode_launch_params(H_KV, kv_group_num)
    lean_Mp = torch.empty(
        (total_programs, _LEAN_BLOCK_M), dtype=torch.float32, device=dev
    )
    lean_Lp = torch.empty(
        (total_programs, _LEAN_BLOCK_M), dtype=torch.float32, device=dev
    )
    lean_Op = torch.empty(
        (total_programs, _LEAN_BLOCK_M, D_V), dtype=torch.float32, device=dev
    )
    lean_locks = torch.zeros((total_programs,), dtype=torch.int32, device=dev)

    kv_indptr = torch.arange(0, (B + 1) * S, step=S, device=dev, dtype=torch.int32)
    kv_indices = torch.arange(0, tot, device=dev, dtype=torch.int32)
    q = torch.randn(B, H_Q, D, dtype=dt, device=dev)
    k = torch.randn(tot, H_KV, D, dtype=dt, device=dev)
    v = torch.randn(tot, H_KV, D_V, dtype=dt, device=dev)
    num_kv_splits = torch.full((B,), MAX_KV_SPLITS, dtype=torch.int32, device=dev)

    attn_logits = torch.empty(
        (B, H_Q, MAX_KV_SPLITS, D_V), dtype=torch.float32, device=dev
    )
    attn_lse = torch.empty((B, H_Q, MAX_KV_SPLITS), dtype=torch.float32, device=dev)
    o_std = torch.zeros(B, H_Q, D_V, dtype=dt, device=dev)
    decode_attention_fwd_grouped(
        q,
        k,
        v,
        o_std,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        MAX_KV_SPLITS,
        sm,
        1.0,
    )

    attn_logits2 = torch.empty_like(attn_logits)
    attn_lse2 = torch.empty_like(attn_lse)
    o_lean = torch.zeros(B, H_Q, D_V, dtype=dt, device=dev)
    decode_attention_fwd(
        q,
        k,
        v,
        o_lean,
        kv_indptr,
        kv_indices,
        attn_logits2,
        attn_lse2,
        num_kv_splits,
        MAX_KV_SPLITS,
        sm,
        1.0,
        1.0,
        enable_lean=True,
        lean_Mp=lean_Mp,
        lean_Lp=lean_Lp,
        lean_Op=lean_Op,
        lean_locks=lean_locks,
    )
    return o_std, o_lean


@unittest.skipUnless(torch.cuda.is_available(), "Lean decode kernel requires a GPU")
class TestLeanAttentionParity(CustomTestCase):
    """Lean must be numerically identical to the standard SplitK kernel."""

    def test_parity_across_gqa_shapes(self):
        # Contexts kept modest so CI stays fast; parity is context-independent.
        for name, H_Q, H_KV, D in GQA_SHAPES:
            for B in (1, 8):
                for S in (8192, 32768):
                    with self.subTest(model=name, batch=B, ctx=S):
                        o_std, o_lean = _run_pair(H_Q, H_KV, D, B, S)
                        cos = torch.nn.functional.cosine_similarity(
                            o_lean.flatten().float(), o_std.flatten().float(), dim=0
                        ).item()
                        self.assertGreater(
                            cos,
                            0.999,
                            f"{name} b={B} ctx={S}: Lean diverged from SplitK (cos={cos:.5f})",
                        )
                        # No NaN/Inf leaked from the persistent-grid reduction.
                        self.assertTrue(
                            torch.isfinite(o_lean).all(),
                            f"{name}: non-finite Lean output",
                        )


@unittest.skipUnless(
    torch.cuda.is_available(), "gate is exercised alongside the kernel path"
)
class TestLeanSeqlenGate(CustomTestCase):
    """The auto-gate must enable Lean in its win region and stay off elsewhere."""

    def test_gate_enables_long_context_low_batch(self):
        # Qwen GQA (28Q/4KV): long context at batch 1 is squarely Lean's win region.
        H_Q, kv_group = 28, 7
        self.assertTrue(
            lean_decode_seqlen_gate(
                H_Q, kv_group, batch=1, seq_lens_sum=131072, is_mla=False
            ),
            "gate should enable Lean for batch=1 @ 128K",
        )

    def test_gate_off_for_short_context(self):
        H_Q, kv_group = 28, 7
        self.assertFalse(
            lean_decode_seqlen_gate(
                H_Q, kv_group, batch=1, seq_lens_sum=2048, is_mla=False
            ),
            "gate should keep Lean off for batch=1 @ 2K (standard kernel wins)",
        )

    def test_gate_off_for_mla(self):
        # MLA's isolated win does not survive the MoE/TP decode step, so it is gated off.
        H_Q, kv_group = 128, 128
        self.assertFalse(
            lean_decode_seqlen_gate(
                H_Q, kv_group, batch=1, seq_lens_sum=131072, is_mla=True
            ),
            "gate must keep Lean off for MLA regardless of context",
        )

    def test_gate_threshold_falls_with_batch(self):
        # The crossover context falls as batch grows: a context that is below the
        # single-request threshold should still enable Lean at higher batch.
        H_Q, kv_group = 28, 7
        ctx = 32768
        low_batch = lean_decode_seqlen_gate(
            H_Q, kv_group, batch=1, seq_lens_sum=ctx, is_mla=False
        )
        high_batch = lean_decode_seqlen_gate(
            H_Q, kv_group, batch=8, seq_lens_sum=ctx * 8, is_mla=False
        )
        # At batch 8 the same per-request context should be at least as likely to enable Lean.
        self.assertTrue(
            high_batch or not low_batch,
            "gate batch relaxation is inconsistent (higher batch should not be stricter)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
