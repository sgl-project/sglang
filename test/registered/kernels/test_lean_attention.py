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
  2. **Gate logic** — the eager ``lean_decode_seqlen_gate`` enables Lean in the
     long-context / low-batch regime and keeps it off for short context, and the
     CUDA-graph ``lean_capture_policy`` bakes Lean from capture-time signals (batch,
     head-tiles, is_mla) since captured seq_lens are the fill value. MLA is gated on
     batch (off at b1, on at b>=8), not blanket-off.

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
    lean_capture_policy,
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


def _lean_scratch(H_KV, kv_group_num, D_V, dev):
    total_programs, _, _ = _lean_decode_launch_params(H_KV, kv_group_num)
    return (
        torch.empty((total_programs, _LEAN_BLOCK_M), dtype=torch.float32, device=dev),
        torch.empty((total_programs, _LEAN_BLOCK_M), dtype=torch.float32, device=dev),
        torch.empty(
            (total_programs, _LEAN_BLOCK_M, D_V), dtype=torch.float32, device=dev
        ),
        torch.zeros((total_programs,), dtype=torch.int32, device=dev),
    )


def _run_pair_fp8(H_Q, H_KV, D, B, S, fp8_dtype, dev="cuda", seed=0):
    """Standard vs Lean on **fp8** K/V with non-unit k_scale/v_scale.

    Both arms go through the public ``decode_attention_fwd`` dispatch (enable_lean False/True),
    which folds k_scale into sm_scale and applies v_scale — the exact production path. They share
    the same fp8 inputs and dequant scales, so their outputs must agree (the fp8 quantization
    error is identical for both); this guards that Lean's fp8 dtype handling matches the standard
    kernel. Returns (o_std, o_lean).
    """
    torch.manual_seed(seed)
    D_V = D
    kv_group_num = H_Q // H_KV
    sm = 1.0 / (D**0.5)
    tot = B * S
    fp8_max = torch.finfo(fp8_dtype).max

    q = torch.randn(B, H_Q, D, dtype=torch.float16, device=dev)
    k_ref = torch.randn(tot, H_KV, D, dtype=torch.float32, device=dev)
    v_ref = torch.randn(tot, H_KV, D_V, dtype=torch.float32, device=dev)
    # Per-tensor symmetric quantization to fp8, mirroring how fp8 KV is stored + dequantized.
    k_scale = (k_ref.abs().max() / fp8_max).item()
    v_scale = (v_ref.abs().max() / fp8_max).item()
    k = (k_ref / k_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    v = (v_ref / v_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)

    kv_indptr = torch.arange(0, (B + 1) * S, step=S, device=dev, dtype=torch.int32)
    kv_indices = torch.arange(0, tot, device=dev, dtype=torch.int32)
    num_kv_splits = torch.full((B,), MAX_KV_SPLITS, dtype=torch.int32, device=dev)

    def _call(enable_lean):
        attn_logits = torch.empty(
            (B, H_Q, MAX_KV_SPLITS, D_V), dtype=torch.float32, device=dev
        )
        attn_lse = torch.empty((B, H_Q, MAX_KV_SPLITS), dtype=torch.float32, device=dev)
        o = torch.zeros(B, H_Q, D_V, dtype=torch.float16, device=dev)
        mp, lp, op, locks = _lean_scratch(H_KV, kv_group_num, D_V, dev)
        decode_attention_fwd(
            q,
            k,
            v,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            MAX_KV_SPLITS,
            sm,
            k_scale,
            v_scale,
            enable_lean=enable_lean,
            lean_Mp=mp,
            lean_Lp=lp,
            lean_Op=op,
            lean_locks=locks,
        )
        return o

    return _call(False), _call(True)


def _run_pair_paged(
    H_Q, H_KV, D, B, S, page_size, dev="cuda", dt=torch.float16, seed=0
):
    """Standard vs Lean on a **paged** 4-D KV buffer ``[num_pages, page_size, head, dim]``.

    The KV cache is stored in pages and addressed through scattered slot ids in ``kv_indices``
    (a permutation), so the kernel's page-aware address math (``kv_loc // page_size`` /
    ``kv_loc % page_size``) is genuinely exercised — not the contiguous fast path. Both arms read
    the identical buffer + indices, so their outputs must agree. Returns (o_std, o_lean).
    """
    torch.manual_seed(seed)
    D_V = D
    kv_group_num = H_Q // H_KV
    sm = 1.0 / (D**0.5)
    tot = B * S
    assert tot % page_size == 0, (
        "test setup: total tokens must be a multiple of page_size"
    )
    num_pages = tot // page_size

    # 4-D paged KV buffers [num_pages, page_size, head, dim] (the shared-pool layout).
    k = torch.randn(num_pages, page_size, H_KV, D, dtype=dt, device=dev)
    v = torch.randn(num_pages, page_size, H_KV, D_V, dtype=dt, device=dev)

    kv_indptr = torch.arange(0, (B + 1) * S, step=S, device=dev, dtype=torch.int32)
    # Scatter slots across pages so page_id/tok_in_p vary within every BLOCK_N tile.
    kv_indices = torch.randperm(tot, device=dev).to(torch.int32)
    q = torch.randn(B, H_Q, D, dtype=dt, device=dev)
    num_kv_splits = torch.full((B,), MAX_KV_SPLITS, dtype=torch.int32, device=dev)

    def _call(enable_lean):
        attn_logits = torch.empty(
            (B, H_Q, MAX_KV_SPLITS, D_V), dtype=torch.float32, device=dev
        )
        attn_lse = torch.empty((B, H_Q, MAX_KV_SPLITS), dtype=torch.float32, device=dev)
        o = torch.zeros(B, H_Q, D_V, dtype=dt, device=dev)
        mp, lp, op, locks = _lean_scratch(H_KV, kv_group_num, D_V, dev)
        decode_attention_fwd(
            q,
            k,
            v,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            MAX_KV_SPLITS,
            sm,
            1.0,
            1.0,
            page_size=page_size,
            enable_lean=enable_lean,
            lean_Mp=mp,
            lean_Lp=lp,
            lean_Op=op,
            lean_locks=locks,
        )
        return o

    return _call(False), _call(True)


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

    def test_fp8_kv_parity(self):
        # Phase 2: Lean must handle fp8 KV cache the same way the standard kernel does
        # (cast q->K.dtype for the MMA, fold k_scale into sm_scale, apply v_scale). Guards the
        # regression where the Lean path crashed on fp8 K ("Unsupported rhs dtype fp8e4nv").
        fp8_dtype = None
        for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
            if hasattr(torch, name):
                fp8_dtype = getattr(torch, name)
                break
        if fp8_dtype is None:
            self.skipTest("no fp8 e4m3 dtype available in this torch build")
        for name, H_Q, H_KV, D in GQA_SHAPES:
            for B in (1, 8):
                for S in (8192, 32768):
                    with self.subTest(model=name, batch=B, ctx=S, dtype=str(fp8_dtype)):
                        o_std, o_lean = _run_pair_fp8(H_Q, H_KV, D, B, S, fp8_dtype)
                        self.assertTrue(
                            torch.isfinite(o_lean).all(),
                            f"{name}: non-finite Lean fp8 output",
                        )
                        cos = torch.nn.functional.cosine_similarity(
                            o_lean.flatten().float(), o_std.flatten().float(), dim=0
                        ).item()
                        self.assertGreater(
                            cos,
                            0.99,
                            f"{name} b={B} ctx={S}: Lean fp8 diverged from SplitK (cos={cos:.5f})",
                        )

    def test_paged_kv_parity(self):
        # Lean must read a paged 4-D KV buffer the same way the standard kernel does. Guards
        # the page-aware address math (kv_loc // page_size, kv_loc % page_size); a regression
        # to the contiguous-only form would scramble reads and drop cos well below 1.
        for name, H_Q, H_KV, D in GQA_SHAPES:
            for page_size in (16, 64):
                with self.subTest(model=name, page_size=page_size):
                    o_std, o_lean = _run_pair_paged(
                        H_Q, H_KV, D, B=2, S=8192, page_size=page_size
                    )
                    self.assertTrue(
                        torch.isfinite(o_lean).all(),
                        f"{name} ps={page_size}: non-finite Lean paged output",
                    )
                    cos = torch.nn.functional.cosine_similarity(
                        o_lean.flatten().float(), o_std.flatten().float(), dim=0
                    ).item()
                    self.assertGreater(
                        cos,
                        0.999,
                        f"{name} ps={page_size}: Lean paged diverged from SplitK (cos={cos:.5f})",
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

    def test_gate_mla_batch_threshold(self):
        # MLA is gated on batch, not a blanket off: b1 loses hard (CU-saturated), b>=8
        # is parity/ragged-win. The eager gate must reflect that boundary. (Guards against
        # both a regression to the old blanket-off and to an always-on for MLA.)
        H_Q, kv_group = 128, 128
        self.assertFalse(
            lean_decode_seqlen_gate(
                H_Q, kv_group, batch=1, seq_lens_sum=131072, is_mla=True
            ),
            "MLA at batch=1 is a catastrophic loss; gate must stay off",
        )
        self.assertTrue(
            lean_decode_seqlen_gate(
                H_Q, kv_group, batch=8, seq_lens_sum=8 * 65536, is_mla=True
            ),
            "MLA at batch>=8 with long context is a win; gate must enable",
        )

    def test_gate_off_when_seq_lens_sum_missing(self):
        # The EAGLE draft runner (and gpu-only batches) call decode without a CPU length
        # mirror, so seq_lens_sum is None. The gate must fall back to the standard kernel
        # instead of dividing None by batch (which raised TypeError and crashed the
        # scheduler under EAGLE3 speculative decoding).
        H_Q, kv_group = 28, 7
        self.assertFalse(
            lean_decode_seqlen_gate(
                H_Q, kv_group, batch=8, seq_lens_sum=None, is_mla=False
            ),
            "gate must return False (not raise) when seq_lens_sum is None",
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


class TestLeanCapturePolicy(CustomTestCase):
    """The CUDA-graph capture-time bake policy keys on (tiles, is_mla, batch) only —
    captured seq_lens are the fill value, so it cannot use context. These pin the
    calibrated thresholds (CALIBRATION.md): a threshold drift or a degraded predicate
    (always-on / always-off) turns the corresponding case red."""

    def test_gqa_bakes_at_and_above_batch_16(self):
        # Qwen GQA (28Q/4KV -> tiles=4): unconditional-win boundary is batch>=16.
        H_Q, kv_group = 28, 7
        self.assertFalse(
            lean_capture_policy(H_Q, kv_group, batch=8, is_mla=False),
            "GQA capture must not bake at batch=8 (context-split / uniform-short regresses)",
        )
        self.assertTrue(
            lean_capture_policy(H_Q, kv_group, batch=16, is_mla=False),
            "GQA capture must bake at batch>=16 (unconditional win)",
        )

    def test_mla_bakes_at_and_above_batch_8_never_at_low_batch(self):
        # MLA (128Q/128KV): b1 is a catastrophic loss, b>=8 is parity/ragged-win.
        H_Q, kv_group = 128, 128
        self.assertFalse(
            lean_capture_policy(H_Q, kv_group, batch=1, is_mla=True),
            "MLA capture must never bake at batch=1 (0.4-0.55x loss)",
        )
        self.assertTrue(
            lean_capture_policy(H_Q, kv_group, batch=8, is_mla=True),
            "MLA capture must bake at batch>=8",
        )

    def test_heavy_tp_shard_never_bakes(self):
        # tiles<4 (e.g. Llama-70B @TP=8: 8 query heads/GPU, kv_group=1 -> tiles=8?) —
        # use a genuine heavy shard: 2 query heads, kv_group=1 -> tiles=2 (<4). Known ~4x
        # regression at 32K, not calibrated for capture -> never bake even at high batch.
        self.assertFalse(
            lean_capture_policy(2, 1, batch=32, is_mla=False),
            "heavy TP shard (tiles<4) must never bake under capture",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
