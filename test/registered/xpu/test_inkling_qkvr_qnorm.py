"""Inkling's in-place q RMSNorm on the packed QKVR row, on Intel XPU.

`q` is a strided slice of one `MergedColumnParallelLinear` output row
(`[q | k | v | r]`), so reshaping it to `(T*heads, head_dim)` for `apply_qk_norm`
materializes a compaction copy. The XPU path instead norms `q` in place with
`fused_qk_norm_rope`, which walks `num_heads_q + num_heads_k` heads but strides
rows by all three head counts -- so the KV+R tail is declared as untouched V
heads and never written.

That trick is only sound while the tail is a whole number of `head_dim` columns,
and only while rotary stays off. The cases below guard, in order:

  A. The padding arithmetic, and that a row whose tail is not whole heads is
     rejected rather than mis-sliced (`xpu_q_norm_geometry`).
  B. `q` matching `F.rms_norm` on the packed row, across both layer geometries
     at TP=8/TP=4 in bf16 and fp16.
  C. The KV+R tail staying byte-identical -- the invariant the V-head padding
     rests on.
  D. `q_norm.weight` surviving a call. The op's schema declares `q_weight`
     mutable (`Tensor($1! -> )`) and the live parameter is passed unwrapped when
     its dtype already matches the row, so nothing but this case would notice
     the kernel writing back through it.
  E. `rotary_dim=0` really disabling rotary rather than being inert at position
     0, with a `rotary_dim=head_dim` control that proves the case has power.
  F. The position-id cache reslicing correctly when the token count shrinks.

This file lives under `test/registered/xpu/` and registers XPU CI only, so no
per-class device skip is needed -- see the `register_xpu_ci` BKM. Do not add
`register_cuda_ci` here: CI collects by file, and the fused path is XPU-only.
"""

import unittest

import torch
import torch.nn.functional as F
from sgl_kernel import fused_qk_norm_rope

from sglang.srt.models.inkling_common.attn import (
    InklingAttention,
    xpu_q_norm_geometry,
)
from sglang.srt.models.inkling_common.norm import RMSNorm
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

DEVICE = "xpu"
HEAD_DIM = 128
D_REL = 16
EPS = 1e-6

# (num_tp_heads, num_tp_kv_heads) for Inkling's two layer groups. The SWA rows
# carry twice the KV heads of the full-attention rows.
ADMITTED = {
    "swa_tp8": (8, 2),
    "full_tp8": (8, 1),
    "swa_tp4": (16, 4),
    "full_tp4": (16, 2),
}
# TP=16/32 shrink d_rel * num_tp_heads below head_dim, so the tail stops being a
# whole number of heads and the gate has to refuse the row.
REJECTED = {"swa_tp16": (4, 1), "swa_tp32": (2, 1)}

TOKEN_COUNTS = (1, 2, 8, 512, 4096)
# 2 bf16 ULP near the output magnitude of a normed unit-variance row; the
# difference is a 1-ULP reciprocal-sqrt rounding, not a systematic error.
TOLERANCE = {
    torch.bfloat16: dict(rtol=8e-3, atol=1e-3),
    torch.float16: dict(rtol=1e-3, atol=1e-4),
}


class _Row:
    """Binds the real `_xpu_norm_q_inplace` to the attributes it reads.

    Instantiating `InklingAttention` needs a full model config and an
    initialized distributed group, so the method under test is bound onto a stub
    carrying only what it touches. The pad arithmetic comes from the production
    helper rather than a copy of it.
    """

    norm_q_inplace = InklingAttention._xpu_norm_q_inplace

    def __init__(self, num_tp_heads, num_tp_kv_heads, dtype=torch.bfloat16):
        self.num_tp_heads = num_tp_heads
        self.num_tp_kv_heads = num_tp_kv_heads
        self.d_rel = D_REL
        self.head_dim = HEAD_DIM
        self._xpu_q_norm_pad, self.geometry_ok = xpu_q_norm_geometry(
            head_dim=HEAD_DIM,
            num_tp_heads=num_tp_heads,
            num_tp_kv_heads=num_tp_kv_heads,
            d_rel=D_REL,
        )
        self._xpu_q_norm_pos = None
        self.q_width = HEAD_DIM * num_tp_heads
        self.width = (
            self.q_width + 2 * HEAD_DIM * num_tp_kv_heads + D_REL * num_tp_heads
        )
        # Same dtype as the row on purpose: `_xpu_norm_q_inplace` does
        # `weight.to(qkvr.dtype)`, which is a no-op here, so the kernel receives
        # the live parameter and case D is testing the risky path.
        self.q_norm = RMSNorm(HEAD_DIM, eps=EPS).to(DEVICE).to(dtype)
        generator = torch.Generator(device=DEVICE).manual_seed(11)
        with torch.no_grad():
            self.q_norm.weight.copy_(
                1.0
                + 0.05
                * torch.randn(
                    HEAD_DIM, generator=generator, device=DEVICE, dtype=torch.float32
                )
            )

    def packed_row(self, num_tokens, dtype, seed=0):
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        return torch.randn(
            num_tokens,
            self.width,
            generator=generator,
            device=DEVICE,
            dtype=dtype,
        )

    def reference_q(self, row):
        return F.rms_norm(
            row[:, : self.q_width].reshape(-1, HEAD_DIM),
            (HEAD_DIM,),
            self.q_norm.weight,
            EPS,
        )


class TestInklingQkvrQNorm(CustomTestCase):
    # --- A: the padding arithmetic and its negative branch --------------------

    def test_gate_admits_inkling_geometries_with_whole_head_padding(self):
        """Every shipped geometry must be admitted, with the pad covering the tail exactly.

        `num_heads_v=pad` is what hides the KV+R tail. If `pad * head_dim` were
        ever short of the real tail width, the kernel's row stride would not
        match the packed row and `q` would be normed off alignment.
        """
        for name, (num_q, num_kv) in ADMITTED.items():
            with self.subTest(name):
                row = _Row(num_q, num_kv)
                self.assertTrue(row.geometry_ok)
                tail_width = row.width - row.q_width
                self.assertEqual(row._xpu_q_norm_pad * HEAD_DIM, tail_width)

    def test_gate_rejects_rows_whose_tail_is_not_whole_heads(self):
        """TP=16/32 must be refused, not silently mis-sliced.

        At those degrees `d_rel * num_tp_heads` drops below `head_dim`, so the
        tail is a fraction of a V head and the trick is unsound. A predicate
        that degraded to always-true would leave no other case to catch it --
        the forward path would just take the fused branch and write past `q`.
        """
        for name, (num_q, num_kv) in REJECTED.items():
            with self.subTest(name):
                row = _Row(num_q, num_kv)
                self.assertFalse(row.geometry_ok)
                self.assertNotEqual(
                    row._xpu_q_norm_pad * HEAD_DIM, row.width - row.q_width
                )

    def test_rejected_geometry_is_also_caught_by_the_kernel(self):
        """A geometry mistake must fail loudly rather than mis-slice the row.

        The gate is the first line of defence, but the kernel's own
        `TORCH_CHECK(qkv.size(1) == total_heads * head_dim)` is the backstop, so
        forcing a rejected geometry through raises instead of corrupting `k`.
        """
        row = _Row(*REJECTED["swa_tp16"])
        packed = row.packed_row(8, torch.bfloat16)
        with self.assertRaises(RuntimeError):
            row.norm_q_inplace(packed)

    # --- B: numerics against F.rms_norm --------------------------------------

    def test_q_matches_rms_norm_on_the_packed_row(self):
        """The in-place path must agree with `F.rms_norm` on a compacted copy.

        This is the whole point of the change: the compaction copy is removed, so
        the only thing keeping `q` correct is that the kernel strides the packed
        row the way the pad arithmetic claims.
        """
        for dtype in (torch.bfloat16, torch.float16):
            for name, (num_q, num_kv) in ADMITTED.items():
                row = _Row(num_q, num_kv, dtype=dtype)
                for num_tokens in TOKEN_COUNTS:
                    with self.subTest(dtype=dtype, geometry=name, T=num_tokens):
                        packed = row.packed_row(num_tokens, dtype, seed=num_tokens)
                        expected = row.reference_q(packed.clone())
                        row.norm_q_inplace(packed)
                        torch.testing.assert_close(
                            packed[:, : row.q_width].reshape(-1, HEAD_DIM).float(),
                            expected.float(),
                            **TOLERANCE[dtype],
                        )

    # --- C: the invariant the V-head padding rests on ------------------------

    def test_kv_and_r_tail_is_byte_identical(self):
        """`k`, `v` and `r` must come back untouched, bit for bit.

        The tail is passed off as V heads, which the kernel is documented to
        skip. Nothing downstream would notice a stray write until attention
        scores went subtly wrong, so this is asserted with `torch.equal` rather
        than a tolerance.
        """
        for dtype in (torch.bfloat16, torch.float16):
            for name, (num_q, num_kv) in ADMITTED.items():
                row = _Row(num_q, num_kv, dtype=dtype)
                for num_tokens in TOKEN_COUNTS:
                    with self.subTest(dtype=dtype, geometry=name, T=num_tokens):
                        packed = row.packed_row(num_tokens, dtype, seed=num_tokens)
                        original_tail = packed[:, row.q_width :].clone()
                        row.norm_q_inplace(packed)
                        self.assertTrue(
                            torch.equal(packed[:, row.q_width :], original_tail)
                        )

    # --- D: the mutable-schema hazard ----------------------------------------

    def test_q_norm_weight_is_not_mutated(self):
        """The live `q_norm.weight` must survive, despite a mutable op schema.

        `fused_qk_norm_rope` declares `q_weight` as `Tensor($1! -> )`, and the
        row's dtype matches the parameter's here, so `weight.to(dtype)` hands
        the kernel the parameter itself instead of a copy. A write-back would
        corrupt the model's weights in place, degrading generations with no
        error anywhere.
        """
        for dtype in (torch.bfloat16, torch.float16):
            row = _Row(*ADMITTED["swa_tp8"], dtype=dtype)
            weight_before = row.q_norm.weight.clone()
            for num_tokens in TOKEN_COUNTS:
                with self.subTest(dtype=dtype, T=num_tokens):
                    row.norm_q_inplace(row.packed_row(num_tokens, dtype))
                    self.assertTrue(torch.equal(row.q_norm.weight, weight_before))

    # --- E: rotary is off, and the case can tell ------------------------------

    def test_rotary_is_disabled_rather_than_inert_at_position_zero(self):
        """`rotary_dim=0` must ignore position ids outright.

        Inkling has no RoPE, and the path relies on `rotary_dim=0` to skip every
        lane's rotary branch. Feeding zeros would prove nothing on its own --
        rotary at position 0 is the identity -- so the same row is normed with
        ascending position ids and must come out unchanged.
        """
        num_tokens = 8
        row = _Row(*ADMITTED["swa_tp8"])
        packed_zero_pos = row.packed_row(num_tokens, torch.bfloat16, seed=5)
        packed_real_pos = packed_zero_pos.clone()

        row.norm_q_inplace(packed_zero_pos)

        ascending = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        row._xpu_q_norm_pos = ascending
        row.norm_q_inplace(packed_real_pos)
        self.assertTrue(torch.equal(packed_real_pos, packed_zero_pos))

    def test_rotary_control_shows_the_position_ids_would_otherwise_bite(self):
        """Control for the case above: `rotary_dim=head_dim` must change the row.

        Without this, a kernel that silently ignored `position_ids` altogether
        would pass the previous case for the wrong reason.
        """
        num_tokens = 8
        row = _Row(*ADMITTED["swa_tp8"])
        packed_no_rotary = row.packed_row(num_tokens, torch.bfloat16, seed=5)
        packed_with_rotary = packed_no_rotary.clone()
        ascending = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        row._xpu_q_norm_pos = ascending
        row.norm_q_inplace(packed_no_rotary)

        weight = row.q_norm.weight
        fused_qk_norm_rope(
            packed_with_rotary,
            num_heads_q=row.num_tp_heads,
            num_heads_k=0,
            num_heads_v=row._xpu_q_norm_pad,
            head_dim=HEAD_DIM,
            eps=EPS,
            q_weight=weight,
            k_weight=weight,
            base=1.0e4,
            is_neox=False,
            position_ids=ascending,
            rotary_dim=HEAD_DIM,
        )
        self.assertFalse(torch.equal(packed_with_rotary, packed_no_rotary))

    # --- F: the position-id cache --------------------------------------------

    def test_position_cache_reslices_when_the_token_count_shrinks(self):
        """The cached position buffer must be resliced, not reused whole.

        It is grown on demand and kept across calls, so a shrinking batch has to
        be handed `pos[:num_tokens]`. Passing the full buffer would trip the
        kernel's `position_ids.numel() == num_tokens` contract on the first
        decode step after a long prefill.
        """
        row = _Row(*ADMITTED["swa_tp8"])
        for num_tokens in (512, 8, 4096, 1):
            with self.subTest(T=num_tokens):
                row.norm_q_inplace(row.packed_row(num_tokens, torch.bfloat16))
                self.assertGreaterEqual(row._xpu_q_norm_pos.numel(), num_tokens)


if __name__ == "__main__":
    unittest.main()
