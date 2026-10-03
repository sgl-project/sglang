"""MXF4 (e2m1, group 32) activation quantization kernel tests.

Covers python/sglang/kernels/ops/quantization/mxfp4_group_quant.py:
  - quant_mxfp4_group32 (comparison-chain e2m1 coding)
  - quant_mxfp4_group32_v2 (SM100 hardware cvt based coding)
  - silu_mul_quant_mxfp4 (fused SiLU-mul + quant, with/without swiglu clamp)
  - silu_mul_quant_mxfp4_masked (the masked / padded twin of the above)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from sglang.kernels.ops.quantization.mxfp4_group_quant import (
    quant_mxfp4_group32,
    quant_mxfp4_group32_v2,
    silu_mul_quant_mxfp4,
    silu_mul_quant_mxfp4_masked,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _ref_quant(x: torch.Tensor):
    """Torch reference of the comparison-chain kernel: returns the per-element
    e2m1 nibble codes (M, K) and the per-group ue8m0 exponents (M, K // 32)."""
    M, K = x.shape
    xg = x.view(M, K // 32, 32)
    amax = xg.abs().amax(dim=-1)
    sf = torch.maximum(amax / 6.0, torch.full_like(amax, 1.0e-4))
    bits = sf.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    exp = exp + (bits & 0x7FFFFF != 0).to(torch.int32)
    exp = exp.clamp(1, 254)

    scale = (exp << 23).view(torch.float32)
    xn = (xg / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    ax = xn.abs()
    idx = (
        (ax > 0.25).to(torch.uint8)
        + (ax > 0.75).to(torch.uint8)
        + (ax > 1.25).to(torch.uint8)
        + (ax > 1.75).to(torch.uint8)
        + (ax > 2.5).to(torch.uint8)
        + (ax > 3.5).to(torch.uint8)
        + (ax > 5.0).to(torch.uint8)
    )
    sign = ((xn < 0) & (idx != 0)).to(torch.uint8) << 3
    code = (idx | sign).view(M, K)
    return code, exp


def _ref_packed_words(exp: torch.Tensor):
    """Pack reference exponents into int32 ue8m0 words the kernel's way."""
    M = exp.shape[0]
    sh = torch.arange(4, device=exp.device, dtype=torch.int32) * 8
    words = (exp.view(M, -1, 4) << sh).sum(dim=-1)
    return words.to(torch.int32)


def _dequant(q: torch.Tensor, sf_words: torch.Tensor):
    """Dequantize packed e2m1 + packed ue8m0 back to float, groupwise."""
    M, Kh = q.shape
    K = Kh * 2
    codes = torch.stack((q & 0xF, (q >> 4) & 0xF), dim=-1).view(M, K)
    sign = 1 - 2 * (codes >> 3)
    # long(), not uint8: torch now treats a uint8 index tensor as a boolean
    # mask, not an integer index (see the indexing-dtype deprecation notice).
    mag = (codes & 7).long()
    vals = torch.tensor(_E2M1_VALUES, device=q.device, dtype=torch.float32)[mag]
    # sf_words is (M, K // 128): each int32 word packs 4 exponent *bytes*
    # (one per group of 32), so unpacking is a bit-shift, not a reshape —
    # sf_words.view(M, -1, 4) silently reinterprets 4 adjacent *words* as
    # if they were the 4 bytes of one, which only worked by accident when
    # K // 128 was itself a multiple of 4.
    sh = torch.arange(4, device=sf_words.device, dtype=torch.int32) * 8
    exp = (sf_words.unsqueeze(-1) >> sh) & 0xFF  # (M, K // 128, 4) -> flatten
    scale = torch.pow(2.0, exp.float() - 127.0).reshape(M, K // 32)  # (M, NG)
    scale_b = scale.repeat_interleave(32, dim=-1)  # (M, K), group-major order
    return sign * vals * scale_b


def _group_scale(sf_words: torch.Tensor):
    M = sf_words.shape[0]
    sh = torch.arange(4, device=sf_words.device, dtype=torch.int32) * 8
    exp = (sf_words.unsqueeze(-1) >> sh) & 0xFF
    NG = exp.shape[1] * exp.shape[2]
    scale = torch.pow(2.0, exp.float() - 127.0).reshape(M, NG)
    return scale.repeat_interleave(32, dim=-1)


def _make_x(M, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((M, K), generator=g, device="cuda", dtype=torch.float32)
    # Mix in zeros / small values so the 1e-4 scale floor is exercised.
    x[:, :32] = 0.0
    x[:, 32:64] = 1e-6
    return x.to(torch.bfloat16)


class TestQuantMxfp4Group32(CustomTestCase):
    def test_matches_reference_bit_exact(self):
        # K=1152/1280 exercise masked tail blocks (BLOCK_K=1024, K not a
        # multiple of BLOCK_K) — the stores must stay in bounds.
        for M, K in ((7, 128), (3, 256), (4, 1024), (5, 2048), (5, 1152), (3, 1280)):
            with self.subTest(M=M, K=K):
                x = _make_x(M, K, seed=K + M)
                q, sf = quant_mxfp4_group32(x)

                self.assertEqual(q.shape, (M, K // 2))
                self.assertEqual(q.dtype, torch.int8)
                self.assertEqual(sf.shape, (M, K // 128))
                self.assertEqual(sf.dtype, torch.int32)

                ref_code, ref_exp = _ref_quant(x.float())
                lo, hi = ref_code[:, 0::2], ref_code[:, 1::2]
                ref_q = ((lo & 0xF) | ((hi & 0xF) << 4)).to(torch.int8)
                self.assertTrue(torch.equal(q, ref_q))
                self.assertTrue(torch.equal(sf, _ref_packed_words(ref_exp)))

    def test_dequant_error_bounded(self):
        x = _make_x(9, 512, seed=1)
        q, sf = quant_mxfp4_group32(x)
        err = (_dequant(q, sf) - x.float()).abs()
        bound = _group_scale(sf)
        self.assertLessEqual(err.max().item(), 1.01 * bound.max().item())

    def test_k_must_be_multiple_of_128(self):
        x = torch.zeros((2, 100), device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            quant_mxfp4_group32(x)


@unittest.skipIf(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
    "v2 uses the SM100 cvt.rn.satfinite.e2m1x2.f32 instruction",
)
class TestQuantMxfp4Group32V2(CustomTestCase):
    def test_dequant_error_bounded(self):
        # K=2176 exercises a masked scale-word tail block (BLOCK_K=2048).
        for M, K in ((7, 128), (3, 512), (4, 2048), (3, 2176)):
            with self.subTest(M=M, K=K):
                x = _make_x(M, K, seed=K)
                q, sf = quant_mxfp4_group32_v2(x)

                self.assertEqual(q.shape, (M, K // 2))
                self.assertEqual(q.dtype, torch.int8)
                self.assertEqual(sf.shape, (M, K // 128))
                self.assertEqual(sf.dtype, torch.int32)

                err = (_dequant(q, sf) - x.float()).abs()
                bound = _group_scale(sf)
                self.assertLessEqual(err.max().item(), 1.01 * bound.max().item())

    def test_scale_words_match_reference(self):
        # Scales are identical to the comparison-chain reference; only the
        # mantissa coding (ties / zero sign) may differ.
        x = _make_x(6, 256, seed=2)
        _, sf = quant_mxfp4_group32_v2(x)
        _, ref_exp = _ref_quant(x.float())
        self.assertTrue(torch.equal(sf, _ref_packed_words(ref_exp)))

    def test_agrees_with_v1(self):
        # The hardware cvt and the comparison chain differ only at exact
        # rounding ties and on the sign of zero: nibble magnitudes may differ
        # by at most one code index, signs must match. K=2176 spans a v2
        # BLOCK_K=2048 program plus a 128-element tail, so the pipelined
        # multi-iteration and masked-tail paths must also agree with v1.
        for M, K in ((8, 256), (5, 2176), (1, 128)):
            with self.subTest(M=M, K=K):
                x = _make_x(M, K, seed=3 + K)
                q1, sf1 = quant_mxfp4_group32(x)
                q2, sf2 = quant_mxfp4_group32_v2(x)
                self.assertTrue(torch.equal(sf1, sf2))
                codes1 = torch.stack((q1 & 0xF, (q1 >> 4) & 0xF), dim=-1).view(-1).int()
                codes2 = torch.stack((q2 & 0xF, (q2 >> 4) & 0xF), dim=-1).view(-1).int()
                mag1, mag2 = codes1 & 7, codes2 & 7
                self.assertLessEqual((mag1 - mag2).abs().max().item(), 1)
                # Signs may differ only on zero (magnitude 0), which decodes as 0.
                sign_mismatch = (
                    ((codes1 >> 3) != (codes2 >> 3)) & (mag1 != 0) & (mag2 != 0)
                )
                self.assertFalse(sign_mismatch.any().item())

    def test_single_row_minimum_shape(self):
        # M=1 is the smallest legal shape; the m_al=4 scale storage then
        # carries 3 padding columns that must stay zero (NaN if garbage).
        x = _make_x(1, 256, seed=7)
        q, sf = quant_mxfp4_group32_v2(x)
        self.assertEqual(q.shape, (1, 128))
        self.assertEqual(sf.shape, (1, 2))
        err = (_dequant(q, sf) - x.float()).abs()
        bound = _group_scale(sf)
        self.assertLessEqual(err.max().item(), 1.01 * bound.max().item())


@unittest.skipIf(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
    "fused kernel uses the SM100 cvt.rn.satfinite.e2m1x2.f32 instruction",
)
class TestSiluMulQuantMxfp4(CustomTestCase):
    def _run(self, T, H, limit):
        g = torch.Generator(device="cuda").manual_seed(H + T)
        gateup = torch.randn(
            (T, 2 * H), generator=g, device="cuda", dtype=torch.float32
        )
        # Push some values past the clamp so the limit path is exercised.
        gateup[:, 0] = 50.0
        gateup[:, H] = -50.0
        gateup = gateup.to(torch.bfloat16)

        q, sf = silu_mul_quant_mxfp4(gateup, limit)
        self.assertEqual(q.shape, (T, H // 2))
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(sf.shape, (T, H // 128))
        self.assertEqual(sf.dtype, torch.int32)

        gate = gateup[:, :H].float()
        up = gateup[:, H:].float()
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        act = gate * torch.sigmoid(gate) * up

        err = (_dequant(q, sf) - act).abs()
        bound = _group_scale(sf)
        self.assertLessEqual(err.max().item(), 1.01 * bound.max().item())

    def test_without_clamp(self):
        self._run(5, 256, None)

    def test_with_swiglu_limit(self):
        self._run(5, 256, 10.0)

    def test_with_fractional_swiglu_limit(self):
        # A non-integral limit must be honoured as-is, not truncated to 10.
        self._run(5, 256, 10.5)

    def test_masked_tail_block_with_limit(self):
        # H=2176 -> BLOCK_H=2048 with a 128-element tail program.
        self._run(5, 2176, 10.0)

    def test_h_must_be_multiple_of_128(self):
        gateup = torch.zeros((2, 200), device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            silu_mul_quant_mxfp4(gateup)


@unittest.skipIf(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
    "masked kernel uses the SM100 cvt.rn.satfinite.e2m1x2.f32 instruction",
)
class TestSiluMulQuantMxfp4Masked(CustomTestCase):
    """The masked (EP-MoE) twin of `silu_mul_quant_mxfp4`.

    `masked_m` declares how many of the padded rows of each expert are live, so
    the same expert rows must quantize exactly as the contiguous kernel would.
    """

    E, M_MAX, H = 4, 256, 512

    def _gateup(self, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        gateup = torch.randn(
            (self.E, self.M_MAX, 2 * self.H),
            generator=g,
            device="cuda",
            dtype=torch.float32,
        )
        # Push some values past the clamp so the limit path is exercised.
        gateup[:, :, 0] = 50.0
        gateup[:, :, self.H] = -50.0
        return gateup.to(torch.bfloat16)

    @staticmethod
    def _masked_m(counts):
        return torch.tensor(counts, device="cuda", dtype=torch.int32)

    def _assert_matches_contiguous(self, gateup, masked_m, q, sf, limit):
        for e, mm in enumerate(masked_m.tolist()):
            with self.subTest(expert=e):
                ref_q, ref_sf = silu_mul_quant_mxfp4(gateup[e, :mm], limit)
                self.assertTrue(torch.equal(q[e, :mm], ref_q))
                self.assertTrue(torch.equal(sf[e, :mm], ref_sf))

    def _run(self, counts, topk, num_real_tokens, limit=None, seed=0):
        gateup = self._gateup(seed)
        masked_m = self._masked_m(counts)
        q, sf = silu_mul_quant_mxfp4_masked(
            gateup, masked_m, topk, num_real_tokens, limit
        )
        self.assertEqual(q.shape, (self.E, self.M_MAX, self.H // 2))
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(sf.shape, (self.E, self.M_MAX, self.H // 128))
        self.assertEqual(sf.dtype, torch.int32)
        return gateup, masked_m, q, sf

    def test_matches_the_contiguous_kernel_per_expert(self):
        for limit in (None, 10.0, 10.5):
            with self.subTest(limit=limit):
                # sum(counts) == 14 <= 6 * 3, so every grid program finds work.
                gateup, masked_m, q, sf = self._run([3, 4, 5, 2], 3, 6, limit)
                self._assert_matches_contiguous(gateup, masked_m, q, sf, limit)

    def test_surplus_grid_programs_are_inert(self):
        # num_real_tokens * topk only bounds masked_m.sum() from above, and it
        # can exceed m_max: the surplus programs must exit before addressing a
        # row, so a grid grown past m_max cannot perturb any live row. Rows at
        # or beyond masked_m are left uninitialized, hence the per-expert slice.
        counts = [100, 50, 30, 20]
        gateup, masked_m, q, sf = self._run(counts, 2, 100)
        self._assert_matches_contiguous(gateup, masked_m, q, sf, None)
        _, _, inflated_q, inflated_sf = self._run(counts, 2, 200)
        for e, mm in enumerate(counts):
            with self.subTest(expert=e):
                self.assertTrue(torch.equal(inflated_q[e, :mm], q[e, :mm]))
                self.assertTrue(torch.equal(inflated_sf[e, :mm], sf[e, :mm]))

    def test_dequant_error_bounded(self):
        limit = 10.0
        gateup, masked_m, q, sf = self._run([3, 4, 5, 2], 3, 6, limit)
        for e, mm in enumerate(masked_m.tolist()):
            with self.subTest(expert=e):
                gate = gateup[e, :mm, : self.H].float().clamp(max=limit)
                up = gateup[e, :mm, self.H :].float().clamp(-limit, limit)
                act = gate * torch.sigmoid(gate) * up
                err = (_dequant(q[e, :mm], sf[e, :mm]) - act).abs()
                self.assertLessEqual(
                    err.max().item(), 1.01 * _group_scale(sf[e, :mm]).max().item()
                )

    def test_empty_experts_and_empty_batch(self):
        # A fully padded expert costs nothing but must not read the batch.
        gateup = self._gateup(seed=1)
        masked_m = self._masked_m([0, 0, 0, 0])
        q, sf = silu_mul_quant_mxfp4_masked(gateup, masked_m, 1, 4)
        self.assertEqual(q.shape, (self.E, self.M_MAX, self.H // 2))
        self.assertEqual(sf.shape, (self.E, self.M_MAX, self.H // 128))
        q, sf = silu_mul_quant_mxfp4_masked(gateup, self._masked_m([1, 1, 1, 1]), 1, 0)
        self.assertEqual(q.shape, (self.E, self.M_MAX, self.H // 2))
        # No padded rows at all: the first (and only) dimension is the expert one.
        q, sf = silu_mul_quant_mxfp4_masked(
            self._gateup(seed=2)[:, :0], self._masked_m([0, 0, 0, 0]), 1, 4
        )
        self.assertEqual(q.shape, (self.E, 0, self.H // 2))
        self.assertEqual(sf.shape, (self.E, 0, self.H // 128))

    def test_masked_hardware_cvt_path_requires_sm100(self):
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaises(RuntimeError):
                silu_mul_quant_mxfp4_masked(
                    self._gateup(), self._masked_m([1, 1, 1, 1]), 1, 4
                )

    def test_rejects_bad_operands(self):
        good = self._gateup()
        good_masked_m = self._masked_m([1, 1, 1, 1])
        E, m_max, two_h = good.shape

        def blank(width, device="cuda"):
            return torch.empty((E, m_max, width), dtype=torch.bfloat16, device=device)

        for label, args in (
            ("not 3D", (good.reshape(E, m_max * two_h), good_masked_m, 1, 4)),
            ("not bf16", (good.float(), good_masked_m, 1, 4)),
            ("not contiguous", (good[:, :, ::2], good_masked_m, 1, 4)),
            ("not CUDA", (blank(two_h, device="cpu"), good_masked_m.cpu(), 1, 4)),
            ("empty hidden", (blank(0), good_masked_m, 1, 4)),
            ("odd hidden", (blank(two_h + 1), good_masked_m, 1, 4)),
            ("hidden not a multiple of 128", (blank(2 * 129), good_masked_m, 1, 4)),
            ("masked_m wrong length", (good, self._masked_m([1, 1, 1]), 1, 4)),
            (
                "masked_m not int32",
                (good, self._masked_m([1, 1, 1, 1]).to(torch.int64), 1, 4),
            ),
            ("non-positive topk", (good, good_masked_m, 0, 4)),
            ("masked_m on cpu", (good, good_masked_m.cpu(), 1, 4)),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    silu_mul_quant_mxfp4_masked(*args)


class TestValidationBranches(CustomTestCase):
    """The guards each quantizer raises on, plus the empty-input early exits.

    A caller that hands over the wrong tensor lands on one of these, so they are
    worth pinning even though the happy path is covered above. Note
    `silu_mul_quant_mxfp4`'s `H <= 0` branch is unreachable: `N <= 0` and
    `N % 2 != 0` already reject everything below `N == 2`, which gives `H == 1`.
    """

    @staticmethod
    def _bf16(*shape, device="cuda"):
        return torch.empty(*shape, dtype=torch.bfloat16, device=device)

    def test_rejects_tensors_the_input_check_dislikes(self):
        for label, bad in (
            ("not 2D", torch.empty(4, dtype=torch.bfloat16, device="cuda")),
            ("not CUDA", self._bf16(4, 128, device="cpu")),
            ("not bf16", torch.empty(4, 128, dtype=torch.float16, device="cuda")),
            ("not contiguous", self._bf16(4, 256)[:, ::2]),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    quant_mxfp4_group32(bad)

    def test_hardware_cvt_paths_require_sm100(self):
        # Both v2 entry points must fail loudly on pre-Blackwell rather than
        # quietly pick another encoder.
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaises(RuntimeError):
                quant_mxfp4_group32_v2(self._bf16(2, 128))
            with self.assertRaises(RuntimeError):
                silu_mul_quant_mxfp4(self._bf16(2, 256))

    def test_v1_rejects_non_positive_k_and_returns_on_empty_m(self):
        with self.assertRaises(ValueError):
            quant_mxfp4_group32(self._bf16(2, 0))
        q, sf = quant_mxfp4_group32(self._bf16(0, 128))
        self.assertEqual((q.shape, q.dtype), ((0, 64), torch.int8))
        self.assertEqual((sf.shape, sf.dtype), ((0, 1), torch.int32))

    def test_v2_rejects_bad_k_and_returns_on_empty_m(self):
        with self.assertRaises(ValueError):
            quant_mxfp4_group32_v2(self._bf16(2, 0))
        with self.assertRaises(ValueError):
            quant_mxfp4_group32_v2(self._bf16(2, 192))
        q, sf = quant_mxfp4_group32_v2(self._bf16(0, 128))
        self.assertEqual((q.shape, q.dtype), ((0, 64), torch.int8))
        self.assertEqual((sf.shape, sf.dtype), ((0, 1), torch.int32))

    def test_silu_quant_rejects_bad_hidden_and_returns_on_empty_t(self):
        with self.assertRaises(ValueError):
            silu_mul_quant_mxfp4(self._bf16(2, 0))
        with self.assertRaises(ValueError):
            silu_mul_quant_mxfp4(self._bf16(2, 3))
        with self.assertRaises(ValueError):
            silu_mul_quant_mxfp4(self._bf16(2, 2))
        q, sf = silu_mul_quant_mxfp4(self._bf16(0, 256))
        self.assertEqual((q.shape, q.dtype), ((0, 64), torch.int8))
        self.assertEqual((sf.shape, sf.dtype), ((0, 1), torch.int32))


if __name__ == "__main__":
    unittest.main()
