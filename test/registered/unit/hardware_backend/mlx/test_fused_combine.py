"""Verification harness for the fused MoE combine Metal kernel.

Target: ``sglang.srt.hardware_backend.mlx.moe.fused_combine``. The kernel fuses
``out[b, h] = sum_k y[b, k, h] * scores[b, k]`` into one Metal dispatch, with an
fp32 product and fp32 accumulate narrowed to ``y.dtype`` on write.

What this proves
----------------
1. Regression guard: the original fp16-y / fp16-scores claim across 17 shapes,
   bit-exact against an fp32 ground-truth reference computed in the kernel's own
   accumulation order.
2. The S1 gate widening: scores now carry an independent Metal dtype, so the
   common production combo (fp16 y with fp32 routing scores) and bf16 y both run
   on the fused path instead of falling back.
3. The fused Metal kernel actually fires on every widened dtype case (a spy on
   ``_get_kernel``), so a silent fallback cannot satisfy the test. A negative
   control confirms ineligible inputs do take the fallback.

Acceptance standard (scores-dtype driven, and grounded in fp arithmetic)
------------------------------------------------------------------------
- ``scores`` fp16 -> STRICT bit-exact. When both operands are low precision
  (<=11-bit mantissa each) every ``y * s`` product is exact in fp32 (<=22 bits
  <= 24), so FMA contraction and separate mul+add coincide and the kernel's
  sequential fp32 accumulate matches the reference bit-for-bit.
- ``scores`` fp32 -> correctly rounded to within a tight narrowing tolerance.
  fp32 scores make the product inexact in fp32, so the Metal compiler's FMA
  contraction diverges from a separate mul+add reference by a sub-fp32-ULP,
  which tips <=2 fp16/bf16 ULP on a small fraction of elements. The fused path
  is strictly closer to the true reduction, so this is correctly-rounded fp16,
  not a defect.

Hermetic: tensor fixtures only, no model download, no network. Skips on
non-Apple-Silicon platforms and when ``mlx`` is missing.
"""

from __future__ import annotations

import importlib.util
import platform
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

# Registered with the CPU suite (runtime no-op marker, parsed via AST). On
# non-Apple-Silicon CI runners the whole TestCase skips via the @skipUnless
# guard below, so this is the harmless "yes this test exists" registry signal.
register_cpu_ci(est_time=90, suite="base-a-test-cpu")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "Apple-Silicon-only test (requires Darwin/arm64 + mlx)"

if _IS_APPLE_SILICON and _HAS_MLX:
    import mlx.core as mx

    from sglang.srt.hardware_backend.mlx.moe import fused_combine as fc

# 17 shapes inside the eligibility envelope (H % 256 == 0): decode (B == 1) and
# prefill (B > 1), TOP_K in {1, 2, 4, 8, 16}, several real hidden sizes.
_SHAPES = [
    (1, 8, 2048),  # Qwen3-30B-A3B decode shape
    (1, 8, 256),  # minimum eligible H
    (1, 1, 512),  # TOP_K == 1 (weighted copy)
    (1, 2, 1024),
    (1, 4, 768),
    (1, 8, 4096),
    (2, 8, 2048),
    (4, 8, 512),
    (8, 8, 1024),
    (16, 4, 256),
    (32, 8, 2048),
    (1, 8, 5120),
    (1, 8, 7168),  # DeepSeek-scale hidden
    (1, 16, 2048),  # high TOP_K
    (128, 2, 256),  # large B, small H
    (3, 8, 768),  # odd B
    (1, 8, 1536),
]

# dtype combos, all eligible after the S1 gate widening. (y dtype, scores dtype)
_STRICT_COMBOS = [
    ("fp16y_fp16s", "float16", "float16"),  # original regression guard
    ("bf16y_fp16s", "bfloat16", "float16"),  # bf16 y support
]
_ROUNDED_COMBOS = [
    ("fp16y_fp32s", "float16", "float32"),  # production combo (real routers)
    ("bf16y_fp32s", "bfloat16", "float32"),  # bf16 y with fp32 scores
]

# Correctly-rounded narrowing tolerance for the fp32-scores cases, sized to a
# few low-precision ULP (observed worst was 2). fp16 ULP ~= 2**-10 relative,
# bf16 ULP ~= 2**-7 relative; these bounds carry headroom over the observed max
# while still failing hard on a broken kernel (off by O(magnitude)).
_ROUNDED_TOL = {
    "float16": 2**-8,
    "bfloat16": 2**-6,
}


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestFusedCombine(unittest.TestCase):
    """Bit-exact / correctly-rounded verification for ``fused_combine``."""

    # (combo, shape, standard, max_ulp, fused_fired, passed) rows for the table.
    _results: list = []

    @staticmethod
    def _reference_fp32(y, scores):
        """fp32 ground truth in the kernel's accumulation order (sequential k).

        Promotes both operands to fp32, forms the product in fp32, and
        accumulates in fp32 left-to-right over TOP_K, mirroring the Metal k loop.
        """
        _, topk, hidden = y.shape
        yf = y.astype(mx.float32)
        sf = scores.astype(mx.float32)
        acc = mx.zeros((y.shape[0], hidden), dtype=mx.float32)
        for k in range(topk):
            acc = acc + yf[:, k, :] * sf[:, k][:, None]
        return acc

    @staticmethod
    def _run_fused_traced(y, scores):
        """Run fused_combine while spying on _get_kernel.

        Returns (out, fused_fired). fused_fired is True iff the Metal dispatch
        path was entered (not the reference fallback).
        """
        calls = []
        orig = fc._get_kernel

        def spy(y_dtype, scores_dtype):
            calls.append((y_dtype, scores_dtype))
            return orig(y_dtype, scores_dtype)

        fc._get_kernel = spy
        try:
            out = fc.fused_combine(y, scores)
            mx.eval(out)
        finally:
            fc._get_kernel = orig
        return out, (len(calls) > 0)

    @staticmethod
    def _low_precision_ulp(a, b):
        """Max integer ULP gap between two same-dtype low-precision arrays.

        Both are viewed as uint16 bit patterns; for finite same-sign values the
        IEEE encoding is monotone, so |int(a) - int(b)| is the ULP distance.
        Used for reporting only; the pass/fail gate is array_equal / allclose.
        """
        ai = a.view(mx.uint16).astype(mx.int32)
        bi = b.view(mx.uint16).astype(mx.int32)
        return int(mx.max(mx.abs(ai - bi)).item())

    def _make_inputs(self, shape, y_name, s_name, seed):
        mx.random.seed(seed)
        B, K, H = shape
        y = mx.random.uniform(-2.0, 2.0, (B, K, H)).astype(getattr(mx, y_name))
        scores = mx.random.uniform(0.0, 1.0, (B, K)).astype(getattr(mx, s_name))
        return y, scores

    def test_scores_fp16_bit_exact(self):
        """fp16-scores combos: strictly bit-exact vs fp32 ground truth, 17 shapes."""
        for ci, (cname, y_name, s_name) in enumerate(_STRICT_COMBOS):
            for si, shape in enumerate(_SHAPES):
                with self.subTest(combo=cname, shape=shape):
                    y, scores = self._make_inputs(shape, y_name, s_name, 1000 * ci + si)
                    self.assertTrue(fc.can_fuse(y, scores))
                    out, fired = self._run_fused_traced(y, scores)
                    expected = self._reference_fp32(y, scores).astype(
                        getattr(mx, y_name)
                    )
                    exact = bool(mx.array_equal(out, expected).item())
                    ulp = 0 if exact else self._low_precision_ulp(out, expected)
                    self._results.append(
                        (cname, shape, "bit-exact", ulp, fired, fired and exact)
                    )
                    self.assertTrue(fired, f"{cname} {shape}: fused path did not fire")
                    self.assertEqual(out.dtype, getattr(mx, y_name))
                    self.assertTrue(
                        exact,
                        f"{cname} {shape}: not bit-exact, max ULP={ulp}",
                    )

    def test_scores_fp32_correctly_rounded(self):
        """fp32-scores combos: correctly rounded to within narrowing tolerance."""
        for ci, (cname, y_name, s_name) in enumerate(_ROUNDED_COMBOS):
            tol = _ROUNDED_TOL[y_name]
            for si, shape in enumerate(_SHAPES):
                with self.subTest(combo=cname, shape=shape):
                    y, scores = self._make_inputs(
                        shape, y_name, s_name, 2000 * (ci + 1) + si
                    )
                    self.assertTrue(fc.can_fuse(y, scores))
                    out, fired = self._run_fused_traced(y, scores)
                    gt = self._reference_fp32(y, scores)
                    expected = gt.astype(getattr(mx, y_name))
                    close = bool(
                        mx.allclose(
                            out.astype(mx.float32), gt, rtol=tol, atol=tol
                        ).item()
                    )
                    ulp = self._low_precision_ulp(out, expected)
                    self._results.append(
                        (cname, shape, "rounded", ulp, fired, fired and close)
                    )
                    self.assertTrue(fired, f"{cname} {shape}: fused path did not fire")
                    self.assertEqual(out.dtype, getattr(mx, y_name))
                    self.assertTrue(
                        close,
                        f"{cname} {shape}: exceeds narrowing tol {tol}, max ULP={ulp}",
                    )

    def test_scores_dtype_reaches_kernel_independently(self):
        """The (y, scores) dtype pair passed to the kernel is independent (S1 core)."""
        shape = (1, 8, 2048)
        for _, y_name, s_name in _STRICT_COMBOS + _ROUNDED_COMBOS:
            with self.subTest(y=y_name, scores=s_name):
                y, scores = self._make_inputs(shape, y_name, s_name, 7)
                seen = []
                orig = fc._get_kernel
                fc._get_kernel = lambda yd, sd: seen.append((yd, sd)) or orig(yd, sd)
                try:
                    mx.eval(fc.fused_combine(y, scores))
                finally:
                    fc._get_kernel = orig
                self.assertEqual(seen, [(getattr(mx, y_name), getattr(mx, s_name))])

    def test_fallback_taken_when_ineligible(self):
        """Negative control: ineligible inputs take the fallback, not the kernel.

        Proves the fused-fired spy actually discriminates and that can_fuse
        rejects out-of-envelope dtypes/shapes while still returning correct math.
        """
        cases = [
            ("fp32_y_rejected", (1, 8, 2048), "float32", "float32"),
            ("H_not_mult_256", (1, 8, 500), "float16", "float16"),
            ("scores_rank_bad", (1, 8, 2048), "float16", "float16"),  # mangled below
        ]
        for name, shape, y_name, s_name in cases:
            with self.subTest(case=name):
                y, scores = self._make_inputs(shape, y_name, s_name, 11)
                if name == "scores_rank_bad":
                    scores = scores.reshape(1, 8, 1)  # rank 3 -> ineligible
                self.assertFalse(fc.can_fuse(y, scores))
                out, fired = self._run_fused_traced(y, scores)
                self.assertFalse(fired, f"{name}: fused path fired on ineligible input")
                # Fallback still returns the correct reduction.
                ref = (y * scores[..., None]).sum(axis=-2)
                self.assertTrue(
                    mx.allclose(
                        out.astype(mx.float32),
                        ref.astype(mx.float32),
                        rtol=1e-2,
                        atol=1e-2,
                    ).item()
                )

    @classmethod
    def tearDownClass(cls):
        if not cls._results:
            return
        print("\n\n=== fused_combine verification matrix ===")
        print(
            f"{'combo':13} {'shape':16} {'standard':10} {'maxULP':>6} "
            f"{'fired':>5} {'pass':>4}"
        )
        for cname, shape, standard, ulp, fired, passed in cls._results:
            print(
                f"{cname:13} {str(shape):16} {standard:10} {ulp:>6} "
                f"{str(fired):>5} {('OK' if passed else 'FAIL'):>4}"
            )
        n = len(cls._results)
        npass = sum(1 for r in cls._results if r[5])
        print(
            f"--- {npass}/{n} cases pass "
            f"({sum(1 for r in cls._results if r[4])}/{n} fired the fused path) ---"
        )


if __name__ == "__main__":
    unittest.main()
