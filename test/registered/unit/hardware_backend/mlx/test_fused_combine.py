"""Verification for the fused MoE combine kernel (``fused_combine``).

A spy on ``metal_jit.get`` asserts the Metal kernel dispatched on every
eligible case; negative controls confirm ineligible inputs fall back in
``y.dtype``.

Acceptance is scores-dtype driven. fp16 and bf16 scores: bit-exact vs an
fp32 reference, since <=11-bit mantissas make every ``y * s`` product exact
in fp32 (<=22 bits <= 24) and FMA contraction coincides with separate
mul+add. fp32 scores: the product is inexact, FMA contraction diverges by a
sub-fp32-ULP, and the narrowed result is asserted <=2 fp16 / <=1 bf16 ULP.

Hermetic: tensor fixtures only; skips off Darwin/arm64 or without ``mlx``.
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

    from sglang.srt.hardware_backend.mlx import metal_jit
    from sglang.srt.hardware_backend.mlx.moe import fused_combine as fc

# Shapes inside the eligibility envelope (H % 256 == 0), decode and prefill.
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

# (name, y dtype, scores dtype), all inside the can_fuse dtype envelope.
_STRICT_COMBOS = [
    ("fp16y_fp16s", "float16", "float16"),
    ("bf16y_fp16s", "bfloat16", "float16"),
    ("fp16y_bf16s", "float16", "bfloat16"),
    ("bf16y_bf16s", "bfloat16", "bfloat16"),  # bf16 router (Qwen3-30B-A3B regime)
]
_ROUNDED_COMBOS = [
    ("fp16y_fp32s", "float16", "float32"),  # production combo (real routers)
    ("bf16y_fp32s", "bfloat16", "float32"),
]

# Correctly-rounded narrowing tolerance for the fp32-scores cases, sized to a
# few low-precision ULP (observed worst was 2). fp16 ULP ~= 2**-10 relative,
# bf16 ULP ~= 2**-7 relative; these bounds carry headroom over the observed max
# while still failing hard on a broken kernel (off by O(magnitude)).
_ROUNDED_TOL = {
    "float16": 2**-8,
    "bfloat16": 2**-6,
}

# Enforced integer-ULP ceiling for the fp32-scores cases, set to the observed
# maxima with no headroom: FMA contraction accounts for at most 2 fp16 / 1 bf16
# ULP, so anything past these means the kernel's arithmetic changed.
_ROUNDED_MAX_ULP = {
    "float16": 2,
    "bfloat16": 1,
}


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestFusedCombine(unittest.TestCase):
    """Bit-exact / correctly-rounded verification for ``fused_combine``."""

    # (combo, shape, standard, max_ulp, fused_fired, passed) rows for the table.
    _results: list = []

    @staticmethod
    def _reference_fp32(y, scores):
        """fp32 ground truth in the kernel's accumulation order (sequential k).

        Rank generic: leading dims flatten exactly like the kernel dispatch,
        and the result is returned in the flattened [B_flat, H] layout.
        """
        topk, hidden = y.shape[-2], y.shape[-1]
        yf = y.astype(mx.float32).reshape(-1, topk, hidden)
        sf = scores.astype(mx.float32).reshape(-1, topk)
        acc = mx.zeros((yf.shape[0], hidden), dtype=mx.float32)
        for k in range(topk):
            acc = acc + yf[:, k, :] * sf[:, k][:, None]
        return acc

    @staticmethod
    def _run_fused_traced(y, scores):
        """Run fused_combine spying on metal_jit.get; returns (out, fused_fired)."""
        calls = []
        orig = metal_jit.get

        def spy(name, *dtypes):
            if name == "fused_moe_combine":
                calls.append(dtypes)
            return orig(name, *dtypes)

        metal_jit.get = spy
        try:
            out = fc.fused_combine(y, scores)
            mx.eval(out)
        finally:
            metal_jit.get = orig
        return out, (len(calls) > 0)

    @staticmethod
    def _low_precision_ulp(a, b):
        """Max integer ULP gap between two same-dtype low-precision arrays.

        Both are viewed as uint16 bit patterns; for finite same-sign values the
        IEEE encoding is monotone, so |int(a) - int(b)| is the ULP distance.
        """
        ai = a.view(mx.uint16).astype(mx.int32)
        bi = b.view(mx.uint16).astype(mx.int32)
        return int(mx.max(mx.abs(ai - bi)).item())

    def _make_inputs(self, shape, y_name, s_name, seed):
        """y with the given shape (rank >= 3), scores shaped y.shape[:-1]."""
        mx.random.seed(seed)
        y = mx.random.uniform(-2.0, 2.0, tuple(shape)).astype(getattr(mx, y_name))
        scores = mx.random.uniform(0.0, 1.0, tuple(shape[:-1])).astype(
            getattr(mx, s_name)
        )
        return y, scores

    def test_scores_fp16_bit_exact(self):
        """fp16-scores combos: strictly bit-exact vs the fp32 ground truth."""
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
                    passed = fired and close and ulp <= _ROUNDED_MAX_ULP[y_name]
                    self._results.append((cname, shape, "rounded", ulp, fired, passed))
                    self.assertTrue(fired, f"{cname} {shape}: fused path did not fire")
                    self.assertEqual(out.dtype, getattr(mx, y_name))
                    self.assertTrue(
                        close,
                        f"{cname} {shape}: exceeds narrowing tol {tol}, max ULP={ulp}",
                    )
                    self.assertLessEqual(
                        ulp,
                        _ROUNDED_MAX_ULP[y_name],
                        f"{cname} {shape}: max ULP={ulp} > {_ROUNDED_MAX_ULP[y_name]}",
                    )

    def test_leading_dims_match_reference(self):
        """Rank >= 4 inputs (the mlx-lm combine site) dispatch and match.

        Shapes mirror the serving trace on Qwen1.5-MoE-A2.7B: [1, L, k, H]
        prefill, [R, 1, k, H] batched decode, plus a rank 5 sweep case.
        """
        strict_shapes = [
            (1, 268, 4, 2048),  # prefill, L=268
            (2, 1, 4, 2048),  # batched decode, R=2
            (1, 1, 4, 2048),  # single-request decode
            (3, 5, 8, 512),
            (2, 3, 2, 4, 256),  # rank 5
        ]
        for si, shape in enumerate(strict_shapes):
            for cname, y_name, s_name in _STRICT_COMBOS:
                with self.subTest(combo=cname, shape=shape):
                    y, scores = self._make_inputs(shape, y_name, s_name, 3000 + si)
                    self.assertTrue(fc.can_fuse(y, scores))
                    out, fired = self._run_fused_traced(y, scores)
                    self.assertTrue(fired, f"{cname} {shape}: fused path did not fire")
                    self.assertEqual(tuple(out.shape), tuple(shape[:-2]) + shape[-1:])
                    self.assertEqual(out.dtype, getattr(mx, y_name))
                    expected = self._reference_fp32(y, scores).astype(
                        getattr(mx, y_name)
                    )
                    self.assertTrue(
                        bool(
                            mx.array_equal(out.reshape(-1, shape[-1]), expected).item()
                        ),
                        f"{cname} {shape}: not bit-exact vs reference",
                    )
        # One fp32-scores rank 4 case under the rounded standard.
        shape = (1, 268, 4, 2048)
        y, scores = self._make_inputs(shape, "float16", "float32", 3100)
        self.assertTrue(fc.can_fuse(y, scores))
        out, fired = self._run_fused_traced(y, scores)
        self.assertTrue(fired)
        expected = self._reference_fp32(y, scores).astype(mx.float16)
        ulp = self._low_precision_ulp(out.reshape(-1, shape[-1]), expected)
        self.assertLessEqual(ulp, _ROUNDED_MAX_ULP["float16"])

    def test_leading_dim_guard_negatives(self):
        """Shape mismatches under the generalized contract fall back."""
        y, _ = self._make_inputs((2, 3, 4, 256), "float16", "float16", 21)
        cases = [
            ("rank_gap_two", mx.zeros((2, 4), dtype=mx.float16)),
            ("lead_mismatch", mx.zeros((3, 3, 4), dtype=mx.float16)),
            ("topk_mismatch", mx.zeros((2, 3, 5), dtype=mx.float16)),
        ]
        for name, scores in cases:
            with self.subTest(case=name):
                self.assertFalse(fc.can_fuse(y, scores))
        y2d = mx.zeros((4, 256), dtype=mx.float16)
        self.assertFalse(fc.can_fuse(y2d, mx.zeros((4,), dtype=mx.float16)))

    def test_scores_dtype_reaches_kernel_independently(self):
        """The (y, scores) dtype pair reaches the kernel unmerged."""
        shape = (1, 8, 2048)
        for _, y_name, s_name in _STRICT_COMBOS + _ROUNDED_COMBOS:
            with self.subTest(y=y_name, scores=s_name):
                y, scores = self._make_inputs(shape, y_name, s_name, 7)
                seen = []
                orig = metal_jit.get

                def spy(name, *dtypes):
                    if name == "fused_moe_combine":
                        seen.append(dtypes)
                    return orig(name, *dtypes)

                metal_jit.get = spy
                try:
                    mx.eval(fc.fused_combine(y, scores))
                finally:
                    metal_jit.get = orig
                self.assertEqual(seen, [(getattr(mx, y_name), getattr(mx, s_name))])

    def test_fallback_taken_when_ineligible(self):
        """Ineligible inputs take the fallback (proves the fired spy discriminates)."""
        cases = [
            ("fp32_y_rejected", (1, 8, 2048), "float32", "float32"),
            ("H_not_mult_256", (1, 8, 500), "float16", "float16"),
            ("H_not_mult_256_fp32s", (1, 8, 500), "float16", "float32"),
            ("scores_rank_bad", (1, 8, 2048), "float16", "float16"),  # mangled below
            ("B_zero", (0, 8, 256), "float16", "float32"),
            ("K_zero", (1, 0, 256), "float16", "float32"),
            ("H_zero", (1, 8, 0), "float16", "float32"),
        ]
        for name, shape, y_name, s_name in cases:
            with self.subTest(case=name):
                y, scores = self._make_inputs(shape, y_name, s_name, 11)
                if name == "scores_rank_bad":
                    scores = scores.reshape(1, 8, 1)  # rank 3 -> ineligible
                self.assertFalse(fc.can_fuse(y, scores))
                out, fired = self._run_fused_traced(y, scores)
                self.assertFalse(fired, f"{name}: fused path fired on ineligible input")
                self.assertEqual(out.dtype, getattr(mx, y_name))
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


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestFusedMoeCombineKernelOp(unittest.TestCase):
    """Decorator plus class registration surface of the combine op."""

    def test_op_class_shape(self):
        self.assertTrue(issubclass(fc.FusedMoeCombineKernel, metal_jit.MetalJitOp))
        self.assertEqual(fc.FusedMoeCombineKernel.source, fc._KERNEL_SOURCE)
        # The module level guard is the class's own, not a diverging copy.
        self.assertIs(fc.can_fuse, fc.FusedMoeCombineKernel.can_fuse)

    def test_warmup_specs_default_empty(self):
        # Inherited default; the AOT policy layer lands in a follow up PR.
        self.assertEqual(list(fc.FusedMoeCombineKernel().warmup_specs(model=None)), [])


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestPatchMoeCombine(unittest.TestCase):
    """patch_moe_combine_with_fused: subclass swap with no-op idempotency."""

    def _tiny_model(self):
        """Real mlx-lm MoE blocks at toy dims, random init, no downloads."""
        import mlx.nn as nn
        from mlx_lm.models import qwen2_moe, qwen3_moe

        q2 = qwen2_moe.Qwen2MoeSparseMoeBlock(
            qwen2_moe.ModelArgs(
                model_type="qwen2_moe",
                hidden_size=64,
                num_hidden_layers=1,
                intermediate_size=128,
                num_attention_heads=4,
                num_experts_per_tok=2,
                num_experts=4,
                moe_intermediate_size=32,
                shared_expert_intermediate_size=64,
                rms_norm_eps=1e-6,
                vocab_size=128,
            )
        )
        q3 = qwen3_moe.Qwen3MoeSparseMoeBlock(
            qwen3_moe.ModelArgs(
                model_type="qwen3_moe",
                hidden_size=64,
                num_hidden_layers=1,
                intermediate_size=128,
                num_attention_heads=4,
                num_experts=4,
                num_experts_per_tok=2,
                decoder_sparse_step=1,
                mlp_only_layers=[],
                moe_intermediate_size=32,
                rms_norm_eps=1e-6,
                vocab_size=128,
                num_key_value_heads=2,
                head_dim=16,
                rope_theta=10000.0,
                tie_word_embeddings=False,
                max_position_embeddings=2048,
                norm_topk_prob=True,
            )
        )

        class _Layer(nn.Module):
            def __init__(self, mlp):
                super().__init__()
                self.mlp = mlp

        class _Inner(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers

        class _Model(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.model = _Inner(layers)

        return _Model([_Layer(q2), _Layer(q3)]), q2, q3

    def test_patch_is_idempotent_no_op(self):
        model, q2, q3 = self._tiny_model()
        x = mx.random.normal((1, 3, 64))
        stock_q2, stock_q3 = q2(x), q3(x)
        mx.eval(stock_q2, stock_q3)

        self.assertEqual(fc.patch_moe_combine_with_fused(model), 2)
        self.assertTrue(type(q2).__name__.endswith("_FusedCombine"))
        self.assertTrue(type(q3).__name__.endswith("_FusedCombine"))
        once_q2, once_q3 = q2(x), q3(x)

        cls2, cls3 = type(q2), type(q3)
        self.assertEqual(fc.patch_moe_combine_with_fused(model), 0)
        self.assertIs(type(q2), cls2)
        self.assertIs(type(q3), cls3)
        twice_q2, twice_q3 = q2(x), q3(x)
        self.assertTrue(bool(mx.array_equal(once_q2, twice_q2).item()))
        self.assertTrue(bool(mx.array_equal(once_q3, twice_q3).item()))
        # H=64 fails the H % 256 gate, so the patched forward takes the inline
        # fallback: bit-identical to the stock forward.
        self.assertTrue(bool(mx.array_equal(stock_q2, once_q2).item()))
        self.assertTrue(bool(mx.array_equal(stock_q3, once_q3).item()))


if __name__ == "__main__":
    unittest.main()
