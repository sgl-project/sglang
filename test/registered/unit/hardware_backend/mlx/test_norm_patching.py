"""Unit tests for the MLX RMSNorm patch pass and wrapper dispatch/fallback.

These run in CI stage A, which does NOT build the Metal kernel, so nothing here
may import or require ``sgl_kernel._metal``. Patching is exercised by
monkeypatching ``norm_wrapper._load_custom_rms_norm`` (same pattern as
``test_attention_patching.py`` uses for the RoPE loader).
"""

from __future__ import annotations

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn

    import sglang.srt.hardware_backend.mlx.norm_wrapper as norm_wrapper
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.mlx.norm_wrapper import (
        MLXRMSNormWrapper,
        patch_model_norms,
    )

    _DIMS = 8

    class GemmaStyleNorm(nn.Module):
        """The ``(1 + w)`` variant. Not ``nn.RMSNorm``; must never be patched."""

        def __init__(self, dims: int, eps: float = 1e-6):
            super().__init__()
            self.weight = mx.zeros((dims,))
            self.eps = eps

        def __call__(self, x: mx.array) -> mx.array:
            return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)

    class TinyLayer(nn.Module):
        def __init__(self, norm_cls):
            super().__init__()
            self.input_layernorm = norm_cls(_DIMS)
            self.post_attention_layernorm = norm_cls(_DIMS)

    class TinyModel(nn.Module):
        """Norms both nested in a layer list and as a direct attribute, so the
        patch pass's list traversal and plain-attribute paths are both hit."""

        def __init__(self, norm_cls=nn.RMSNorm):
            super().__init__()
            self.layers = [TinyLayer(norm_cls), TinyLayer(norm_cls)]
            self.norm = norm_cls(_DIMS)

    def _fake_kernel(x, w, eps):
        return mx.fast.rms_norm(x, w, eps)

    class _patched_loader:
        """Temporarily make the kernel loader return ``kernel`` (or None)."""

        def __init__(self, kernel):
            self._kernel = kernel

        def __enter__(self):
            self._orig = norm_wrapper._load_custom_rms_norm
            norm_wrapper._load_custom_rms_norm = lambda: self._kernel
            return self

        def __exit__(self, *exc):
            norm_wrapper._load_custom_rms_norm = self._orig


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxNormPatching(unittest.TestCase):
    def test_env_gate_off_is_a_no_op(self):
        model = TinyModel()
        with envs.SGLANG_MLX_USE_CUSTOM_RMSNORM.override(False):
            with _patched_loader(_fake_kernel):
                self.assertEqual(patch_model_norms(model), 0)
        self.assertIsInstance(model.norm, nn.RMSNorm)
        self.assertNotIsInstance(model.norm, MLXRMSNormWrapper)

    def test_kernel_unavailable_is_a_no_op(self):
        # This is CI's own situation: gate on, metallib never built.
        model = TinyModel()
        with envs.SGLANG_MLX_USE_CUSTOM_RMSNORM.override(True):
            with _patched_loader(None):
                self.assertEqual(patch_model_norms(model), 0)
        self.assertNotIsInstance(model.norm, MLXRMSNormWrapper)

    def test_patch_is_idempotent_and_reaches_nested_norms(self):
        model = TinyModel()
        with envs.SGLANG_MLX_USE_CUSTOM_RMSNORM.override(True):
            with _patched_loader(_fake_kernel):
                self.assertEqual(patch_model_norms(model), 5)
                for layer in model.layers:
                    self.assertIsInstance(layer.input_layernorm, MLXRMSNormWrapper)
                    self.assertIsInstance(
                        layer.post_attention_layernorm, MLXRMSNormWrapper
                    )
                self.assertIsInstance(model.norm, MLXRMSNormWrapper)
                self.assertEqual(patch_model_norms(model), 0)

    def test_gemma_style_norm_is_rejected(self):
        model = TinyModel(norm_cls=GemmaStyleNorm)
        with envs.SGLANG_MLX_USE_CUSTOM_RMSNORM.override(True):
            with _patched_loader(_fake_kernel):
                self.assertEqual(patch_model_norms(model), 0)
        self.assertIsInstance(model.norm, GemmaStyleNorm)

    def test_unsupported_dtype_falls_back_to_inner(self):
        calls = []

        def recording_kernel(x, w, eps):
            calls.append(x.shape)
            return mx.fast.rms_norm(x, w, eps)

        inner = nn.RMSNorm(_DIMS)
        wrapper = MLXRMSNormWrapper(inner, recording_kernel)
        x = mx.random.normal((2, _DIMS)).astype(mx.float16)
        y = wrapper(x)
        self.assertEqual(len(calls), 0)
        self.assertTrue(mx.allclose(y, mx.fast.rms_norm(x, inner.weight, inner.eps)))

        x = mx.random.normal((2, 3, _DIMS)).astype(mx.float16)
        y = wrapper(x)
        self.assertEqual(len(calls), 0)
        self.assertTrue(mx.allclose(y, mx.fast.rms_norm(x, inner.weight, inner.eps)))

    def test_supported_input_dispatches_to_kernel(self):
        calls = []

        def recording_kernel(x, w, eps):
            calls.append(x.shape)
            return mx.fast.rms_norm(x, w, eps)

        wrapper = MLXRMSNormWrapper(nn.RMSNorm(_DIMS), recording_kernel)
        x = mx.random.normal((2, _DIMS))
        y = wrapper(x)
        self.assertEqual(calls[-1], (2, _DIMS))
        self.assertEqual(y.shape, x.shape)

        x = mx.random.normal((2, 3, _DIMS))
        y = wrapper(x)
        self.assertEqual(calls[-1], (6, _DIMS))
        self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
