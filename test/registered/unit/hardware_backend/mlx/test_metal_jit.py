"""Unit tests for the MetalJitKernel JIT declaration surface and its
name-keyed registry.

Covers per-key compile caching, kernel name formatting, dtype_tag output,
warm_once dedupe and its non-array assertion, one on-device dispatch through
a trivial kernel, the registry's get/warm_once round trip via decorator
registration, per-name cache isolation, duplicate-name rejection, and the
@kernel decorator's validation plus the MetalJitOp default surface.

Hermetic: tiny synthetic kernels only; skips off Darwin/arm64 or without
``mlx``.
"""

from __future__ import annotations

import importlib.util
import platform
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

# Registered with the CPU suite (runtime no-op marker, parsed via AST). On
# non-Apple-Silicon CI runners the whole TestCase skips via the @skipUnless
# guard below.
register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "Apple-Silicon-only test (requires Darwin/arm64 + mlx)"

if _IS_APPLE_SILICON and _HAS_MLX:
    import mlx.core as mx

    from sglang.srt.hardware_backend.mlx import metal_jit

_ADD_ONE_SOURCE = r"""
    uint tid = thread_position_in_grid.x;
    out[tid] = x[tid] + T(1);
"""


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMetalJitKernel(unittest.TestCase):
    def _make(self):
        return metal_jit.MetalJitKernel(
            name_template="metal_jit_test_add_one_{0}",
            input_names=["x"],
            output_names=["out"],
            source=_ADD_ONE_SOURCE,
        )

    def test_dtype_tag(self):
        tags = {metal_jit.dtype_tag(d) for d in (mx.float16, mx.bfloat16, mx.float32)}
        self.assertEqual(len(tags), 3)
        for tag in tags:
            self.assertNotIn(".", tag)
            self.assertNotIn("mlx", tag)

    def test_get_caches_per_key(self):
        kern = self._make()
        a1 = kern.get(mx.float16)
        a2 = kern.get(mx.float16)
        b = kern.get(mx.bfloat16)
        self.assertIs(a1, a2)
        self.assertIsNot(a1, b)

    def test_on_device_dispatch(self):
        kern = self._make()
        compiled = kern.get(mx.float16)
        x = mx.arange(256).astype(mx.float16)
        (out,) = compiled(
            inputs=[x],
            template=[("T", mx.float16)],
            grid=(256, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(256,)],
            output_dtypes=[mx.float16],
        )
        self.assertTrue(bool(mx.array_equal(out, x + 1).item()))

    def test_warm_once_runs_dispatch_once_per_key(self):
        kern = self._make()
        calls = []

        def dispatch():
            calls.append(1)
            return mx.zeros(1)

        self.assertTrue(kern.warm_once(("k1",), dispatch))
        self.assertFalse(kern.warm_once(("k1",), dispatch))
        self.assertEqual(len(calls), 1)
        self.assertTrue(kern.warm_once(("k2",), dispatch))
        self.assertEqual(len(calls), 2)

    def test_warm_once_asserts_on_non_array(self):
        kern = self._make()
        with self.assertRaises(AssertionError):
            kern.warm_once(("bad",), lambda: None)
        # The failed warm must not mark the key: a later valid dispatch runs.
        self.assertTrue(kern.warm_once(("bad",), lambda: mx.zeros(1)))


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMetalJitRegistry(unittest.TestCase):
    """Name-keyed registry: @kernel registration, get()/warm_once() dispatch."""

    def _register(self, name):
        @metal_jit.kernel(
            name=name,
            name_template=f"{name}_{{0}}",
            input_names=["x"],
            output_names=["out"],
        )
        class _Op(metal_jit.MetalJitOp):
            source = _ADD_ONE_SOURCE

        return _Op

    def test_register_get_round_trip(self):
        self._register("metal_jit_test_registry_round_trip")
        compiled = metal_jit.get("metal_jit_test_registry_round_trip", mx.float16)
        x = mx.arange(256).astype(mx.float16)
        (out,) = compiled(
            inputs=[x],
            template=[("T", mx.float16)],
            grid=(256, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(256,)],
            output_dtypes=[mx.float16],
        )
        self.assertTrue(bool(mx.array_equal(out, x + 1).item()))

    def test_cache_independent_per_name(self):
        self._register("metal_jit_test_registry_a")
        self._register("metal_jit_test_registry_b")
        a1 = metal_jit.get("metal_jit_test_registry_a", mx.float16)
        b1 = metal_jit.get("metal_jit_test_registry_b", mx.float16)
        a2 = metal_jit.get("metal_jit_test_registry_a", mx.float16)
        self.assertIs(a1, a2)
        self.assertIsNot(a1, b1)

    def test_duplicate_name_raises(self):
        self._register("metal_jit_test_registry_dup")
        with self.assertRaises(ValueError):
            self._register("metal_jit_test_registry_dup")

    def test_warm_once_routes_and_dedupes_by_name(self):
        self._register("metal_jit_test_registry_warm")
        calls = []

        def dispatch():
            calls.append(1)
            return mx.zeros(1)

        name = "metal_jit_test_registry_warm"
        self.assertTrue(metal_jit.warm_once(name, ("k1",), dispatch))
        self.assertFalse(metal_jit.warm_once(name, ("k1",), dispatch))
        self.assertEqual(len(calls), 1)


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMetalJitOpDecorator(unittest.TestCase):
    """@metal_jit.kernel validation and the MetalJitOp default surface."""

    def _decorate(self, name, cls_body_source=_ADD_ONE_SOURCE):
        @metal_jit.kernel(
            name=name,
            name_template=f"{name}_{{0}}",
            input_names=["x"],
            output_names=["out"],
        )
        class _Op(metal_jit.MetalJitOp):
            source = cls_body_source

        return _Op

    def test_decorator_returns_class_unchanged(self):
        op_cls = self._decorate("metal_jit_test_op_identity")
        self.assertTrue(issubclass(op_cls, metal_jit.MetalJitOp))
        self.assertEqual(op_cls.source, _ADD_ONE_SOURCE)

    def test_rejects_non_metaljitop_class(self):
        with self.assertRaises(TypeError):

            @metal_jit.kernel(
                name="metal_jit_test_op_not_subclass",
                name_template="metal_jit_test_op_not_subclass_{0}",
                input_names=["x"],
                output_names=["out"],
            )
            class _NotAnOp:
                source = _ADD_ONE_SOURCE

    def test_missing_source_raises_and_leaves_name_free(self):
        name = "metal_jit_test_op_missing_source"
        with self.assertRaises(TypeError):

            @metal_jit.kernel(
                name=name,
                name_template=f"{name}_{{0}}",
                input_names=["x"],
                output_names=["out"],
            )
            class _NoSource(metal_jit.MetalJitOp):
                pass

        # Validation precedes registry insertion, so the failed registration
        # must not consume the name.
        self._decorate(name)

    def test_warmup_specs_default_empty(self):
        op_cls = self._decorate("metal_jit_test_op_warmup_default")
        self.assertEqual(list(op_cls().warmup_specs(model=None)), [])

    def test_resolved_kernel_name_matches_template(self):
        name = "metal_jit_test_op_name_template"

        @metal_jit.kernel(
            name=name,
            name_template=f"{name}_y{{0}}_s{{1}}",
            input_names=["x"],
            output_names=["out"],
        )
        class _Op(metal_jit.MetalJitOp):
            source = _ADD_ONE_SOURCE

        captured = []
        orig = mx.fast.metal_kernel

        def spy(*, name, **kwargs):
            captured.append(name)
            return orig(name=name, **kwargs)

        mx.fast.metal_kernel = spy
        try:
            metal_jit.get(name, mx.float16, mx.float32)
        finally:
            mx.fast.metal_kernel = orig
        self.assertEqual(captured, [f"{name}_yfloat16_sfloat32"])


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMetalJitOpDispatchTemplate(unittest.TestCase):
    """Base dispatch routes through can_fuse to the fused or fallback hook."""

    def _stub_op(self, eligible):
        class _Stub(metal_jit.MetalJitOp):
            def can_fuse(self, *args, **kwargs):
                return eligible

            def dispatch_fused(self, *args, **kwargs):
                return ("fused", args, kwargs)

            def dispatch_fallback(self, *args, **kwargs):
                return ("fallback", args, kwargs)

        return _Stub()

    def test_dispatch_routes_to_fused_when_eligible(self):
        self.assertEqual(
            self._stub_op(True).dispatch(1, k=2), ("fused", (1,), {"k": 2})
        )

    def test_dispatch_routes_to_fallback_when_ineligible(self):
        self.assertEqual(
            self._stub_op(False).dispatch(1, k=2), ("fallback", (1,), {"k": 2})
        )


if __name__ == "__main__":
    unittest.main()
