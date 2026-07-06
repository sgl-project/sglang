"""Unit tests for the MetalJitKernel JIT declaration surface and its
name-keyed registry.

Covers per-key compile caching, kernel name formatting, dtype_tag output,
warm_once dedupe and its non-array assertion, one on-device dispatch through
a trivial kernel, and the registry's register/get/warm_once round trip,
per-name cache isolation, and duplicate-name rejection.

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
    """Name-keyed registry: register()/get()/warm_once() over MetalJitKernel."""

    def _register(self, name):
        metal_jit.register(
            name,
            name_template=f"{name}_{{0}}",
            input_names=["x"],
            output_names=["out"],
            source=_ADD_ONE_SOURCE,
        )

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


if __name__ == "__main__":
    unittest.main()
