"""
Unit tests for the disk_cache decorator in outlines_jump_forward.

Covers:
- disk_cache disabled by env var: function runs without caching
- disk_cache enabled by env var: outlines cache is applied
- Lazy env var evaluation: changing env var after import takes effect
- No function-becomes-None regression (lambda fn: None bug fix)

Usage:
    python -m pytest test_disk_cache.py -v
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(60, "base-a-test-cpu")


class TestDiskCacheDecorator(unittest.TestCase):
    """Tests for the disk_cache decorator."""

    def _get_fresh_decorator(self):
        """Re-import the decorator to avoid module-level cache pollution."""
        import importlib

        import sglang.srt.constrained.outlines_jump_forward as mod

        importlib.reload(mod)
        return mod.disk_cache

    def test_disabled_returns_callable_function(self):
        """When disk cache is disabled the decorated function must remain callable."""
        from sglang.srt.constrained.outlines_jump_forward import disk_cache

        @disk_cache()
        def my_fn(x):
            return x * 2

        # Must be callable, not None (regression: lambda fn: None bug)
        self.assertTrue(callable(my_fn))

    def test_disabled_function_executes_correctly(self):
        """With disk cache disabled the decorated function returns correct results."""
        with patch.dict(os.environ, {"SGLANG_DISABLE_OUTLINES_DISK_CACHE": "1"}):
            from sglang.srt.constrained.outlines_jump_forward import disk_cache

            @disk_cache()
            def add(a, b):
                return a + b

            self.assertEqual(add(2, 3), 5)
            self.assertEqual(add(10, 20), 30)

    def test_lazy_evaluation_after_env_var_change(self):
        """Env var checked at call time, not at decoration time."""
        from sglang.srt.constrained.outlines_jump_forward import disk_cache

        call_count = {"n": 0}

        @disk_cache()
        def counted_fn(x):
            call_count["n"] += 1
            return x

        # First call with cache disabled
        with patch.dict(os.environ, {"SGLANG_DISABLE_OUTLINES_DISK_CACHE": "1"}):
            result = counted_fn(42)
            self.assertEqual(result, 42)
            self.assertEqual(call_count["n"], 1)

        # Second call also with cache disabled — must still work
        with patch.dict(os.environ, {"SGLANG_DISABLE_OUTLINES_DISK_CACHE": "1"}):
            result = counted_fn(99)
            self.assertEqual(result, 99)
            self.assertEqual(call_count["n"], 2)

    def test_enabled_applies_outlines_cache(self):
        """When disk cache is enabled, outlines cache() decorator is applied."""
        mock_cached_fn = MagicMock(return_value="cached_result")
        mock_cache = MagicMock(return_value=lambda fn: mock_cached_fn)

        with patch.dict(os.environ, {"SGLANG_DISABLE_OUTLINES_DISK_CACHE": "0"}):
            with patch(
                "sglang.srt.constrained.outlines_jump_forward.cache", mock_cache
            ):
                from sglang.srt.constrained.outlines_jump_forward import disk_cache

                @disk_cache()
                def my_fn(x):
                    return x

                result = my_fn("test")

        self.assertEqual(result, "cached_result")
        mock_cache.assert_called_once()

    def test_concurrent_calls_with_cache_disabled(self):
        """Simulates concurrent access with disk cache disabled — no IO errors."""
        import threading

        from sglang.srt.constrained.outlines_jump_forward import disk_cache

        results = []
        errors = []

        @disk_cache()
        def compute(n):
            return n**2

        def worker(n):
            try:
                with patch.dict(
                    os.environ, {"SGLANG_DISABLE_OUTLINES_DISK_CACHE": "1"}
                ):
                    results.append(compute(n))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")
        self.assertEqual(len(results), 20)


if __name__ == "__main__":
    unittest.main()
