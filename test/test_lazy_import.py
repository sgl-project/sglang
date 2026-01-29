"""Tests for lazy import optimization.

This test verifies that the LazyImport class works correctly and
that importing it doesn't load heavy dependencies.
"""

import sys
import unittest


class TestLazyImport(unittest.TestCase):
    """Test suite for LazyImport functionality."""

    def test_lazy_import_module_is_lightweight(self):
        """Test that sglang.lazy doesn't import heavy dependencies."""
        # Unload the module if it was already imported by another test
        if "sglang.lazy" in sys.modules:
            del sys.modules["sglang.lazy"]

        modules_before = set(sys.modules.keys())

        # Import the lazy module
        from sglang.lazy import LazyImport  # noqa: F401

        modules_after = set(sys.modules.keys())
        new_modules = modules_after - modules_before

        # These heavy modules should NOT be loaded
        heavy_modules = ["numpy", "torch", "requests", "IPython", "pydantic", "tqdm"]
        for heavy in heavy_modules:
            loaded_heavy = [m for m in new_modules if m.startswith(heavy)]
            self.assertEqual(
                len(loaded_heavy),
                0,
                f"Heavy module '{heavy}' was loaded when importing LazyImport: {loaded_heavy}",
            )

    def test_lazy_import_delays_loading(self):
        """Test that LazyImport doesn't load the module until accessed."""
        from sglang.lazy import LazyImport

        # Create a lazy import for a module that's unlikely to be pre-loaded
        lazy_obj = LazyImport("decimal", "Decimal")

        # The module should not be loaded yet
        self.assertIsNone(lazy_obj._module)

        # Access the module
        result = lazy_obj("3.14")

        # Now it should be loaded
        self.assertIsNotNone(lazy_obj._module)
        self.assertEqual(str(result), "3.14")

    def test_lazy_import_getattr(self):
        """Test that LazyImport properly delegates attribute access."""
        from sglang.lazy import LazyImport

        # Lazy import the datetime class
        lazy_datetime = LazyImport("datetime", "datetime")

        # Accessing a class method should trigger __getattr__ and load the module
        now = lazy_datetime.now()
        self.assertIsNotNone(now)

        # Verify the module was loaded
        self.assertIsNotNone(lazy_datetime._module)

        # Check that we got a datetime object
        import datetime

        self.assertIsInstance(now, datetime.datetime)

    def test_lazy_import_repr_before_load(self):
        """Test repr before module is loaded."""
        from sglang.lazy import LazyImport

        # Use a module unlikely to be pre-loaded
        lazy_obj = LazyImport("fractions", "Fraction")

        repr_str = repr(lazy_obj)

        self.assertIn("LazyImport", repr_str)
        self.assertIn("not loaded", repr_str)
        self.assertIn("fractions", repr_str)

    def test_lazy_import_repr_after_load(self):
        """Test repr after module is loaded."""
        from sglang.lazy import LazyImport

        lazy_obj = LazyImport("json", "dumps")

        # Trigger loading
        lazy_obj({"test": 1})

        # After loading, repr should show the actual module
        repr_str = repr(lazy_obj)
        self.assertNotIn("not loaded", repr_str)

    def test_backward_compatibility_utils_import(self):
        """Test that LazyImport can still be imported from utils."""
        # This import should work for backward compatibility
        from sglang.utils import LazyImport as LazyImportFromUtils
        from sglang.lazy import LazyImport as LazyImportFromLazy

        # They should be the same class
        self.assertIs(LazyImportFromUtils, LazyImportFromLazy)

    def test_thread_safety(self):
        """Test that LazyImport is thread-safe."""
        import threading
        from sglang.lazy import LazyImport

        lazy_obj = LazyImport("collections", "OrderedDict")
        results = []
        errors = []

        def load_module():
            try:
                # Access the lazy import from multiple threads
                obj = lazy_obj()
                obj["key"] = "value"
                results.append(obj["key"])
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=load_module) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")

        # Verify all threads got the correct result
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r == "value" for r in results))


if __name__ == "__main__":
    unittest.main()
