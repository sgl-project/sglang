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
        # Clear any cached imports
        modules_before = set(sys.modules.keys())

        # Import the lazy module
        from sglang.lazy import LazyImport

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

        # Create a lazy import
        lazy_obj = LazyImport("json", "dumps")

        # The module should not be loaded yet
        self.assertIsNone(lazy_obj._module)

        # Access the module
        result = lazy_obj({"key": "value"})

        # Now it should be loaded
        self.assertIsNotNone(lazy_obj._module)
        self.assertEqual(result, '{"key": "value"}')

    def test_lazy_import_getattr(self):
        """Test that LazyImport properly delegates attribute access."""
        from sglang.lazy import LazyImport

        # Use a standard library module for testing
        lazy_path = LazyImport("os.path", "join")

        # __name__ should trigger loading and attribute access
        result = lazy_path("a", "b", "c")
        self.assertIn("b", result)

    def test_lazy_import_repr_before_load(self):
        """Test repr before module is loaded."""
        from sglang.lazy import LazyImport

        lazy_obj = LazyImport("json", "dumps")
        repr_str = repr(lazy_obj)

        self.assertIn("LazyImport", repr_str)
        self.assertIn("not loaded", repr_str)
        self.assertIn("json", repr_str)

    def test_backward_compatibility_utils_import(self):
        """Test that LazyImport can still be imported from utils."""
        # This import should work for backward compatibility
        from sglang.utils import LazyImport as LazyImportFromUtils
        from sglang.lazy import LazyImport as LazyImportFromLazy

        # They should be the same class
        self.assertIs(LazyImportFromUtils, LazyImportFromLazy)


if __name__ == "__main__":
    unittest.main()
