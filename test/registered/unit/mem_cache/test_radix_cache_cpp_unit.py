"""Unit tests for fail-closed C++ radix-cache request validation."""

import importlib
import sys
import types
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRadixCacheCppCacheSalt(CustomTestCase):
    def test_cache_salt_is_rejected_without_loading_cpp_extension(self):
        extension_name = "sglang.srt.mem_cache.cpp_radix_tree.radix_tree"
        module_name = "sglang.srt.mem_cache.radix_cache_cpp"
        fake_extension = types.ModuleType(extension_name)
        fake_extension.IOHandle = object
        fake_extension.RadixTreeCpp = object
        fake_extension.TreeNodeCpp = object

        original_module = sys.modules.pop(module_name, None)
        try:
            with patch.dict(sys.modules, {extension_name: fake_extension}):
                module = importlib.import_module(module_name)
                module.RadixCacheCpp._reject_cache_salt(None)
                with self.assertRaisesRegex(ValueError, "experimental C\\+\\+"):
                    module.RadixCacheCpp._reject_cache_salt("tenant-a")
        finally:
            sys.modules.pop(module_name, None)
            if original_module is not None:
                sys.modules[module_name] = original_module


if __name__ == "__main__":
    unittest.main()
