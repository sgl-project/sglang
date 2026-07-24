"""Unit tests for CompilationConfig and register_split_op in srt/compilation/compilation_config.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import sglang.srt.compilation.compilation_config as _config_mod
from sglang.srt.compilation.compilation_config import (
    CompilationConfig,
    register_split_op,
)
from sglang.test.test_utils import CustomTestCase


class TestRegisterSplitOp(CustomTestCase):
    def setUp(self):
        # Save SPLIT_OPS contents; restore after each test to avoid pollution.
        self._orig_ops = list(_config_mod.SPLIT_OPS)

    def tearDown(self):
        _config_mod.SPLIT_OPS[:] = self._orig_ops

    def test_explicit_op_name_appends_prefixed_string(self):
        @register_split_op(op_name="my_custom_op")
        def some_func():
            pass

        self.assertIn("sglang.my_custom_op", _config_mod.SPLIT_OPS)

    def test_no_op_name_uses_function_dunder_name(self):
        @register_split_op()
        def unique_fn_for_test():
            pass

        self.assertIn("sglang.unique_fn_for_test", _config_mod.SPLIT_OPS)

    def test_decorated_function_is_still_callable(self):
        sentinel = []

        @register_split_op(op_name="sentinel_op")
        def push_sentinel():
            sentinel.append(1)

        push_sentinel()
        self.assertEqual(sentinel, [1])

    def test_decorated_function_returns_original(self):
        def original():
            return 42

        decorated = register_split_op(op_name="ret_op")(original)
        self.assertIs(decorated, original)

    def test_multiple_registrations_accumulate(self):
        before = len(_config_mod.SPLIT_OPS)

        @register_split_op(op_name="op_alpha")
        def fa():
            pass

        @register_split_op(op_name="op_beta")
        def fb():
            pass

        self.assertEqual(len(_config_mod.SPLIT_OPS), before + 2)
        self.assertIn("sglang.op_alpha", _config_mod.SPLIT_OPS)
        self.assertIn("sglang.op_beta", _config_mod.SPLIT_OPS)


class TestCompilationConfigConstruction(CustomTestCase):
    def setUp(self):
        self._orig_ops = list(_config_mod.SPLIT_OPS)

    def tearDown(self):
        _config_mod.SPLIT_OPS[:] = self._orig_ops

    def test_capture_sizes_stored(self):
        cfg = CompilationConfig(capture_sizes=[1, 4, 8])
        self.assertEqual(cfg.get_capture_sizes(), [1, 4, 8])

    def test_default_compiler_is_eager(self):
        cfg = CompilationConfig(capture_sizes=[1])
        self.assertEqual(cfg.compiler, "eager")

    def test_custom_compiler_stored(self):
        cfg = CompilationConfig(capture_sizes=[1], compiler="eager")
        self.assertEqual(cfg.compiler, "eager")

    def test_enable_debug_mode_default_is_false(self):
        cfg = CompilationConfig(capture_sizes=[1])
        self.assertFalse(cfg.get_enable_debug_mode())

    def test_enable_debug_mode_can_be_set(self):
        cfg = CompilationConfig(capture_sizes=[1], enable_debug_mode=True)
        self.assertTrue(cfg.get_enable_debug_mode())

    def test_split_ops_populated_from_global_split_ops(self):
        @register_split_op(op_name="injected_op")
        def _dummy():
            pass

        cfg = CompilationConfig(capture_sizes=[1])
        self.assertIn("sglang.injected_op", cfg.split_ops)

    def test_split_ops_is_independent_copy_from_global(self):
        cfg = CompilationConfig(capture_sizes=[1])
        original_len = len(cfg.split_ops)
        # Mutating the global after construction must not affect the config.
        _config_mod.SPLIT_OPS.append("sglang.post_construction_op")
        self.assertEqual(len(cfg.split_ops), original_len)

    def test_configure_inductor_noop_for_eager_compiler(self):
        # Should not raise or import inductor config.
        cfg = CompilationConfig(capture_sizes=[1], compiler="eager")
        self.assertEqual(cfg.compiler, "eager")


class TestCompilationConfigMutators(CustomTestCase):
    def test_add_split_op_appends_to_instance_split_ops(self):
        cfg = CompilationConfig(capture_sizes=[1])
        cfg.add_split_op("my.op")
        self.assertIn("my.op", cfg.split_ops)

    def test_add_split_op_does_not_affect_global_split_ops(self):
        cfg = CompilationConfig(capture_sizes=[1])
        before = list(_config_mod.SPLIT_OPS)
        cfg.add_split_op("instance.only.op")
        self.assertEqual(_config_mod.SPLIT_OPS, before)

    def test_add_traced_file_stored_in_set(self):
        cfg = CompilationConfig(capture_sizes=[1])
        cfg.add_traced_file("/some/file.py")
        self.assertIn("/some/file.py", cfg.get_traced_files())

    def test_add_traced_file_deduplicates(self):
        cfg = CompilationConfig(capture_sizes=[1])
        cfg.add_traced_file("/path/a.py")
        cfg.add_traced_file("/path/a.py")
        self.assertEqual(len(cfg.get_traced_files()), 1)

    def test_get_traced_files_empty_initially(self):
        cfg = CompilationConfig(capture_sizes=[1])
        self.assertEqual(len(cfg.get_traced_files()), 0)

    def test_multiple_traced_files_all_stored(self):
        cfg = CompilationConfig(capture_sizes=[1])
        cfg.add_traced_file("/a.py")
        cfg.add_traced_file("/b.py")
        self.assertIn("/a.py", cfg.get_traced_files())
        self.assertIn("/b.py", cfg.get_traced_files())
        self.assertEqual(len(cfg.get_traced_files()), 2)


if __name__ == "__main__":
    unittest.main()
