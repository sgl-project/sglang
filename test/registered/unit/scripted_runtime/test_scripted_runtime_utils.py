"""Unit tests for scripted_runtime/utils — pure process-boundary helpers, no engine.

Covers ``resolve_fn`` (the "module:qualname" parsing, nested attribute walk, and
the ValueError / TypeError / propagated-import error paths) and
``ensure_script_importable`` (insert-once-at-front sys.path semantics).
"""

from __future__ import annotations

import json
import os
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime.utils import ensure_script_importable, resolve_fn
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestResolveFn(CustomTestCase):
    """resolve_fn turns 'module.path:qualname' into the function object, or rejects it."""

    def test_resolves_top_level_function(self):
        """A 'module:function' path resolves to the function object itself."""
        self.assertIs(resolve_fn("json:dumps"), json.dumps)

    def test_resolves_nested_attribute_path(self):
        """A dotted qualname after the colon walks attributes (os:path.join)."""
        self.assertIs(resolve_fn("os:path.join"), os.path.join)

    def test_rejects_missing_colon(self):
        """A path without a ':' separator is rejected."""
        with self.assertRaisesRegex(ValueError, "module.path:function_name"):
            resolve_fn("json.dumps")

    def test_rejects_empty_module(self):
        """An empty module segment is rejected."""
        with self.assertRaisesRegex(ValueError, "module.path:function_name"):
            resolve_fn(":dumps")

    def test_rejects_empty_function(self):
        """An empty function segment is rejected."""
        with self.assertRaisesRegex(ValueError, "module.path:function_name"):
            resolve_fn("json:")

    def test_rejects_non_callable_target(self):
        """Resolving a non-callable attribute (math.pi) raises TypeError."""
        with self.assertRaisesRegex(TypeError, "not callable"):
            resolve_fn("math:pi")

    def test_propagates_missing_module_error(self):
        """An unimportable module surfaces the underlying import error."""
        with self.assertRaises(ModuleNotFoundError):
            resolve_fn("sglang_no_such_module_zzz:foo")

    def test_propagates_missing_attribute_error(self):
        """A missing attribute on a real module surfaces AttributeError."""
        with self.assertRaises(AttributeError):
            resolve_fn("json:no_such_attribute")


class TestEnsureScriptImportable(CustomTestCase):
    """ensure_script_importable inserts a directory onto sys.path exactly once."""

    _FAKE_ENTRY = "/tmp/__scripted_runtime_ut_fake_sys_path__"

    def setUp(self):
        self._orig_path = list(sys.path)

    def tearDown(self):
        sys.path[:] = self._orig_path

    def test_inserts_new_entry_at_front(self):
        """A previously-absent directory is inserted at the front of sys.path."""
        self.assertNotIn(self._FAKE_ENTRY, sys.path)

        ensure_script_importable(self._FAKE_ENTRY)

        self.assertEqual(sys.path[0], self._FAKE_ENTRY)

    def test_noop_when_entry_is_none(self):
        """A None entry leaves sys.path untouched."""
        ensure_script_importable(None)

        self.assertEqual(sys.path, self._orig_path)

    def test_noop_when_entry_already_present(self):
        """An already-present directory is not inserted a second time."""
        sys.path.insert(0, self._FAKE_ENTRY)

        ensure_script_importable(self._FAKE_ENTRY)

        self.assertEqual(sys.path.count(self._FAKE_ENTRY), 1)


if __name__ == "__main__":
    unittest.main()
