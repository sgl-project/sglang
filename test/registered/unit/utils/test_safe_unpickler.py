"""Unit tests for SafeUnpickler builtins hardening — no server, no model loading.

Covers GHSA-h74r-pwx2-6qr2: the builtins deny-list missed getattr/__import__,
so a pickle could chain __import__("os") + getattr(os, "system") to RCE even
though ("os", "system") is denied.
"""

import pickle
import unittest

from sglang.srt.utils.common import safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class _Reduce:
    """Pickles as callable(*args) so we can emit a GLOBAL for any builtin."""

    def __init__(self, func, args):
        self._func = func
        self._args = args

    def __reduce__(self):
        return (self._func, self._args)


class TestSafeUnpickler(CustomTestCase):
    def _blocked(self, func, args):
        payload = pickle.dumps(_Reduce(func, args))
        with self.assertRaises(RuntimeError):
            safe_pickle_loads(payload)

    def test_import_blocked(self):
        # __import__("os") is the first half of the RCE chain
        self._blocked(__import__, ("os",))

    def test_getattr_blocked(self):
        # getattr(os, "system") is the second half
        self._blocked(getattr, (object(), "__class__"))

    def test_dangerous_builtins_blocked(self):
        for func, args in [
            (setattr, (object(), "x", 1)),
            (globals, ()),
            (open, ("/etc/passwd",)),
        ]:
            self._blocked(func, args)

    def test_benign_payload_still_loads(self):
        for obj in [{"a": 1}, [1, 2, 3], (1, "x"), b"bytes", {1, 2}]:
            self.assertEqual(safe_pickle_loads(pickle.dumps(obj)), obj)


if __name__ == "__main__":
    unittest.main()
