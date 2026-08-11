"""Unit tests for SafeUnpickler exact-name allowlist hardening.

Covers CVE-2026-15969 (GHSA-h74r-pwx2-6qr2): the previous prefix-based
allowlist let an attacker chain builtins.getattr + builtins.__import__ (or
operator.attrgetter + pickletools.sys) into os.system() even though
("os", "system") was on the deny-list.
"""

import io
import operator
import pickle
import unittest

from sglang.srt.utils.common import SafeUnpickler, safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Reduce:
    """Pickles as callable(*args) so we can emit a GLOBAL for any callable."""

    def __init__(self, func, args):
        self._func = func
        self._args = args

    def __reduce__(self):
        return (self._func, self._args)


def _pglob(module, name):
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def _pstr(s):
    b = s.encode()
    return b"\x8c" + bytes([len(b)]) + b


_PROTO = b"\x80\x04"
_T1, _T2, _REDUCE, _STOP = b"\x85", b"\x86", b"R", b"."


def _build_rce_chain(cmd):
    """Full CVE-2026-15969 chain: getattr(__import__('os'), 'system')(cmd)."""
    return (
        _PROTO
        + _pglob("builtins", "getattr")
        + _pglob("builtins", "__import__")
        + _pstr("os") + _T1 + _REDUCE
        + _pstr("system") + _T2 + _REDUCE
        + _pstr(cmd) + _T1 + _REDUCE
        + _STOP
    )


def _build_operator_bypass_chain(cmd):
    """operator.attrgetter/itemgetter + pickletools.sys chain."""
    return (
        _PROTO
        + _pglob("operator", "attrgetter") + _pstr("system") + _T1 + _REDUCE
        + _pglob("operator", "itemgetter") + _pstr("os") + _T1 + _REDUCE
        + _pglob("operator", "attrgetter") + _pstr("modules") + _T1 + _REDUCE
        + _pglob("pickletools", "sys")
        + _T1 + _REDUCE
        + _T1 + _REDUCE
        + _T1 + _REDUCE
        + _pstr(cmd) + _T1 + _REDUCE
        + _STOP
    )


class TestSafeUnpickler(CustomTestCase):
    def _blocked(self, func, args):
        payload = pickle.dumps(_Reduce(func, args))
        with self.assertRaises(RuntimeError):
            safe_pickle_loads(payload)

    def test_import_blocked(self):
        # __import__("os") is the first half of the RCE chain
        self._blocked(__import__, ("os",))

    def test_getattr_blocked(self):
        # getattr(os, "system") is the second half of the RCE chain
        self._blocked(getattr, (object(), "__class__"))

    def test_operator_primitives_blocked(self):
        # operator.attrgetter / itemgetter / getitem bypass primitives
        self._blocked(operator.attrgetter, ("system",))
        self._blocked(operator.itemgetter, ("os",))
        self._blocked(operator.getitem, (dict(), "k"))

    def test_pickletools_blocked(self):
        with self.assertRaises(RuntimeError):
            safe_pickle_loads(_pglob("pickletools", "sys") + _STOP)

    def test_dangerous_builtins_blocked(self):
        for func, args in [
            (setattr, (object(), "x", 1)),
            (eval, ("1",)),
            (exec, ("pass",)),
            (open, ("/etc/passwd",)),
        ]:
            self._blocked(func, args)

    def test_full_rce_chain_blocked(self):
        with self.assertRaises(RuntimeError):
            SafeUnpickler(
                io.BytesIO(_build_rce_chain("touch /tmp/sglang_pwned"))
            ).load()

    def test_full_operator_bypass_chain_blocked(self):
        with self.assertRaises(RuntimeError):
            SafeUnpickler(
                io.BytesIO(_build_operator_bypass_chain("touch /tmp/sglang_pwned2"))
            ).load()

    def test_benign_payloads_still_load(self):
        for obj in [
            {"a": 1, "b": [1, 2, 3]},
            [1, 2, 3],
            (1, "x"),
            b"bytes",
            {1, 2, 3},
            frozenset({1, 2}),
            "hello",
            42,
            3.14,
            True,
        ]:
            self.assertEqual(safe_pickle_loads(pickle.dumps(obj)), obj)

    def test_tensor_roundtrip(self):
        import torch

        t = torch.zeros(2, 3)
        restored = safe_pickle_loads(pickle.dumps(t))
        torch.testing.assert_close(restored, t)


if __name__ == "__main__":
    unittest.main()
