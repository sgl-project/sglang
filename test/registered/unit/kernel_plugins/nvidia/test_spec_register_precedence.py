# SPDX-License-Identifier: Apache-2.0
"""Pure-Python contracts for megakernel spec registration precedence.

The registry contract lives in a module that imports torch/msgspec at import
time. This test injects minimal stubs when those are absent and loads the
contract by file path (isolated module name), so the precedence rules can be
verified on a CPU-only box with nothing but the standard library::

    python3 test/registered/unit/kernel_plugins/nvidia/test_spec_register_precedence.py

It fixes the regression where a built-in spec silently swallowed an external
registration of the same architecture. Built-in ownership stays the intentional
default (startup must not break), but an ineffective registration is now
observable: register() returns a bool and emits a diagnostic warning, and only
an explicit ``override=True`` replaces the owner.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _stub_missing_dependencies() -> None:
    """Only inject stubs when the real packages are unavailable."""
    try:
        import msgspec  # noqa: F401
    except ImportError:
        # spec.py derives many frozen structs from msgspec.Struct. A base class
        # that tolerates the ``frozen=True`` class keyword is enough to import.
        class _Struct:
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__()

        stub = types.ModuleType("msgspec")
        stub.Struct = _Struct
        sys.modules["msgspec"] = stub

    try:
        import torch  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch")
        stub.Tensor = object
        stub.device = object
        stub.Generator = object
        stub.nn = SimpleNamespace(Module=object)
        sys.modules["torch"] = stub


def _load_registry():
    """Load spec.py by path so no sglang package __init__ (or torch) is needed."""
    _stub_missing_dependencies()
    relative = Path("python/sglang/kernels/plugin_kernels/nvidia/spec.py")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.is_file():
            module_spec = importlib.util.spec_from_file_location(
                "_nvidia_spec_register_contract", str(candidate)
            )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"cannot locate {relative} from {__file__}")


# Register with the CI matrix when the harness is importable; a no-op locally.
try:
    from sglang.test.ci.ci_register import register_cpu_ci
except Exception:  # no package / no torch on a bare CPU box
    pass
else:
    register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_REGISTRY = _load_registry()
KIMI_ARCH = "KimiK3ForConditionalGeneration"
BUILTIN_PATH = _REGISTRY._BUILTIN_SPECS[KIMI_ARCH]
EXTERNAL_PATH = "sgl_nvidia_mega.kimi_k3.spec:SPEC"


class TestSpecRegisterPrecedence(unittest.TestCase):
    def setUp(self):
        self.r = _REGISTRY
        self.r._REGISTRY.clear()
        self.r._OVERRIDES.clear()

    def tearDown(self):
        self.r._REGISTRY.clear()
        self.r._OVERRIDES.clear()

    def _override(self, architecture, path):
        with patch.object(self.r, "logger") as logger:
            result = self.r.register(
                architecture=architecture, path=path, override=True
            )
        return result, logger

    def _register_builtin(self, path):
        """The external entry point's default call for a built-in architecture."""
        with patch.object(self.r, "logger") as logger:
            result = self.r.register(architecture=KIMI_ARCH, path=path)
        return result, logger

    def _resolved_why(self, architectures, isa="sm_103"):
        """The path find_spec() actually tried, via its rejection reason.

        A non-MegakernelSpec sentinel makes find_spec() report the resolved
        "module:attribute" path without needing to construct a real spec (which
        msgspec.Struct would not allow with these stubs).
        """
        with patch.object(
            self.r.importlib,
            "import_module",
            return_value=SimpleNamespace(SPEC=object()),
        ) as import_module:
            _, why = self.r.find_spec(architectures=architectures, isa=isa)
        return why, import_module

    # 1. a first registration succeeds -------------------------------------
    def test_first_registration_succeeds(self):
        self.assertTrue(self.r.register(architecture="ArchA", path="ext.a:SPEC"))
        self.assertEqual(self.r._REGISTRY["ArchA"], "ext.a:SPEC")
        self.assertTrue(self.r.spec_registered(["ArchA"]))
        self.assertIn("ArchA", self.r.registered_architectures())

    # 2. a conflicting external re-registration still raises ---------------
    def test_duplicate_external_registration_still_raises(self):
        self.r.register(architecture="ArchA", path="ext.a:SPEC")
        with self.assertRaises(ValueError) as caught:
            self.r.register(architecture="ArchA", path="ext.b:SPEC")
        message = str(caught.exception)
        self.assertIn("already registered", message)
        self.assertIn("ext.a:SPEC", message)  # who owns the name
        self.assertIn("ext.b:SPEC", message)  # what was rejected
        self.assertIn("override=True", message)
        self.assertEqual(self.r._REGISTRY["ArchA"], "ext.a:SPEC")

    def test_identical_reregistration_is_idempotent(self):
        """Re-registering the same path stays a no-op (entry points load twice)."""
        self.assertTrue(self.r.register(architecture="ArchA", path="ext.a:SPEC"))
        self.assertTrue(self.r.register(architecture="ArchA", path="ext.a:SPEC"))
        self.assertEqual(self.r._REGISTRY["ArchA"], "ext.a:SPEC")
        self.assertEqual(self.r._OVERRIDES, {})

    # 3. override=True replaces an external registration and takes effect ---
    def test_override_replaces_external_registration(self):
        self.r.register(architecture="ArchA", path="ext.a:SPEC")
        result, _ = self._override("ArchA", "ext.c:SPEC")
        self.assertTrue(result)
        self.assertEqual(self.r._OVERRIDES["ArchA"], "ext.c:SPEC")
        why, import_module = self._resolved_why(["ArchA"])
        self.assertIn("ext.c:SPEC", why)
        self.assertNotIn("ext.a:SPEC", why)
        import_module.assert_called_once_with("ext.c")

    def test_override_is_logged_not_silent(self):
        self.r.register(architecture="ArchA", path="ext.a:SPEC")
        _, logger = self._override("ArchA", "ext.c:SPEC")
        logger.warning.assert_called_once()
        message = " ".join(str(a) for a in logger.warning.call_args.args)
        self.assertIn("ArchA", message)
        self.assertIn("ext.c:SPEC", message)

    # 4. builtin collisions: default stays ineffective but is diagnosable --
    def test_builtin_registration_is_ineffective_but_diagnosable(self):
        """No exception at startup; the caller can tell it did not take effect."""
        result, logger = self._register_builtin(EXTERNAL_PATH)
        self.assertFalse(result)  # explicit "did not take effect" status
        logger.warning.assert_called_once()
        message = " ".join(str(a) for a in logger.warning.call_args.args)
        self.assertIn("did NOT take effect", message)
        self.assertIn(BUILTIN_PATH, message)  # which built-in owns the name
        self.assertIn(EXTERNAL_PATH, message)  # what was rejected
        self.assertIn("override=True", message)  # how to replace explicitly
        self.assertNotIn(KIMI_ARCH, self.r._REGISTRY)
        self.assertNotIn(KIMI_ARCH, self.r._OVERRIDES)

    def test_identical_builtin_path_registration_still_warns(self):
        """The old fully-silent return for an identical path now has a signal."""
        result, logger = self._register_builtin(BUILTIN_PATH)
        self.assertFalse(result)
        logger.warning.assert_called_once()
        message = " ".join(str(a) for a in logger.warning.call_args.args)
        self.assertIn(BUILTIN_PATH, message)
        self.assertIn("override=True", message)

    def test_default_startup_registration_raises_nothing(self):
        """The external entry point must never break startup by default."""
        try:
            for path in (EXTERNAL_PATH, BUILTIN_PATH, EXTERNAL_PATH):
                with patch.object(self.r, "logger"):
                    self.r.register(architecture=KIMI_ARCH, path=path)
        except Exception as exc:  # pragma: no cover - the regression guard
            self.fail(f"default registration raised {type(exc).__name__}: {exc}")
        self.assertEqual(self.r._REGISTRY, {})
        self.assertEqual(self.r._OVERRIDES, {})

    # 5. find_spec(): built-in priority unchanged, override is the exception -
    def test_builtin_priority_is_unchanged_in_find_spec(self):
        """A built-in owns the model even when listed after another architecture."""
        self.r.register(architecture="ArchA", path="ext.a:SPEC")
        why, import_module = self._resolved_why(["ArchA", KIMI_ARCH])
        self.assertIn(BUILTIN_PATH, why)
        self.assertNotIn("ext.a:SPEC", why)
        import_module.assert_called_once_with(BUILTIN_PATH.split(":")[0])

    def test_override_replaces_the_builtin_in_find_spec(self):
        result, _ = self._override(KIMI_ARCH, EXTERNAL_PATH)
        self.assertTrue(result)
        why, import_module = self._resolved_why([KIMI_ARCH])
        self.assertIn(EXTERNAL_PATH, why)
        self.assertNotIn(BUILTIN_PATH, why)
        import_module.assert_called_once_with("sgl_nvidia_mega.kimi_k3.spec")

    def test_builtin_predicates_still_hold_after_an_override(self):
        """builtin vs external detection is preserved, not collapsed."""
        self.assertTrue(self.r.builtin_spec_registered([KIMI_ARCH]))
        self.assertTrue(self.r.spec_registered([KIMI_ARCH]))
        self._override(KIMI_ARCH, EXTERNAL_PATH)
        self.assertTrue(self.r.builtin_spec_registered([KIMI_ARCH]))
        self.assertTrue(self.r.spec_registered([KIMI_ARCH]))
        self.assertIn(KIMI_ARCH, self.r.registered_architectures())
        self.assertFalse(self.r.builtin_spec_registered(["ArchA"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
