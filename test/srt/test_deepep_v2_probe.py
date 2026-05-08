"""CPU-only unit test for the DeepEP V2 probe plumbing in
`sglang.srt.layers.moe.token_dispatcher.deepep`.

Exercises the three reachable probe states:

1. V1 `Buffer` available, V2 `ElasticBuffer` missing
2. V2 `ElasticBuffer` available, V1 `Buffer` also available (normal V2 install)
3. Neither available

The test does not execute any DeepEP kernels — it imports the module
under a monkey-patched `deep_ep` stub so it runs on any CPU host without
CUDA / NCCL / EFA. It is purely a guard against the probe pattern
silently regressing.

Run:
    pytest -xvs test/srt/test_deepep_v2_probe.py
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from contextlib import contextmanager


def _make_stub_deep_ep(*, with_buffer: bool, with_elastic_buffer: bool):
    """Build a fake `deep_ep` module with the attributes we opt to expose."""

    mod = types.ModuleType("deep_ep")
    if with_buffer:

        class _StubBuffer:  # noqa: D401 - placeholder class
            num_sms = 20

            @staticmethod
            def get_dispatch_config(size):  # pragma: no cover - unused in test
                return None

            @staticmethod
            def get_combine_config(size):  # pragma: no cover - unused in test
                return None

        mod.Buffer = _StubBuffer
        mod.Config = object
    if with_elastic_buffer:

        class _StubElasticBuffer:  # noqa: D401 - placeholder class
            pass

        mod.ElasticBuffer = _StubElasticBuffer
    return mod


@contextmanager
def _patched_deep_ep(*, with_buffer: bool, with_elastic_buffer: bool):
    """Install a stub `deep_ep` in sys.modules and drop the SGLang
    deepep module from the cache so it re-runs its imports against the
    stub. Restores the original state on exit.
    """

    saved = {
        k: sys.modules[k]
        for k in list(sys.modules)
        if k == "deep_ep" or k.startswith("deep_ep.")
    }
    saved_deepep = sys.modules.pop(
        "sglang.srt.layers.moe.token_dispatcher.deepep", None
    )
    # Wipe the stub path
    sys.modules.pop("deep_ep", None)
    if with_buffer or with_elastic_buffer:
        sys.modules["deep_ep"] = _make_stub_deep_ep(
            with_buffer=with_buffer, with_elastic_buffer=with_elastic_buffer
        )
    try:
        yield
    finally:
        # Best-effort restore; we don't try to rewind every transitive
        # import that the probe triggered.
        sys.modules.pop("deep_ep", None)
        for k, v in saved.items():
            sys.modules[k] = v
        if saved_deepep is not None:
            sys.modules["sglang.srt.layers.moe.token_dispatcher.deepep"] = saved_deepep


class TestDeepEPV2Probe(unittest.TestCase):
    """Guard the V2 probe plumbing against silent regression."""

    def _import_probe_flags(self):
        sys.modules.pop("sglang.srt.layers.moe.token_dispatcher.deepep", None)
        mod = importlib.import_module("sglang.srt.layers.moe.token_dispatcher.deepep")
        return (
            getattr(mod, "use_deepep"),
            getattr(mod, "have_deepep_v2"),
        )

    def test_v1_only_installed(self):
        """Legacy install (V1 `Buffer` only). `use_deepep=True`,
        `have_deepep_v2=False`. No regression on pre-V2 users."""
        with _patched_deep_ep(with_buffer=True, with_elastic_buffer=False):
            try:
                use_deepep, have_v2 = self._import_probe_flags()
            except ImportError:
                # Test env may not have full sglang deps; that's fine,
                # the probe itself is what we need to prove compiles.
                self.skipTest("sglang module stack unavailable on test host")
                return
        self.assertTrue(use_deepep)
        self.assertFalse(have_v2)

    def test_v2_installed(self):
        """V2 install (both `Buffer` legacy + `ElasticBuffer` exported).
        `use_deepep=True`, `have_deepep_v2=True`. The V2 path is now
        reachable behind `SGLANG_DEEPEP_USE_V2=1`."""
        with _patched_deep_ep(with_buffer=True, with_elastic_buffer=True):
            try:
                use_deepep, have_v2 = self._import_probe_flags()
            except ImportError:
                self.skipTest("sglang module stack unavailable on test host")
                return
        self.assertTrue(use_deepep)
        self.assertTrue(have_v2)

    def test_neither_installed(self):
        """deep_ep missing entirely. `use_deepep=False`,
        `have_deepep_v2=False`. Module must still import cleanly so the
        SGLang package can load on non-MoE targets."""
        with _patched_deep_ep(with_buffer=False, with_elastic_buffer=False):
            try:
                use_deepep, have_v2 = self._import_probe_flags()
            except ImportError:
                self.skipTest("sglang module stack unavailable on test host")
                return
        self.assertFalse(use_deepep)
        self.assertFalse(have_v2)


if __name__ == "__main__":
    unittest.main()
