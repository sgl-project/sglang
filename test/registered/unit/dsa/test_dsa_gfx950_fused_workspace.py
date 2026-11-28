"""Workspace bookkeeping: the buffers are scratch, so one per (device, fp8
dtype) serves every layer, and it is only ever allocated up front.  A counting
double stands in for _Workspace, so only the bookkeeping is under test."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.attention.dsa import gfx950_fused_indexer as fd
from sglang.srt.layers.attention.dsa import utils as dsa_utils
from sglang.test.ci.ci_register import register_cpu_ci

# GLM-5.2: 78 target layers + 1 draft layer, each with its own Indexer.
NUM_LAYERS = 79


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class CountingWorkspace:
    made = 0

    def __init__(self, *, device, max_cols, fp8_dtype):
        CountingWorkspace.made += 1
        self.max_cols = max_cols
        self.page_table = None


class TestFusedIndexerHardwareGate(unittest.TestCase):
    """The kernels are raw gfx950 code, so the gate admits exactly that target:
    not another vendor, not an older or newer gfx family."""

    def _supported(self, hip):
        with mock.patch.object(
            fd, "current_arch", return_value=SimpleNamespace(hip=hip)
        ):
            return fd.supported_hardware()

    def test_only_gfx950(self):
        self.assertTrue(self._supported(SimpleNamespace(gcn_arch="gfx950")))
        for gcn_arch in ("gfx942", "gfx1250", "gfx9500"):
            with self.subTest(gcn_arch=gcn_arch):
                self.assertFalse(self._supported(SimpleNamespace(gcn_arch=gcn_arch)))

    def test_not_hip(self):
        self.assertFalse(self._supported(None))


class TestAiterPreshuffleGate(unittest.TestCase):
    """SGLANG_USE_AITER is one switch for every ROCm path: the preshuffle gate
    accepts each spelling the declared variable accepts."""

    def test_use_aiter_spellings(self):
        gate = dsa_utils.aiter_can_use_preshuffle_paged_mqa
        self.addCleanup(gate.cache_clear)
        for value in ("1", "yes"):
            gate.cache_clear()
            env = {
                "SGLANG_USE_AITER": value,
                "SGLANG_DSA_HIP_DISABLE_PRESHUFFLE": "0",
                "AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS": "1",
            }
            with (
                self.subTest(value=value),
                mock.patch.object(dsa_utils, "is_hip", return_value=True),
                mock.patch.dict(os.environ, env),
            ):
                self.assertTrue(gate())


class TestFusedIndexerWorkspace(unittest.TestCase):
    def setUp(self):
        CountingWorkspace.made = 0
        fd._WORKSPACES.clear()
        fd._WARNED_TOO_NARROW = False
        fd._WARNED_NO_WORKSPACE = False
        patch = mock.patch.object(fd, "_Workspace", CountingWorkspace)
        patch.start()
        self.addCleanup(patch.stop)

    def _state(self):
        state = fd.Gfx950FusedIndexer.__new__(fd.Gfx950FusedIndexer)
        state.workspace = None
        return state

    def test_one_allocation_serves_every_layer(self):
        layers = [self._state() for _ in range(NUM_LAYERS)]
        for state in layers:
            fd.prealloc_workspace(device=0, max_cols=4096, fp8_dtype="f8")
            self.assertTrue(
                state.ensure_workspace(device=0, max_cols=4096, fp8_dtype="f8")
            )
        self.assertEqual(CountingWorkspace.made, 1)
        self.assertIs(layers[0].workspace, layers[-1].workspace)

    def test_devices_do_not_share(self):
        fd.prealloc_workspace(device=0, max_cols=4096, fp8_dtype="f8")
        fd.prealloc_workspace(device=1, max_cols=4096, fp8_dtype="f8")
        self.assertEqual(CountingWorkspace.made, 2)

    def test_decode_never_allocates(self):
        """A buffer allocated during graph capture would live in that graph's
        private pool, so the per-call path only binds what prealloc made, and
        says so rather than turning the path off silently."""
        with self.assertLogs(fd.logger, level="WARNING") as cm:
            ok = self._state().ensure_workspace(device=0, max_cols=4096, fp8_dtype="f8")
        self.assertFalse(ok)
        self.assertIn("no workspace was preallocated", "".join(cm.output))
        self.assertEqual(CountingWorkspace.made, 0)

    def test_narrow_workspace_refuses_and_warns(self):
        """The refusal disables the feature for the rest of the run while every
        other log line still reports it as enabled, so it must say so."""
        fd.prealloc_workspace(device=0, max_cols=131072, fp8_dtype="f8")
        with self.assertLogs(fd.logger, level="WARNING") as cm:
            ok = self._state().ensure_workspace(
                device=0, max_cols=1048576, fp8_dtype="f8"
            )
        self.assertFalse(ok)
        self.assertIn("131072", "".join(cm.output))
        self.assertIn("1048576", "".join(cm.output))
        self.assertEqual(CountingWorkspace.made, 1)


if __name__ == "__main__":
    unittest.main()
