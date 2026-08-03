"""Unit tests for the MNNVL auto-inference gate.

The TP8 best-throughput launch used to require exporting
``SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE=1`` by hand. It is now
capability-inferred; these cases pin the negative-branch contracts so a
refactor cannot silently turn the predicate into always-true (engaging fabric
paths on non-fabric clusters) or drop the explicit-off override.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HANDLE = ServerArgs._handle_custom_all_reduce_v2_multinode


def _cleared(*fields):
    """Context helper: run with the given env fields unset, restore after."""
    import contextlib
    import os

    @contextlib.contextmanager
    def ctx():
        backup = {f.name: os.environ.pop(f.name, None) for f in fields}
        try:
            yield
        finally:
            for name, val in backup.items():
                if val is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = val

    return ctx()


class TestCaV2MultinodeAuto(CustomTestCase):
    def test_fabric_multinode_auto_enables(self):
        """GB200/GB300 + nnodes>1 + unset opt-in -> multinode mode on, v2 kept."""
        with _cleared(
            envs.SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE,
            envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2,
        ), patch("sglang.srt.server_args.is_mnnvl_fabric_device", return_value=True):
            _HANDLE(SimpleNamespace(nnodes=2, tp_size=8))
            self.assertTrue(envs.SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE.get())
            self.assertTrue(envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2.get())

    def test_non_fabric_multinode_still_disables_v2(self):
        """Non-fabric multi-node keeps the legacy force-disable (the predicate
        must not degrade to always-true)."""
        with _cleared(
            envs.SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE,
            envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2,
        ), patch("sglang.srt.server_args.is_mnnvl_fabric_device", return_value=False):
            _HANDLE(SimpleNamespace(nnodes=2, tp_size=8))
            self.assertFalse(envs.SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE.get())
            self.assertFalse(envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2.get())

    def test_explicit_off_wins_over_fabric(self):
        """SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE=0 on a fabric device
        must still force-disable v2 (explicit off beats auto-detection)."""
        with _cleared(envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2), patch(
            "sglang.srt.server_args.is_mnnvl_fabric_device", return_value=True
        ), envs.SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE.override("0"):
            _HANDLE(SimpleNamespace(nnodes=2, tp_size=8))
            self.assertFalse(envs.SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE.get())
            self.assertFalse(envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2.get())

    def test_tp16_not_auto_opted_in(self):
        """CustomAllReduceV2 supports world sizes 2..8 only; a TP16 fabric
        launch must not auto-set the multinode opt-in (it would log
        'enabling' and then silently fall back downstream)."""
        with _cleared(
            envs.SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE,
            envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2,
        ), patch("sglang.srt.server_args.is_mnnvl_fabric_device", return_value=True):
            _HANDLE(SimpleNamespace(nnodes=2, tp_size=16))
            self.assertFalse(envs.SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE.is_set())


if __name__ == "__main__":
    unittest.main()
