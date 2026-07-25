"""Unit tests for the MNNVL auto-inference gates.

The TP8 best-throughput launch used to require exporting
``SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE=1`` and
``SGLANG_K3_AR_FUSION=1`` by hand. Both are now capability-inferred; these
cases pin the negative-branch contracts so a refactor cannot silently turn
either predicate into always-true (engaging fabric paths on non-fabric
clusters) or drop the explicit-off override.
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

    def test_single_node_untouched(self):
        """nnodes=1 must not set the multinode opt-in even on fabric devices."""
        with _cleared(
            envs.SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE,
            envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2,
        ), patch("sglang.srt.server_args.is_mnnvl_fabric_device", return_value=True):
            _HANDLE(SimpleNamespace(nnodes=1, tp_size=8))
            self.assertFalse(envs.SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE.is_set())


class TestK3ArFusionGate(CustomTestCase):
    def _reset(self):
        import sglang.srt.layers.k3_ar_fusion as mod

        mod._STATE = None
        mod._INITIALIZED = False
        return mod

    def test_explicit_off_wins(self):
        """SGLANG_K3_AR_FUSION=0 disables the fusion before any capability
        probe (no distributed state is touched)."""
        mod = self._reset()
        with envs.SGLANG_K3_AR_FUSION.override("0"):
            self.assertFalse(mod.enabled())

    def test_auto_requires_sm100(self):
        """Unset env auto-probes only on SM100/SM103; other arches stay on
        the regular all-reduce path."""
        mod = self._reset()
        with _cleared(envs.SGLANG_K3_AR_FUSION), patch(
            "sglang.srt.utils.common.get_device_sm", return_value=90
        ):
            self.assertFalse(mod.enabled())

    def test_auto_skips_symm_mem(self):
        """Bug regression: the auto-probe engaged the fusion under
        --enable-symm-mem, where the pynccl allocator context misroutes
        the o_proj/MoE outputs away from the k3 symm pool and the pull path's
        symm-pool assertion kills the server at graph-capture warmup (hit on
        both plain TP8 and DCP8 launches). Unset env + symm-mem must stay on
        the regular all-reduce path regardless of dcp_size (explicit
        SGLANG_K3_AR_FUSION=1 still force-attempts). DCP without symm-mem is
        inside the validated envelope and is NOT gated off (DCP8 GB300:
        GSM8K in-band, bs=1 +19%)."""
        for symm, dcp, a2a in (
            (True, 1, "none"),
            (True, 8, "none"),
            # Bug regression (EP): under EP a2a (megamoe/deepep) the model's
            # symm-pool allocation contract does not hold on every AR
            # call-site -> the same _find_mc_ptr assertion killed an EP8
            # megamoe launch at warmup on head 480fe4e76.
            (False, 1, "megamoe"),
            (False, 1, "deepep"),
        ):
            mod = self._reset()
            with _cleared(envs.SGLANG_K3_AR_FUSION), patch(
                "sglang.srt.utils.common.get_device_sm", return_value=103
            ), patch(
                "sglang.srt.runtime_context.get_server_args",
                return_value=SimpleNamespace(
                    enable_symm_mem=symm, dcp_size=dcp, moe_a2a_backend=a2a
                ),
            ):
                self.assertFalse(mod.enabled(), f"symm={symm} dcp={dcp} a2a={a2a}")


if __name__ == "__main__":
    unittest.main()
