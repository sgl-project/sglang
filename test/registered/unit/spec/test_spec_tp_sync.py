"""Unit tests for speculative TP sync, including the CUDA-graph memory probe."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSpecTpSyncAvailableMemory(CustomTestCase):
    def _probe(self, sync, group):
        with patch(
            "sglang.srt.speculative.spec_tp_sync.get_available_gpu_memory",
            return_value=2.0,
        ) as get_memory:
            sync.available_memory_gb(SpecTpSyncSite.DSPARK_MEM, "cuda", 0, group=group)
        return get_memory.call_args.kwargs

    def test_memory_probe_reduces_over_override_group_when_attn_tp_is_1(self):
        """DSpark MoE draft under DP attention: attn_tp=1, full TP>1.

        Spec broadcasts stay on the singleton attn-TP group, but CUDA graph
        capture uses the full TP group. The memory probe must min-reduce over
        that capture group so ranks agree on whether to enter capture.
        """
        attention_tp1 = SimpleNamespace(world_size=1)
        full_tp2 = SimpleNamespace(world_size=2, cpu_group=object())
        kwargs = self._probe(SpecTpSync(attention_tp1), full_tp2)
        self.assertTrue(kwargs["distributed"])
        self.assertIs(kwargs["cpu_group"], full_tp2.cpu_group)

    def test_memory_probe_skips_reduction_when_site_is_disabled(self):
        attention_tp1 = SimpleNamespace(world_size=1)
        full_tp2 = SimpleNamespace(world_size=2, cpu_group=object())
        with envs.SGLANG_SPEC_TP_SYNC.override("off"):
            kwargs = self._probe(SpecTpSync(attention_tp1), full_tp2)
        self.assertFalse(kwargs["distributed"])
        self.assertIsNone(kwargs["cpu_group"])

    def test_memory_probe_reduces_when_constructor_group_is_already_multi_rank(self):
        tp2 = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=object())
        kwargs = self._probe(SpecTpSync(tp2), tp2)
        self.assertTrue(kwargs["distributed"])
        self.assertIs(kwargs["cpu_group"], tp2.cpu_group)

    def test_singleton_constructor_still_skips_runtime_broadcast(self):
        attention_tp1 = SimpleNamespace(world_size=1, broadcast=MagicMock())
        sync = SpecTpSync(attention_tp1)
        sync.sync(SpecTpSyncSite.DSPARK_TARGET, MagicMock())
        attention_tp1.broadcast.assert_not_called()
        self.assertFalse(sync.enabled(SpecTpSyncSite.DSPARK_MEM))
        self.assertFalse(sync.enabled(SpecTpSyncSite.DSPARK_TARGET))


if __name__ == "__main__":
    unittest.main()
