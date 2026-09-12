"""``DeepseekV2MoE._fuse_shared_into_reduce`` must treat a missing ``shared_experts`` (the shared expert fused into the routed kernel) as no shared expert instead of raising on the first prefill."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _moe_state(**extra) -> SimpleNamespace:
    """The attributes the gate reads, as ``DeepseekV2MoE.__init__`` leaves them when the
    shared expert is fused into the routed kernel (no ``shared_experts`` module)."""
    return SimpleNamespace(
        _shared_expert_tp1=False, _fuse_shared_experts_inside_sbo=False, **extra
    )


class TestFuseSharedIntoReduceGate(CustomTestCase):
    def setUp(self):
        super().setUp()
        cm = envs.SGLANG_OPT_HIP_FUSED_MOE_REDUCE_ADD.override(True)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        patcher = mock.patch.object(deepseek_v2, "_use_aiter", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_shared_experts_is_false(self):
        gate = DeepseekV2MoE._fuse_shared_into_reduce
        self.assertFalse(gate(_moe_state(), False, 4))

    def test_present_shared_experts_is_true(self):
        gate = DeepseekV2MoE._fuse_shared_into_reduce
        state = _moe_state(shared_experts=object())
        self.assertTrue(gate(state, False, 4))
        self.assertFalse(gate(state, True, 4))
        self.assertFalse(gate(state, False, 0))
        self.assertFalse(gate(_moe_state(shared_experts=None), False, 4))


if __name__ == "__main__":
    unittest.main()
