"""CPU tests for the Qwen MoE fused finalize + shared-expert gate token cap."""

import importlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.environ import envs
from sglang.srt.models import qwen2_moe
from sglang.test.test_utils import CustomTestCase


class TestFusedFinalizeGateCap(CustomTestCase):
    def tearDown(self):
        importlib.reload(qwen2_moe)

    def test_cap_defaults_to_the_small_grid_range_and_follows_the_env(self):
        field = envs.SGLANG_QWEN_MOE_FUSED_FINALIZE_GATE_MAX_TOKENS
        with field.override(""):
            field.clear()
            importlib.reload(qwen2_moe)
            self.assertEqual(qwen2_moe._FUSED_FINALIZE_GATE_MAX_TOKENS, 64)
        with field.override(576):
            importlib.reload(qwen2_moe)
            self.assertEqual(qwen2_moe._FUSED_FINALIZE_GATE_MAX_TOKENS, 576)
        with field.override(0):
            with self.assertRaises(ValueError):
                importlib.reload(qwen2_moe)


if __name__ == "__main__":
    unittest.main()
