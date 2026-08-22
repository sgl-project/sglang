"""Unit tests for the DeepSeek-V4 two-batch-overlap policy."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV4TboPolicy(CustomTestCase):
    def _can_run_tbo(self, *, non_ep: bool, attn_tp_size: int) -> bool:
        model = SimpleNamespace(pp_group=SimpleNamespace(world_size=1))
        forward_batch = SimpleNamespace(
            can_run_tbo=True,
            tbo_children=[object(), object()],
            global_forward_mode=SimpleNamespace(
                is_extend_without_speculative=lambda: True
            ),
        )
        backend = SimpleNamespace(is_none=lambda: non_ep)

        with (
            patch(
                "sglang.srt.layers.moe.is_tbo_enabled",
                return_value=True,
            ),
            patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=False),
            patch.object(
                deepseek_v4,
                "get_moe_a2a_backend",
                return_value=backend,
            ),
            patch.object(
                deepseek_v4,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=attn_tp_size),
            ),
        ):
            return deepseek_v4.DeepseekV4Model._can_run_tbo(model, forward_batch)

    def test_non_ep_tbo_falls_back_with_multiple_attention_tp_ranks(self):
        self.assertFalse(self._can_run_tbo(non_ep=True, attn_tp_size=4))

    def test_non_ep_tbo_remains_enabled_with_one_attention_tp_rank(self):
        self.assertTrue(self._can_run_tbo(non_ep=True, attn_tp_size=1))

    def test_ep_tbo_is_not_restricted_by_attention_tp_size(self):
        self.assertTrue(self._can_run_tbo(non_ep=False, attn_tp_size=4))


if __name__ == "__main__":
    unittest.main()
