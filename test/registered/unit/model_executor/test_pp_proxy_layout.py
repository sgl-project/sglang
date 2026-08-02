"""Unit tests for pipeline-parallel proxy tensor layout helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.model_executor.model_runner_components.misc_utils import (
    compute_pp_proxy_num_tokens,
    should_use_pp_send_allgather,
)
from sglang.srt.models.minimax_m3 import MiniMaxM3Model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPPProxyLayout(CustomTestCase):
    def test_scattered_proxy_uses_local_token_count_after_pp0(self):
        self.assertEqual(
            compute_pp_proxy_num_tokens(
                num_tokens=8, pp_rank=1, token_scatter_factor=2
            ),
            4,
        )

    def test_pp0_and_replicated_proxy_keep_full_token_count(self):
        self.assertEqual(
            compute_pp_proxy_num_tokens(
                num_tokens=8, pp_rank=0, token_scatter_factor=2
            ),
            8,
        )
        self.assertEqual(
            compute_pp_proxy_num_tokens(
                num_tokens=8, pp_rank=2, token_scatter_factor=1
            ),
            8,
        )

    def test_scattered_proxy_requires_divisible_token_count(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            compute_pp_proxy_num_tokens(num_tokens=7, pp_rank=1, token_scatter_factor=2)

    def test_send_allgather_requires_replicated_non_cp_proxy(self):
        self.assertTrue(
            should_use_pp_send_allgather(
                enable_dsa_prefill_context_parallel=False,
                preserve_tp_lanes=False,
            )
        )
        self.assertFalse(
            should_use_pp_send_allgather(
                enable_dsa_prefill_context_parallel=False,
                preserve_tp_lanes=True,
            )
        )
        self.assertFalse(
            should_use_pp_send_allgather(
                enable_dsa_prefill_context_parallel=True,
                preserve_tp_lanes=False,
            )
        )

    def test_lane_preserving_transport_allows_replicated_stage_input(self):
        self.assertEqual(
            compute_pp_proxy_num_tokens(
                num_tokens=8, pp_rank=1, token_scatter_factor=1
            ),
            8,
        )
        self.assertFalse(
            should_use_pp_send_allgather(
                enable_dsa_prefill_context_parallel=False,
                preserve_tp_lanes=True,
            )
        )

    def test_minimax_uses_stage_input_layout_for_proxy_size(self):
        backend = SimpleNamespace(is_none=lambda: False)
        model = object.__new__(MiniMaxM3Model)
        object.__setattr__(model, "pp_group", SimpleNamespace(is_first_rank=False))
        object.__setattr__(model, "start_layer", 0)

        with patch(
            "sglang.srt.models.minimax_m3.get_moe_a2a_backend",
            return_value=backend,
        ), patch(
            "sglang.srt.models.minimax_m3.get_parallel",
            return_value=SimpleNamespace(attn_tp_size=2),
        ):
            for input_mode, expected_factor in (
                (ScatterMode.SCATTERED, 2),
                (ScatterMode.TP_ATTN_FULL, 1),
            ):
                object.__setattr__(
                    model,
                    "layers",
                    [
                        SimpleNamespace(
                            layer_scatter_modes=SimpleNamespace(
                                layer_input_mode=input_mode
                            )
                        )
                    ],
                )
                self.assertEqual(
                    model.get_pp_proxy_token_scatter_factor(), expected_factor
                )


if __name__ == "__main__":
    unittest.main()
