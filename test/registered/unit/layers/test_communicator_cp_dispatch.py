"""Unit tests for CP-specific communicator dispatch."""

import unittest
from functools import partial
from unittest.mock import patch

from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    ScatterMode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fn_name(fn):
    if isinstance(fn, partial):
        fn = fn.func
    return fn.__name__


def _cp_context_with_equal_scattered_and_full_sizes():
    return CommunicateContext(
        process_group_sizes={
            ScatterMode.SCATTERED: 1,
            ScatterMode.TP_ATTN_FULL: 1,
            ScatterMode.FULL: 1,
            ScatterMode.MOE_FULL: 1,
        },
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_dp_size=1,
        attn_cp_rank=0,
        attn_cp_size=2,
        tp_size=2,
        tp_rank=0,
    )


class TestCommunicatorCPDispatch(CustomTestCase):
    def test_cp_prepare_mlp_uses_gather_when_full_matches_scattered_size(self):
        with (
            patch(
                "sglang.srt.layers.communicator.is_dsa_enable_prefill_cp",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communicator.is_mla_prefill_cp_enabled",
                return_value=False,
            ),
        ):
            fn = CommunicateWithAllReduceAndLayerNormFn.get_fn(
                hidden_states_input_mode=ScatterMode.SCATTERED,
                residual_input_mode=ScatterMode.SCATTERED,
                hidden_states_output_mode=ScatterMode.FULL,
                residual_output_mode=ScatterMode.SCATTERED,
                context=_cp_context_with_equal_scattered_and_full_sizes(),
            )

        self.assertEqual(_fn_name(fn), "_cp_gather_hidden_states_and_residual")

    def test_cp_postprocess_uses_scatter_when_full_matches_scattered_size(self):
        with (
            patch(
                "sglang.srt.layers.communicator.is_dsa_enable_prefill_cp",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communicator.is_mla_prefill_cp_enabled",
                return_value=False,
            ),
        ):
            fn = CommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=ScatterMode.FULL,
                residual_input_mode=ScatterMode.SCATTERED,
                output_mode=ScatterMode.SCATTERED,
                context=_cp_context_with_equal_scattered_and_full_sizes(),
            )

        self.assertEqual(_fn_name(fn), "_cp_scatter_hidden_states")

    def test_non_cp_dispatch_still_uses_group_size_shortcut(self):
        with (
            patch(
                "sglang.srt.layers.communicator.is_dsa_enable_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.communicator.is_mla_prefill_cp_enabled",
                return_value=False,
            ),
        ):
            fn = CommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=ScatterMode.FULL,
                residual_input_mode=ScatterMode.SCATTERED,
                output_mode=ScatterMode.SCATTERED,
                context=_cp_context_with_equal_scattered_and_full_sizes(),
            )

        self.assertEqual(_fn_name(fn), "_trivial")


if __name__ == "__main__":
    unittest.main()
