import unittest
from functools import partial
from unittest.mock import patch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.boundary_layout import Layout, TokenAxis
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    ScatterMode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCATTERED = ScatterMode.SCATTERED
TP_ATTN_FULL = ScatterMode.TP_ATTN_FULL
FULL = ScatterMode.FULL
MOE_FULL = ScatterMode.MOE_FULL
Norm = CommunicateWithAllReduceAndLayerNormFn
Pair = CommunicateSummableTensorPairFn


def make_context(*, dp=1, cp=1, tp=1):
    """A context for attention DP x CP x TP, built without the process-wide
    parallel state."""
    return CommunicateContext(
        process_group_sizes={},
        attn_tp_rank=0,
        attn_tp_size=tp,
        attn_dp_size=dp,
        attn_cp_rank=0,
        attn_cp_size=cp,
        tp_size=dp * cp * tp,
        tp_rank=0,
    )


class TestScatterModeLayouts(CustomTestCase):
    def test_modes_shard_nested_axes(self):
        layouts = make_context(dp=2, cp=2, tp=2).layouts
        self.assertEqual(
            layouts[SCATTERED],
            Layout(
                frozenset(
                    {TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, TokenAxis.ATTN_TP_SCATTER}
                )
            ),
        )
        self.assertEqual(
            layouts[TP_ATTN_FULL],
            Layout(frozenset({TokenAxis.ATTN_DP, TokenAxis.ATTN_CP})),
        )
        self.assertEqual(layouts[FULL], Layout(frozenset({TokenAxis.ATTN_CP})))
        self.assertEqual(layouts[MOE_FULL], Layout(frozenset()))

    def test_single_rank_axes_shard_nothing(self):
        cases = [
            # (topology, pairs with equal layouts, pairs with different layouts)
            (
                dict(tp=2),
                [(FULL, TP_ATTN_FULL), (MOE_FULL, FULL)],
                [(SCATTERED, TP_ATTN_FULL)],
            ),
            (dict(dp=2, tp=2), [(MOE_FULL, FULL)], [(FULL, TP_ATTN_FULL)]),
            (
                dict(cp=2),
                [(SCATTERED, TP_ATTN_FULL), (FULL, TP_ATTN_FULL)],
                [(MOE_FULL, FULL)],
            ),
            (dict(), [(SCATTERED, MOE_FULL)], []),
        ]
        for topology, same, different in cases:
            context = make_context(**topology)
            for a, b in same:
                with self.subTest(topology=topology, same=(a, b)):
                    self.assertTrue(context.is_same_layout(a, b))
            for a, b in different:
                with self.subTest(topology=topology, different=(a, b)):
                    self.assertFalse(context.is_same_layout(a, b))


class TestBoundarySelection(CustomTestCase):
    def assert_partial(self, fn, func, residual_input_mode):
        self.assertIsInstance(fn, partial)
        self.assertIs(fn.func, func)
        self.assertEqual(fn.keywords, {"residual_input_mode": residual_input_mode})

    def test_prepare_mlp(self):
        dp_tp = make_context(dp=2, tp=2)
        self.assert_partial(
            Norm.get_fn(TP_ATTN_FULL, SCATTERED, FULL, TP_ATTN_FULL, dp_tp),
            Norm._gather_hidden_states_and_residual,
            SCATTERED,
        )
        self.assert_partial(
            Norm.get_fn(
                TP_ATTN_FULL, TP_ATTN_FULL, MOE_FULL, TP_ATTN_FULL, make_context(cp=2)
            ),
            Norm._gather_hidden_states_and_residual_moe,
            TP_ATTN_FULL,
        )
        self.assert_partial(
            Norm.get_fn(TP_ATTN_FULL, TP_ATTN_FULL, SCATTERED, SCATTERED, dp_tp),
            Norm._scatter_hidden_states_and_residual,
            TP_ATTN_FULL,
        )

    def test_prepare_mlp_without_attention_dp(self):
        # FULL and TP_ATTN_FULL share a layout here but select different paths.
        tp = make_context(tp=2)
        self.assert_partial(
            Norm.get_fn(TP_ATTN_FULL, TP_ATTN_FULL, FULL, TP_ATTN_FULL, tp),
            Norm._gather_hidden_states_and_residual,
            TP_ATTN_FULL,
        )
        self.assertIs(
            Norm.get_fn(TP_ATTN_FULL, TP_ATTN_FULL, TP_ATTN_FULL, TP_ATTN_FULL, tp),
            Norm._tp_attn_all_reduce_and_layernorm,
        )
        self.assertIs(
            Norm.get_fn(TP_ATTN_FULL, TP_ATTN_FULL, FULL, TP_ATTN_FULL, make_context()),
            Norm._simple,
        )
        self.assertIs(
            Norm.get_fn(SCATTERED, SCATTERED, SCATTERED, SCATTERED, tp),
            Norm._simple,
        )
        with self.assertRaises(NotImplementedError):
            Norm.get_fn(SCATTERED, SCATTERED, TP_ATTN_FULL, TP_ATTN_FULL, tp)

    def test_postprocess(self):
        self.assertIs(
            Pair.get_fn(FULL, TP_ATTN_FULL, TP_ATTN_FULL, make_context(dp=2)),
            Pair._scatter_hidden_states,
        )
        self.assertIs(
            Pair.get_fn(FULL, TP_ATTN_FULL, TP_ATTN_FULL, make_context(tp=2)),
            Pair._trivial,
        )
        self.assertIs(
            Pair.get_fn(MOE_FULL, TP_ATTN_FULL, TP_ATTN_FULL, make_context(cp=2)),
            Pair._scatter_hidden_states_moe,
        )
        tp = make_context(tp=2)
        self.assertIs(Pair.get_fn(SCATTERED, SCATTERED, TP_ATTN_FULL, tp), Pair._gather)
        self.assertIs(
            Pair.get_fn(TP_ATTN_FULL, TP_ATTN_FULL, SCATTERED, tp), Pair._scatter
        )

    def test_prepare_attn(self):
        with patch.object(comm, "_use_ag_after_qlora", False):
            self.assertIs(
                CommunicateSimpleFn.get_fn(SCATTERED, TP_ATTN_FULL, make_context(tp=2)),
                CommunicateSimpleFn._scattered_to_tp_attn_full,
            )
            self.assertIs(
                CommunicateSimpleFn.get_fn(SCATTERED, TP_ATTN_FULL, make_context(cp=2)),
                CommunicateSimpleFn._trivial,
            )


if __name__ == "__main__":
    unittest.main()
