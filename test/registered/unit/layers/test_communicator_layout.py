import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.boundary_layout import Layout, TokenAxis
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    ScatterMode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCATTERED = ScatterMode.SCATTERED
TP_ATTN_FULL = ScatterMode.TP_ATTN_FULL
FULL = ScatterMode.FULL
MOE_FULL = ScatterMode.MOE_FULL
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


def modes(attn, layer_input, mlp, middle_residual):
    return SimpleNamespace(
        attn_mode=attn,
        layer_input_mode=layer_input,
        mlp_mode=mlp,
        middle_residual_mode=middle_residual,
    )


class TestBoundarySelection(CustomTestCase):
    def test_prepare_mlp(self):
        dp_tp = make_context(dp=2, tp=2)
        Kind = comm.MlpInputKind
        for layout, context, kind in (
            ((TP_ATTN_FULL, SCATTERED, FULL, TP_ATTN_FULL), dp_tp, Kind.GATHER),
            (
                (TP_ATTN_FULL, TP_ATTN_FULL, MOE_FULL, TP_ATTN_FULL),
                make_context(cp=2),
                Kind.GATHER_MOE_CP,
            ),
            ((TP_ATTN_FULL, TP_ATTN_FULL, SCATTERED, SCATTERED), dp_tp, Kind.SCATTER),
        ):
            with self.subTest(layout=layout):
                self.assertIs(comm.mlp_input_kind(modes(*layout), context), kind)

    def test_prepare_mlp_without_attention_dp(self):
        # FULL and TP_ATTN_FULL share a layout here but select different paths.
        tp = make_context(tp=2)
        Kind = comm.MlpInputKind
        for layout, context, kind in (
            ((TP_ATTN_FULL, TP_ATTN_FULL, FULL, TP_ATTN_FULL), tp, Kind.GATHER),
            (
                (TP_ATTN_FULL, TP_ATTN_FULL, TP_ATTN_FULL, TP_ATTN_FULL),
                tp,
                Kind.ATTN_TP_ALL_REDUCE,
            ),
            (
                (TP_ATTN_FULL, TP_ATTN_FULL, FULL, TP_ATTN_FULL),
                make_context(),
                Kind.NORM,
            ),
        ):
            with self.subTest(layout=layout):
                self.assertIs(comm.mlp_input_kind(modes(*layout), context), kind)
        for layout in (
            (SCATTERED, SCATTERED, TP_ATTN_FULL, TP_ATTN_FULL),
            # The LayerNorm SP region chooses its steps from its declarations.
            (SCATTERED, SCATTERED, SCATTERED, SCATTERED),
        ):
            with self.subTest(layout=layout), self.assertRaises(NotImplementedError):
                comm.mlp_input_kind(modes(*layout), tp)

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


class Fusable:
    def forward_with_allreduce_fusion(self, *args, **kwargs):
        pass


def communicator(layout, context, post_attention_layernorm, cls=comm.LayerCommunicator):
    """A communicator with these layer layouts, built without the process-wide
    parallel state; returns it with its chosen prepare_mlp steps."""
    c = cls.__new__(cls)
    c.layer_scatter_modes = modes(*layout)
    c._context = context
    c.post_attention_layernorm = post_attention_layernorm
    return c, c._select_mlp_input()


GATHER_LAYOUT = (TP_ATTN_FULL, TP_ATTN_FULL, FULL, TP_ATTN_FULL)


class TestMlpInputOrder(CustomTestCase):
    def assert_order(self, order, func, **keywords):
        self.assertIsInstance(order, partial)
        self.assertIs(order.func, func)
        self.assertEqual(order.keywords, keywords)

    def test_facts_fixed_at_construction_pick_the_order(self):
        fusions = (object(),)
        for mode, gathers in ((TP_ATTN_FULL, False), (SCATTERED, True)):
            with self.subTest(residual_input_mode=mode):
                self.assert_order(
                    comm._mlp_input_order(make_context(tp=2), mode, fusions),
                    comm._mlp_input_without_dp,
                    gathers_residual=gathers,
                    fusions=fusions,
                )
        # Attention TP 1 has no partial sum to carry through the gather.
        self.assert_order(
            comm._mlp_input_order(make_context(dp=2), TP_ATTN_FULL, fusions),
            comm._mlp_input_dp_replicate,
            gathers_residual=False,
            reduces_attention_tp=False,
        )
        dp_tp = make_context(dp=2, tp=2)
        self.assert_order(
            comm._mlp_input_order(dp_tp, SCATTERED, fusions),
            comm._mlp_input_dp_partial,
            gathers_residual=True,
        )
        dp_tp.force_layernorm_before_dp_gather = True
        self.assert_order(
            comm._mlp_input_order(dp_tp, SCATTERED, fusions),
            comm._mlp_input_dp_replicate,
            gathers_residual=True,
            reduces_attention_tp=True,
        )

    def test_a_gather_runs_the_order_chosen_at_construction(self):
        for mlp_mode in (FULL, MOE_FULL):
            with self.subTest(mlp_mode=mlp_mode):
                c, (steps, fused) = communicator(
                    (TP_ATTN_FULL, TP_ATTN_FULL, mlp_mode, TP_ATTN_FULL),
                    make_context(tp=2),
                    Fusable(),
                )
                if mlp_mode is MOE_FULL:
                    self.assertIs(steps.func, comm._mlp_input_gather_moe_cp)
                    steps = steps.keywords["gather"]
                self.assertIs(steps.func, comm._mlp_input_gather)
                entry = c._mlp_input_reduce_output_and_update_and_read_residual
                self.assert_order(
                    steps.keywords["order"],
                    comm._mlp_input_without_dp,
                    gathers_residual=False,
                    fusions=(entry,),
                )
                self.assertEqual([f.run for f in fused], [entry])

    def test_no_fused_entry_without_a_fusable_norm_or_under_attention_dp(self):
        for context, norm in (
            (make_context(tp=2), object()),
            (make_context(dp=2, tp=2), Fusable()),
        ):
            with self.subTest(dp=context.attn_dp_size):
                _, (steps, fused) = communicator(GATHER_LAYOUT, context, norm)
                order = steps.keywords["order"]
                self.assertEqual(order.keywords.get("fusions", ()), ())
                self.assertEqual(fused, ())

    def test_other_kinds_take_their_steps(self):
        tp = make_context(tp=2)
        for layout, context, expected in (
            (
                (TP_ATTN_FULL, TP_ATTN_FULL, FULL, TP_ATTN_FULL),
                make_context(),
                comm._mlp_input_norm,
            ),
            (
                (TP_ATTN_FULL, TP_ATTN_FULL, TP_ATTN_FULL, TP_ATTN_FULL),
                tp,
                comm._mlp_input_attn_tp_all_reduce,
            ),
        ):
            with self.subTest(expected=expected.__name__):
                _, (steps, fused) = communicator(layout, context, Fusable())
                self.assertIs(steps, expected)
                self.assertEqual(fused, ())
        _, (steps, _) = communicator(
            (TP_ATTN_FULL, TP_ATTN_FULL, SCATTERED, SCATTERED), tp, Fusable()
        )
        self.assert_order(steps, comm._mlp_input_scatter, scatters_residual=True)

    def test_mhc_picks_its_own_implementation_of_the_kind(self):
        from sglang.srt.layers.communicator_mhc import (
            MHCCommunicateWithAllReduceAndLayerNormFn as MHC,
        )
        from sglang.srt.layers.communicator_mhc import (
            MHCLayerCommunicator,
        )

        mhc = object()
        for layout, func, residual_keyword in (
            (
                GATHER_LAYOUT,
                MHC._gather_hidden_states_and_residual,
                dict(residual_on_slice=False),
            ),
            (
                (TP_ATTN_FULL, TP_ATTN_FULL, SCATTERED, SCATTERED),
                MHC._scatter_hidden_states_and_residual,
                dict(scatters_residual=True),
            ),
        ):
            with self.subTest(func=func.__name__):
                c = MHCLayerCommunicator.__new__(MHCLayerCommunicator)
                c.mhc = mhc
                c.layer_scatter_modes = modes(*layout)
                c._context = make_context(tp=2)
                c.post_attention_layernorm = Fusable()
                steps, fused = c._select_mlp_input()
                self.assert_order(steps, func, **residual_keyword, mhc=mhc)
                self.assertEqual(fused, ())


if __name__ == "__main__":
    unittest.main()
