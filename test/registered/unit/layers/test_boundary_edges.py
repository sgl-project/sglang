import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.boundary_layout import (
    EdgeDecl,
    Layout,
    StageInput,
    StageOutput,
    SumGroup,
    TokenAxis,
    decoder_layer_edges,
    decoder_layer_sides,
    input_scattered_layer_sides,
)
from sglang.srt.layers.communicator import InputRead, make_boundary
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

Pair = comm.CommunicateSummableTensorPairFn


def sizes(*, dp=1, cp=1, tp=1):
    return {
        TokenAxis.ATTN_DP: dp,
        TokenAxis.ATTN_CP: cp,
        TokenAxis.ATTN_TP_SCATTER: tp,
    }


def rows(axis_sizes, *axes):
    return Layout.sharded_over(*axes, axis_sizes=axis_sizes)


class TestDecoderLayerEdges(CustomTestCase):
    """A decoder layer's three boundaries hand the residual on from one to the
    next: each edge starts on the rows the one before it ends on."""

    def test_each_edge_starts_where_the_one_before_ends(self):
        for dp, tp, ffn_local, previous_local, last in itertools.product(
            (1, 2), (1, 2), (False, True), (False, True), (False, True)
        ):
            with self.subTest(
                dp=dp, tp=tp, ffn_local=ffn_local, previous_local=previous_local
            ):
                sides = decoder_layer_sides(
                    axis_sizes=sizes(dp=dp, tp=tp),
                    ffn_on_local_rows=ffn_local,
                    previous_on_local_rows=previous_local,
                    is_last_layer=last,
                    attention_gathers_local_rows=False,
                    ffn_group=SumGroup.TP,
                    leaves_for_next_layer=True,
                    leaves_for_reduce_scatter=True,
                    leaves_for_reduce_scatterv=True,
                )
                edges = decoder_layer_edges(sides)
                self.assertEqual(edges.into_attention.residual, sides.input_rows)
                self.assertEqual(
                    edges.into_attention.residual_to, edges.into_ffn.residual
                )
                self.assertEqual(edges.into_ffn.residual_to, edges.out_of_ffn.residual)
                self.assertEqual(edges.out_of_ffn.residual_to, sides.output_rows)
                self.assertEqual(edges.out_of_ffn.need.layout, sides.output_rows)
                # Nothing owed by construction: a sum left for a batch comes
                # with the value.
                self.assertFalse(edges.into_attention.produced.always_leaves)

    def test_an_input_owed_by_construction_is_completed_onto_the_slice(self):
        axis_sizes = sizes(tp=2)
        sides = input_scattered_layer_sides(
            axis_sizes=axis_sizes, ffn_group=SumGroup.TP, hands_on_partial=True
        )
        edge = decoder_layer_edges(sides).into_attention
        self.assertIs(edge.produced.group, SumGroup.TP)
        self.assertTrue(edge.produced.always_leaves)
        self.assertEqual(edge.residual_to, rows(axis_sizes, TokenAxis.ATTN_TP_SCATTER))
        boundary = make_boundary(edge, reads=InputRead.ATTENTION)
        self.assertIs(boundary.prepare.keywords["layer_input"], comm.tp_reduce_scatter)


class TestTheConsumerHalfReadsOnlyItsOwnSide(CustomTestCase):
    """Two layers build the two halves of one boundary without sharing it; the
    consumer's half never depends on what the producer may leave for a batch,
    which arrives with the value."""

    def test_what_the_producer_may_leave_does_not_change_the_consumer_half(self):
        axis_sizes = sizes(dp=2, tp=2)
        attention = rows(axis_sizes, TokenAxis.ATTN_DP)
        halves = set()
        for group, next_layer, reduce_scatter, reduce_scatterv in itertools.product(
            (None, SumGroup.TP, SumGroup.MOE_OUTPUT),
            (False, True),
            (False, True),
            (False, True),
        ):
            produced = StageOutput(
                attention,
                group=group,
                leaves_for_next_layer=next_layer,
                leaves_for_reduce_scatter=reduce_scatter,
                leaves_for_reduce_scatterv=reduce_scatterv,
            )
            edge = EdgeDecl(
                produced=produced,
                need=StageInput(attention),
                residual=attention,
                residual_to=attention,
            )
            boundary = make_boundary(edge, reads=InputRead.ATTENTION)
            halves.add(
                (
                    boundary.prepare.func,
                    tuple(sorted(boundary.prepare.keywords.items())),
                    boundary.input_move,
                )
            )
        self.assertEqual(len(halves), 1)
        ((func, keywords, move),) = halves
        self.assertIs(func, comm._attention_input_step)
        self.assertIsNone(dict(keywords)["layer_input"])
        self.assertIs(move, comm.CommunicateSimpleFn._trivial)


class TestNonAlternatingEdges(CustomTestCase):
    """Stages that do not alternate attention and FFN build their boundaries
    through the same entry."""

    def test_a_mixer_into_a_mixer(self):
        axis_sizes = sizes(dp=2, tp=2)
        attention = rows(axis_sizes, TokenAxis.ATTN_DP)
        produced = StageOutput(
            attention, group=SumGroup.ATTN_TP, leaves_for_next_layer=True
        )
        edge = EdgeDecl(
            produced=produced,
            need=StageInput(attention),
            residual=attention,
            residual_to=attention,
        )
        # The producer's half has nothing to move; the consumer completes the
        # sum the value carries.
        producer = make_boundary(edge, reads=None)
        self.assertIs(producer.output_move, Pair._trivial)
        consumer = make_boundary(edge, reads=InputRead.ATTENTION)
        self.assertIsNone(consumer.prepare.keywords["layer_input"])
        self.assertIs(consumer.input_move, comm.CommunicateSimpleFn._trivial)

    def test_an_ffn_into_an_ffn(self):
        axis_sizes = sizes(dp=2, tp=2)
        attention = rows(axis_sizes, TokenAxis.ATTN_DP)
        full = rows(axis_sizes)
        # A complete output on the rows the previous layer hands on: the
        # consumer adds and normalizes it, then gathers over attention DP.
        complete = EdgeDecl(
            produced=StageOutput(attention),
            need=StageInput(full),
            residual=attention,
            residual_to=attention,
        )
        step = make_boundary(complete, reads=InputRead.FFN).prepare
        self.assertIs(step.func, comm._mlp_input_dp_replicate)
        self.assertFalse(step.keywords["reduces_attention_tp"])
        # An FFN that may leave its TP sum for a batch hands it on with the
        # value, which completes onto the rows the layer hands on; the next
        # FFN's input completes it, then goes on as for a complete output.
        leaves = EdgeDecl(
            produced=StageOutput(
                attention, group=SumGroup.TP, leaves_for_next_layer=True
            ),
            need=StageInput(full),
            residual=attention,
            residual_to=attention,
        )
        step = make_boundary(leaves, reads=InputRead.FFN).prepare
        self.assertIs(step.func, comm._mlp_input_completing_owed)
        self.assertIs(step.keywords["step"].func, comm._mlp_input_dp_replicate)
        # A producer that always leaves a sum other than the attention TP's is
        # not supported.
        always = EdgeDecl(
            produced=StageOutput(attention, group=SumGroup.TP, always_leaves=True),
            need=StageInput(full),
            residual=attention,
            residual_to=attention,
        )
        with self.assertRaises(NotImplementedError):
            make_boundary(always, reads=InputRead.FFN)

    def test_the_next_ffn_completes_a_sum_left_for_it_once(self):
        full = rows(sizes(tp=2))
        edge = EdgeDecl(
            produced=StageOutput(full, group=SumGroup.TP, leaves_for_next_layer=True),
            need=StageInput(full),
            residual=full,
            residual_to=full,
        )
        step = make_boundary(edge, reads=InputRead.FFN).prepare
        group = SimpleNamespace(all_reduce=MagicMock(side_effect=lambda h: h * 2))

        def norm(hidden, residual):
            # The add and norm: hidden + residual.
            return hidden + residual, hidden + residual

        residual = torch.full((3, 4), 5.0)
        hidden, _ = step(
            comm.UnreducedOutput(torch.ones(3, 4), group=group),
            residual,
            None,
            norm,
            None,
        )
        group.all_reduce.assert_called_once()
        torch.testing.assert_close(hidden, torch.full((3, 4), 7.0))
        # A complete output is not reduced again.
        group.all_reduce.reset_mock()
        hidden, _ = step(torch.full((3, 4), 2.0), residual, None, norm, None)
        group.all_reduce.assert_not_called()
        torch.testing.assert_close(hidden, torch.full((3, 4), 7.0))

    def test_the_producer_half_ends_on_the_rows_it_hands_on(self):
        axis_sizes = sizes(dp=2, tp=2)
        attention = rows(axis_sizes, TokenAxis.ATTN_DP)
        edge = EdgeDecl(
            produced=StageOutput(attention),
            need=StageInput(rows(axis_sizes)),
            residual=attention,
            residual_to=attention,
        )
        with self.assertRaises(NotImplementedError):
            make_boundary(edge, reads=None)


if __name__ == "__main__":
    unittest.main()
