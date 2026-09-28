import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import (
    ADD,
    EdgeDecl,
    FusedMlpInput,
    Layout,
    StageInput,
    StageOutput,
    SumGroup,
    TokenAxis,
)
from sglang.srt.layers.communicator import boundary as comm_boundary
from sglang.srt.layers.communicator import (
    decoder_layer_edges,
    decoder_layer_sides,
    input_scattered_layer_sides,
    make_boundary,
    make_output_boundary,
)
from sglang.srt.layers.communicator import ops as comm_ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
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
        step = make_boundary(edge).prepare.keywords["step"]
        self.assertIs(step.keywords["layer_input"], comm.tp_reduce_scatter)

    def test_each_entry_keeps_the_move_after_its_read(self):
        sides = decoder_layer_sides(
            axis_sizes=sizes(dp=2, tp=2),
            ffn_on_local_rows=False,
            previous_on_local_rows=False,
            is_last_layer=False,
            attention_gathers_local_rows=False,
            ffn_group=SumGroup.TP,
            leaves_for_next_layer=True,
            leaves_for_reduce_scatter=True,
            leaves_for_reduce_scatterv=True,
        )
        edges = decoder_layer_edges(sides)
        moves = {edges.into_attention: MagicMock(), edges.into_ffn: MagicMock()}
        real = comm_boundary.make_boundary

        def with_move(edge, **kwargs):
            return msgspec.structs.replace(real(edge, **kwargs), input_move=moves[edge])

        with patch.object(comm_boundary, "make_boundary", with_move):
            steps = comm_boundary._select_boundary_steps(sides)
        self.assertIs(steps.attention.input_move, moves[edges.into_attention])
        self.assertIs(steps.ffn.input_move, moves[edges.into_ffn])


class TestAMoeOnEachCpShard(CustomTestCase):
    """A MoE whose data-parallel groups are the CP ranks computes each CP shard
    on its own ranks: its input stays sharded over CP, and the attention output
    reaches it without a gather."""

    def test_the_ffn_rows_stay_sharded_over_cp(self):
        axis_sizes = sizes(cp=2, tp=2)
        sides = decoder_layer_sides(
            axis_sizes=axis_sizes,
            ffn_on_local_rows=False,
            previous_on_local_rows=False,
            is_last_layer=False,
            attention_gathers_local_rows=False,
            ffn_group=SumGroup.MOE_OUTPUT,
            leaves_for_next_layer=False,
            leaves_for_reduce_scatter=False,
            leaves_for_reduce_scatterv=False,
            ffn_shards_over_cp=True,
        )
        cp_rows = rows(axis_sizes, TokenAxis.ATTN_CP)
        self.assertEqual(sides.ffn.layout, cp_rows)
        self.assertEqual(sides.ffn_output.layout, cp_rows)
        self.assertIs(sides.ffn_output.group, SumGroup.MOE_OUTPUT)
        edges = decoder_layer_edges(sides)
        self.assertEqual(edges.into_ffn.need.layout, edges.into_ffn.produced.layout)


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
            boundary = make_boundary(edge)
            step = boundary.prepare.keywords["step"]
            halves.add(
                (
                    boundary.prepare.func,
                    step.func,
                    tuple(sorted(step.keywords.items())),
                    boundary.input_move,
                )
            )
        self.assertEqual(len(halves), 1)
        ((func, step, keywords, move),) = halves
        self.assertIs(func, comm_ops._consumer_step)
        self.assertIs(step, comm_ops._read_input)
        self.assertIsNone(dict(keywords)["layer_input"])
        self.assertIsNone(move)


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
        producer = make_output_boundary(edge)
        self.assertIs(producer.output_move, Pair._trivial)
        consumer = make_boundary(edge)
        self.assertIs(consumer.prepare.func, comm_ops._consumer_step)
        self.assertIsNone(consumer.prepare.keywords["step"].keywords["layer_input"])
        self.assertIsNone(consumer.input_move)

    def test_an_ffn_into_an_ffn(self):
        axis_sizes = sizes(dp=2, tp=2)
        attention = rows(axis_sizes, TokenAxis.ATTN_DP)
        full = rows(axis_sizes)
        complete = EdgeDecl(
            produced=StageOutput(attention),
            need=StageInput(full),
            residual=attention,
            residual_to=attention,
        )
        step = make_boundary(complete).prepare.keywords["step"]
        self.assertIs(step.func, comm_ops._mlp_input_dp_replicate)
        self.assertFalse(step.keywords["reduces_attention_tp"])
        # A sum the value carries is completed first, then the same steps run.
        leaves = EdgeDecl(
            produced=StageOutput(
                attention, group=SumGroup.TP, leaves_for_next_layer=True
            ),
            need=StageInput(full),
            residual=attention,
            residual_to=attention,
        )
        prepare = make_boundary(leaves).prepare
        self.assertIs(prepare.func, comm_ops._consumer_step)
        self.assertIs(prepare.keywords["step"].func, comm_ops._mlp_input_dp_replicate)
        # A TP sum left on the rows of one attention-DP shard: the TP group spans
        # the other shards, whose ranks hold other rows, so no all-reduce over it
        # completes these.
        always = EdgeDecl(
            produced=StageOutput(attention, group=SumGroup.TP, always_leaves=True),
            need=StageInput(full),
            residual=attention,
            residual_to=attention,
        )
        with self.assertRaises(NotImplementedError):
            make_boundary(always)

    def test_an_ffn_that_always_leaves_its_tp_sum(self):
        # Without attention DP every TP rank holds the rows, so TP completes the sum.
        full = rows(sizes(tp=2))
        edge = EdgeDecl(
            produced=StageOutput(full, group=SumGroup.TP, always_leaves=True),
            need=StageInput(full),
            residual=full,
            residual_to=full,
        )
        over_tp = comm.FusedMlpInput(
            completes=SumGroup.TP, run=MagicMock(), may_return_new_residual=False
        )
        over_attention_tp = comm.FusedMlpInput(
            completes=SumGroup.ATTN_TP, run=MagicMock(), may_return_new_residual=False
        )
        boundary = make_boundary(edge, fusions=(over_tp, over_attention_tp))
        self.assertEqual(boundary.fused, (over_tp,))
        step = make_boundary(edge).prepare
        self.assertIs(step.keywords["step"].func, comm_ops._mlp_input_without_dp)
        all_reduce = MagicMock(side_effect=lambda h: h * 2)
        with patch_communicator("tensor_model_parallel_all_reduce", all_reduce):
            hidden, residual = step(
                torch.ones(3, 4),
                torch.full((3, 4), 5.0),
                None,
                lambda hidden, residual: (hidden + residual, hidden + residual),
                None,
            )
        all_reduce.assert_called_once()
        torch.testing.assert_close(hidden, torch.full((3, 4), 7.0))


class _WrittenIn:
    """An update that is not a plain add: the output is written into the
    residual only once its sum is complete."""

    adds_plainly = False
    at_producer = False

    def update(self, hidden_states, residual):
        return 2 * hidden_states + residual

    def residual_to_attn_tp_shard(self, residual, context):
        return residual

    def residual_from_attn_tp_shards(self, residual):
        return residual


class TestTheProducersUpdateChoosesTheOrder(CustomTestCase):
    """On the same rows, the steps into a stage follow the update the producer
    declares: a plain add may join a sum before it completes and run inside a
    fused add + norm, another update runs after the sum completes."""

    def edge(self, update):
        axis_sizes = sizes(dp=2, tp=2)
        attention = rows(axis_sizes, TokenAxis.ATTN_DP)
        return EdgeDecl(
            produced=StageOutput(
                attention, group=SumGroup.ATTN_TP, always_leaves=True, update=update
            ),
            need=StageInput(rows(axis_sizes)),
            residual=attention,
            residual_to=attention,
        )

    def test_the_dp_gather_order(self):
        plain = make_boundary(self.edge(comm.ADD)).prepare.keywords["step"]
        self.assertIs(plain.func, comm_ops._mlp_input_dp_partial)
        written = _WrittenIn()
        other = make_boundary(self.edge(written)).prepare.keywords["step"]
        self.assertIs(other.func, comm_ops._mlp_input_dp_replicate)
        self.assertTrue(other.keywords["reduces_attention_tp"])
        self.assertIs(other.keywords["update"], written)

    def test_fused_kernels_take_only_a_plain_add(self):
        full = rows(sizes(tp=2))
        fused = comm.FusedMlpInput(
            completes=SumGroup.ATTN_TP, run=MagicMock(), may_return_new_residual=False
        )
        carried = MagicMock()
        for update, offered in ((comm.ADD, True), (_WrittenIn(), False)):
            with self.subTest(adds_plainly=offered):
                edge = EdgeDecl(
                    produced=StageOutput(
                        full, group=SumGroup.ATTN_TP, always_leaves=True, update=update
                    ),
                    need=StageInput(full),
                    residual=full,
                    residual_to=full,
                )
                boundary = make_boundary(
                    edge, fusions=(fused,), carried_fusions=(carried,)
                )
                self.assertEqual(boundary.fused, (fused,) if offered else ())
                self.assertEqual(
                    boundary.prepare.keywords["carried_fusions"],
                    (carried,) if offered else (),
                )

    def test_the_consumer_reads_as_it_declares(self):
        full = rows(sizes(tp=2))
        read = MagicMock()
        edge = EdgeDecl(
            produced=StageOutput(full),
            need=StageInput(full, read=read),
            residual=full,
            residual_to=full,
        )
        step = make_boundary(edge).prepare.keywords["step"]
        self.assertIs(step.keywords["read"], read)

    def test_the_next_ffn_completes_a_sum_left_for_it_once(self):
        full = rows(sizes(tp=2))
        edge = EdgeDecl(
            produced=StageOutput(full, group=SumGroup.TP, leaves_for_next_layer=True),
            need=StageInput(full),
            residual=full,
            residual_to=full,
        )
        step = make_boundary(edge).prepare
        group = SimpleNamespace(all_reduce=MagicMock(side_effect=lambda h: h * 2))

        def norm(hidden, residual):
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
            make_output_boundary(edge)


class ProbeRead:
    """A read that marks what it reads: the input is the residual plus 100.
    ``norms_plainly`` says whether it may stand in for a norm."""

    def __init__(self, norms_plainly):
        self.norms_plainly = norms_plainly
        self.reads = 0

    def enter(self, hidden_states):
        return hidden_states

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        self.reads += 1
        return residual + 100, residual

    def update_and_read(
        self,
        update,
        hidden_states,
        residual,
        norm,
        quant_format="",
        post_residual_addition=None,
    ):
        return self.read(update.update(hidden_states, residual), norm)


class TestTheConsumerRunsItsDeclaredRead(CustomTestCase):
    """Every input step reads the FFN input with the read the consumer
    declares. A step that completes the sum onto the rows it reads from, or a
    fused add + norm kernel, runs only for a read that leaves the residual as
    it is."""

    def _ffn_input(self, read, *, dp=1, residual_joins_sum=False, fusions=()):
        axis_sizes = sizes(dp=dp, tp=2)
        attention = rows(axis_sizes, TokenAxis.ATTN_DP)
        residual = (
            rows(axis_sizes, TokenAxis.ATTN_DP, TokenAxis.ATTN_TP_SCATTER)
            if residual_joins_sum
            else attention
        )
        need = rows(axis_sizes) if dp > 1 else attention
        boundary = make_boundary(
            EdgeDecl(
                produced=StageOutput(
                    attention, group=SumGroup.ATTN_TP, always_leaves=True, update=ADD
                ),
                need=StageInput(need, read=read),
                residual=residual,
                residual_to=attention,
                residual_joins_sum=residual_joins_sum,
            ),
            fusions=fusions,
        )
        return boundary.prepare.keywords["step"], boundary.fused

    def test_the_dp_partial_reads_the_gathered_sum_with_the_declared_read(self):
        read = ProbeRead(norms_plainly=True)
        step, _ = self._ffn_input(read, dp=2)
        self.assertIs(step.func, comm_ops._mlp_input_dp_partial)
        hidden, residual = torch.ones(2, 4), torch.full((2, 4), 3.0)
        with (
            patch.object(
                comm_ops,
                "_reduce_and_redistribute_output_to_dp",
                lambda h, forward_batch, cp_shard_counts: h * 2,
            ),
            patch.object(comm_ops, "dp_scatter", lambda *args: None),
        ):
            out, out_residual = step(
                hidden, residual, None, MagicMock(), SimpleNamespace(attn_tp_rank=0)
            )
        self.assertEqual(read.reads, 1)
        torch.testing.assert_close(out, torch.full((2, 4), 108.0))
        self.assertIs(out_residual, residual)

    def test_a_read_that_changes_the_residual_replicates_instead(self):
        step, _ = self._ffn_input(ProbeRead(norms_plainly=False), dp=2)
        self.assertIs(step.func, comm_ops._mlp_input_dp_replicate)

    def test_the_residual_joined_sum_is_read_with_the_declared_read(self):
        read = ProbeRead(norms_plainly=True)
        step, _ = self._ffn_input(read, residual_joins_sum=True)
        self.assertIs(step.func, comm_ops._mlp_input_residual_into_sum)
        hidden, residual = torch.ones(4, 4), torch.full((2, 4), 3.0)
        with patch.object(
            comm_ops, "tensor_model_parallel_all_reduce", lambda h: h * 2
        ):
            out, out_residual = step(
                hidden,
                residual,
                None,
                MagicMock(),
                SimpleNamespace(tp_size=2, tp_rank=0),
            )
        self.assertEqual(read.reads, 1)
        expected = torch.tensor([8.0, 8.0, 2.0, 2.0])[:, None].expand(4, 4)
        torch.testing.assert_close(out_residual, expected)
        torch.testing.assert_close(out, expected + 100)

    def test_a_fused_kernel_runs_only_for_a_read_that_leaves_the_residual(self):
        fused = FusedMlpInput(
            completes=SumGroup.ATTN_TP, run=MagicMock(), may_return_new_residual=True
        )
        for norms_plainly in (True, False):
            with self.subTest(norms_plainly=norms_plainly):
                step, tried = self._ffn_input(
                    ProbeRead(norms_plainly), fusions=(fused,)
                )
                self.assertIs(step.func, comm_ops._mlp_input_without_dp)
                expected = (fused,) if norms_plainly else ()
                self.assertEqual(tried, expected)
                self.assertEqual(
                    step.keywords["fusions"], tuple(f.run for f in expected)
                )

    def test_the_attention_input_tries_its_kernels_only_for_such_a_read(self):
        axis_sizes = sizes(tp=2)
        attention = rows(axis_sizes)
        kernel = MagicMock()
        for norms_plainly in (True, False):
            with self.subTest(norms_plainly=norms_plainly):
                read = ProbeRead(norms_plainly)
                edge = EdgeDecl(
                    produced=StageOutput(
                        attention, group=SumGroup.TP, leaves_for_next_layer=True
                    ),
                    need=StageInput(attention, read=read),
                    residual=attention,
                    residual_to=attention,
                )
                boundary = make_boundary(edge, carried_fusions=(kernel,))
                self.assertIs(boundary.prepare.keywords["step"].keywords["read"], read)
                self.assertEqual(
                    boundary.prepare.keywords["carried_fusions"],
                    (kernel,) if norms_plainly else (),
                )


if __name__ == "__main__":
    unittest.main()
