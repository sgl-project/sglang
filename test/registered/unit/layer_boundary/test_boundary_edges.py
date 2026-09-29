import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers.layer_boundary import (
    ADD,
    EdgeDecl,
    FusedMlpInput,
    Layout,
    StageInput,
    StageOutput,
    SumGroup,
    TokenAxis,
    make_boundary,
    make_output_boundary,
)
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.ops import identity_output
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def sizes(*, dp=1, cp=1, tp=1):
    return {
        TokenAxis.ATTN_DP: dp,
        TokenAxis.ATTN_CP: cp,
        TokenAxis.ATTN_TP_SCATTER: tp,
    }


def rows(axis_sizes, *axes):
    return Layout.sharded_over(*axes, axis_sizes=axis_sizes)


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
        self.assertIs(producer.output_move, identity_output)
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
        over_tp = comm.FusedMlpInput(completes=SumGroup.TP, run=MagicMock())
        over_attention_tp = comm.FusedMlpInput(
            completes=SumGroup.ATTN_TP, run=MagicMock()
        )
        boundary = make_boundary(edge, fusions=(over_tp, over_attention_tp))
        self.assertEqual(
            boundary.prepare.keywords["step"].keywords.get("fusions", ()),
            (over_tp.run,),
        )
        step = make_boundary(edge).prepare
        self.assertIs(step.keywords["step"].func, comm_ops._mlp_input_without_dp)
        all_reduce = MagicMock(side_effect=lambda h: h * 2)
        with patch_communicator("tensor_model_parallel_all_reduce", all_reduce):
            hidden, residual = step(
                torch.ones(3, 4),
                torch.full((3, 4), 5.0),
                None,
                lambda hidden, residual: (hidden + residual, hidden + residual),
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

    def residual_to_attn_tp_shard(self, residual):
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
        self.assertNotIn("update", other.keywords)

    def test_capabilities_prebind_both_orders_without_capturing_an_update(self):
        edge = msgspec.structs.replace(
            self.edge(None), update_capabilities=(True, False)
        )
        boundary = make_boundary(edge)
        paths = boundary.prepare.keywords["paths"]
        self.assertEqual(set(paths), {True, False})
        self.assertIs(paths[True].keywords["step"].func, comm_ops._mlp_input_dp_partial)
        self.assertIs(
            paths[False].keywords["step"].func, comm_ops._mlp_input_dp_replicate
        )
        for path in paths.values():
            self.assertNotIn("update", path.keywords["step"].keywords)

    def test_actual_update_selects_a_prebound_path(self):
        full = rows(sizes(tp=1))
        edge = EdgeDecl(
            StageOutput(full, update=None),
            StageInput(full),
            full,
            full,
            update_capabilities=(True, False),
        )
        boundary = make_boundary(edge)

        def norm(value, residual=None):
            return value if residual is None else (value + residual, value + residual)

        for update, expected in ((comm.ADD, 4.0), (_WrittenIn(), 5.0)):
            with self.subTest(plain=update.adds_plainly):
                hidden, residual = boundary.prepare(
                    torch.ones(2, 4),
                    torch.full((2, 4), 3.0),
                    None,
                    norm,
                    update=update,
                )
                torch.testing.assert_close(hidden, torch.full((2, 4), expected))
                torch.testing.assert_close(residual, hidden)

    def test_fused_kernels_take_only_a_plain_add(self):
        full = rows(sizes(tp=2))
        fused = comm.FusedMlpInput(completes=SumGroup.ATTN_TP, run=MagicMock())
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
                self.assertEqual(
                    boundary.prepare.keywords["step"].keywords.get("fusions", ()),
                    (fused.run,) if offered else (),
                )
                self.assertEqual(
                    boundary.prepare.keywords["carried_fusions"],
                    (carried,) if offered else (),
                )

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
        )
        group.all_reduce.assert_called_once()
        torch.testing.assert_close(hidden, torch.full((3, 4), 7.0))
        group.all_reduce.reset_mock()
        hidden, _ = step(torch.full((3, 4), 2.0), residual, None, norm)
        group.all_reduce.assert_not_called()
        torch.testing.assert_close(hidden, torch.full((3, 4), 7.0))

    def test_cross_layer_update_requires_a_lifetime_guarantee(self):
        local = Layout(frozenset())
        stateful_update = SimpleNamespace(adds_plainly=False, at_producer=False)
        edge = EdgeDecl(
            StageOutput(local, update=stateful_update), StageInput(local), local, local
        )
        with self.assertRaisesRegex(NotImplementedError, "lifetime"):
            make_output_boundary(edge)

    def test_pipeline_cannot_reconstruct_a_non_add_update(self):
        local = Layout(frozenset())
        update = SimpleNamespace(
            adds_plainly=False, at_producer=False, can_defer_across_layers=True
        )
        edge = EdgeDecl(
            StageOutput(local, update=update), StageInput(local), local, local
        )
        with get_parallel().override(pp_size=2):
            with self.assertRaisesRegex(NotImplementedError, "pipeline boundaries"):
                make_output_boundary(edge)


class ProbeRead:
    before_gather = False
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
        return boundary.prepare.keywords["step"], boundary.prepare.keywords[
            "step"
        ].keywords.get("fusions", ())

    def test_the_dp_partial_reads_the_gathered_sum_with_the_declared_read(self):
        read = ProbeRead(norms_plainly=True)
        step, _ = self._ffn_input(read, dp=2)
        self.assertIs(step.func, comm_ops._mlp_input_dp_partial)
        hidden, residual = torch.ones(2, 4), torch.full((2, 4), 3.0)
        with (
            get_parallel().override(attn_tp_rank=0),
            patch.object(
                comm_ops,
                "_reduce_and_redistribute_output_to_dp",
                lambda h, forward_batch, cp_shard_counts: h * 2,
            ),
            patch.object(comm_ops, "dp_scatter", lambda *args: None),
        ):
            out, out_residual = step(hidden, residual, None, MagicMock())
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
        with (
            get_parallel().override(tp_size=2, tp_rank=0),
            patch.object(comm_ops, "tensor_model_parallel_all_reduce", lambda h: h * 2),
        ):
            out, out_residual = step(
                hidden,
                residual,
                None,
                MagicMock(),
            )
        self.assertEqual(read.reads, 1)
        expected = torch.tensor([8.0, 8.0, 2.0, 2.0])[:, None].expand(4, 4)
        torch.testing.assert_close(out_residual, expected)
        torch.testing.assert_close(out, expected + 100)

    def test_a_fused_kernel_runs_only_for_a_read_that_leaves_the_residual(self):
        fused = FusedMlpInput(completes=SumGroup.ATTN_TP, run=MagicMock())
        for norms_plainly in (True, False):
            with self.subTest(norms_plainly=norms_plainly):
                step, tried = self._ffn_input(
                    ProbeRead(norms_plainly), fusions=(fused,)
                )
                self.assertIs(step.func, comm_ops._mlp_input_without_dp)
                expected = (fused.run,) if norms_plainly else ()
                self.assertEqual(tried, expected)
                self.assertEqual(step.keywords["fusions"], expected)

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
