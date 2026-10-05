import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers.layer_boundary import (
    PLAIN_ADD,
    EdgeContract,
    FfnInputFusion,
    InputContract,
    Layout,
    OutputContract,
    SumGroup,
    TokenAxis,
    bind_entry,
    bind_exit,
)
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.ops import keep_output
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def sizes(*, dp=1, cp=1, tp=1):
    return {
        TokenAxis.ATTN_DP: dp,
        TokenAxis.ATTN_CP: cp,
        TokenAxis.ATTN_TP: tp,
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
            produced = OutputContract(
                attention,
                group=group,
                may_defer_to_next=next_layer,
                may_reduce_scatter=reduce_scatter,
                may_reduce_scatterv=reduce_scatterv,
            )
            edge = EdgeContract(
                produced=produced,
                need=InputContract(attention),
                residual=attention,
                residual_to=attention,
            )
            boundary = bind_entry(edge)
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
        self.assertIs(func, comm_ops._run_entry)
        self.assertIs(step, comm_ops._update_read)
        self.assertIsNone(dict(keywords)["pre_move"])
        self.assertIsNone(move)


class TestNonAlternatingEdges(CustomTestCase):
    """Stages that do not alternate attention and FFN build their boundaries
    through the same entry."""

    def test_a_mixer_into_a_mixer(self):
        axis_sizes = sizes(dp=2, tp=2)
        attention = rows(axis_sizes, TokenAxis.ATTN_DP)
        produced = OutputContract(
            attention, group=SumGroup.ATTN_TP, may_defer_to_next=True
        )
        edge = EdgeContract(
            produced=produced,
            need=InputContract(attention),
            residual=attention,
            residual_to=attention,
        )
        producer = bind_exit(edge)
        self.assertIs(producer.output_move, keep_output)
        consumer = bind_entry(edge)
        self.assertIs(consumer.prepare.func, comm_ops._run_entry)
        self.assertIsNone(consumer.prepare.keywords["step"].keywords["pre_move"])
        self.assertIsNone(consumer.input_move)

    def test_an_ffn_into_an_ffn(self):
        axis_sizes = sizes(dp=2, tp=2)
        attention = rows(axis_sizes, TokenAxis.ATTN_DP)
        full = rows(axis_sizes)
        complete = EdgeContract(
            produced=OutputContract(attention),
            need=InputContract(full),
            residual=attention,
            residual_to=attention,
        )
        step = bind_entry(complete).prepare.keywords["step"]
        self.assertIs(step.func, comm_ops._reduce_update_read_dp_gather)
        self.assertFalse(step.keywords["reduces_attention_tp"])
        # A sum the value carries is completed first, then the same steps run.
        leaves = EdgeContract(
            produced=OutputContract(
                attention, group=SumGroup.TP, may_defer_to_next=True
            ),
            need=InputContract(full),
            residual=attention,
            residual_to=attention,
        )
        prepare = bind_entry(leaves).prepare
        self.assertIs(prepare.func, comm_ops._run_entry)
        self.assertIs(
            prepare.keywords["step"].func, comm_ops._reduce_update_read_dp_gather
        )
        # A TP sum left on the rows of one attention-DP shard: the TP group spans
        # the other shards, whose ranks hold other rows, so no all-reduce over it
        # completes these.
        always = EdgeContract(
            produced=OutputContract(attention, group=SumGroup.TP, always_partial=True),
            need=InputContract(full),
            residual=attention,
            residual_to=attention,
        )
        with self.assertRaises(NotImplementedError):
            bind_entry(always)

    def test_an_ffn_that_always_leaves_its_tp_sum(self):
        # Without attention DP every TP rank holds the rows, so TP completes the sum.
        full = rows(sizes(tp=2))
        edge = EdgeContract(
            produced=OutputContract(full, group=SumGroup.TP, always_partial=True),
            need=InputContract(full),
            residual=full,
            residual_to=full,
        )
        over_tp = comm.FfnInputFusion(completes=SumGroup.TP, run=MagicMock())
        over_attention_tp = comm.FfnInputFusion(
            completes=SumGroup.ATTN_TP, run=MagicMock()
        )
        boundary = bind_entry(edge, fusions=(over_tp, over_attention_tp))
        self.assertEqual(
            boundary.prepare.keywords["step"].keywords.get("fusions", ()),
            (over_tp.run,),
        )
        step = bind_entry(edge).prepare
        self.assertIs(step.keywords["step"].func, comm_ops._reduce_update_read)
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

    is_plain_add = False
    applied_at_exit = False

    def update(self, hidden_states, residual):
        return 2 * hidden_states + residual

    def slice_residual_attn_tp(self, residual):
        return residual

    def gather_residual_attn_tp(self, residual):
        return residual


class TestTheProducersUpdateChoosesTheOrder(CustomTestCase):
    """On the same rows, the steps into a stage follow the update the producer
    declares: a plain add may join a sum before it completes and run inside a
    fused add + norm, another update runs after the sum completes."""

    def edge(self, update):
        axis_sizes = sizes(dp=2, tp=2)
        attention = rows(axis_sizes, TokenAxis.ATTN_DP)
        return EdgeContract(
            produced=OutputContract(
                attention, group=SumGroup.ATTN_TP, always_partial=True, update=update
            ),
            need=InputContract(rows(axis_sizes)),
            residual=attention,
            residual_to=attention,
        )

    def test_the_dp_gather_order(self):
        plain = bind_entry(self.edge(comm.PLAIN_ADD)).prepare.keywords["step"]
        self.assertIs(plain.func, comm_ops._dp_gather_sum_read)
        written = _WrittenIn()
        other = bind_entry(self.edge(written)).prepare.keywords["step"]
        self.assertIs(other.func, comm_ops._reduce_update_read_dp_gather)
        self.assertTrue(other.keywords["reduces_attention_tp"])
        self.assertNotIn("update", other.keywords)

    def test_capabilities_prebind_both_orders_without_capturing_an_update(self):
        edge = msgspec.structs.replace(
            self.edge(None), arriving_plain_add=(True, False)
        )
        boundary = bind_entry(edge)
        paths = boundary.prepare.keywords["paths"]
        self.assertEqual(set(paths), {True, False})
        self.assertIs(paths[True].keywords["step"].func, comm_ops._dp_gather_sum_read)
        self.assertIs(
            paths[False].keywords["step"].func, comm_ops._reduce_update_read_dp_gather
        )
        for path in paths.values():
            self.assertNotIn("update", path.keywords["step"].keywords)

    def test_actual_update_selects_a_prebound_path(self):
        full = rows(sizes(tp=1))
        edge = EdgeContract(
            OutputContract(full, update=None),
            InputContract(full),
            full,
            full,
            arriving_plain_add=(True, False),
        )
        boundary = bind_entry(edge)

        def norm(value, residual=None):
            return value if residual is None else (value + residual, value + residual)

        for update, expected in ((comm.PLAIN_ADD, 4.0), (_WrittenIn(), 5.0)):
            with self.subTest(plain=update.is_plain_add):
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
        fused = comm.FfnInputFusion(completes=SumGroup.ATTN_TP, run=MagicMock())
        carried = MagicMock()
        for update, offered in ((comm.PLAIN_ADD, True), (_WrittenIn(), False)):
            with self.subTest(is_plain_add=offered):
                edge = EdgeContract(
                    produced=OutputContract(
                        full, group=SumGroup.ATTN_TP, always_partial=True, update=update
                    ),
                    need=InputContract(full),
                    residual=full,
                    residual_to=full,
                )
                boundary = bind_entry(
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
        edge = EdgeContract(
            produced=OutputContract(full, group=SumGroup.TP, may_defer_to_next=True),
            need=InputContract(full),
            residual=full,
            residual_to=full,
        )
        step = bind_entry(edge).prepare
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
        stateful_update = SimpleNamespace(is_plain_add=False, applied_at_exit=False)
        edge = EdgeContract(
            OutputContract(local, update=stateful_update),
            InputContract(local),
            local,
            local,
        )
        with self.assertRaisesRegex(NotImplementedError, "lifetime"):
            bind_exit(edge)

    def test_pipeline_cannot_reconstruct_a_non_add_update(self):
        local = Layout(frozenset())
        update = SimpleNamespace(
            is_plain_add=False, applied_at_exit=False, outlives_layer=True
        )
        edge = EdgeContract(
            OutputContract(local, update=update), InputContract(local), local, local
        )
        with get_parallel().override(pp_size=2):
            with self.assertRaisesRegex(NotImplementedError, "pipeline boundaries"):
                bind_exit(edge)


class ProbeRead:
    reads_before_dp_gather = False
    """A read that marks what it reads: the input is the residual plus 100.
    ``is_plain_norm`` says whether it may stand in for a norm."""

    def __init__(self, is_plain_norm):
        self.is_plain_norm = is_plain_norm
        self.reads = 0

    def init_residual(self, hidden_states):
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
            rows(axis_sizes, TokenAxis.ATTN_DP, TokenAxis.ATTN_TP)
            if residual_joins_sum
            else attention
        )
        need = rows(axis_sizes) if dp > 1 else attention
        boundary = bind_entry(
            EdgeContract(
                produced=OutputContract(
                    attention,
                    group=SumGroup.ATTN_TP,
                    always_partial=True,
                    update=PLAIN_ADD,
                ),
                need=InputContract(need, read=read),
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
        read = ProbeRead(is_plain_norm=True)
        step, _ = self._ffn_input(read, dp=2)
        self.assertIs(step.func, comm_ops._dp_gather_sum_read)
        hidden, residual = torch.ones(2, 4), torch.full((2, 4), 3.0)
        with (
            get_parallel().override(attn_tp_rank=0),
            patch.object(
                comm_ops,
                "dp_gather_sum",
                lambda h, forward_batch, cp_shard_counts: h * 2,
            ),
            patch.object(comm_ops, "dp_scatter", lambda *args: None),
        ):
            out, out_residual = step(hidden, residual, None, MagicMock())
        self.assertEqual(read.reads, 1)
        torch.testing.assert_close(out, torch.full((2, 4), 108.0))
        self.assertIs(out_residual, residual)

    def test_a_read_that_changes_the_residual_replicates_instead(self):
        step, _ = self._ffn_input(ProbeRead(is_plain_norm=False), dp=2)
        self.assertIs(step.func, comm_ops._reduce_update_read_dp_gather)

    def test_the_residual_joined_sum_is_read_with_the_declared_read(self):
        read = ProbeRead(is_plain_norm=True)
        step, _ = self._ffn_input(read, residual_joins_sum=True)
        self.assertIs(step.func, comm_ops._tp_sum_with_residual_read)
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
        fused = FfnInputFusion(completes=SumGroup.ATTN_TP, run=MagicMock())
        for is_plain_norm in (True, False):
            with self.subTest(is_plain_norm=is_plain_norm):
                step, tried = self._ffn_input(
                    ProbeRead(is_plain_norm), fusions=(fused,)
                )
                self.assertIs(step.func, comm_ops._reduce_update_read)
                expected = (fused.run,) if is_plain_norm else ()
                self.assertEqual(tried, expected)
                self.assertEqual(step.keywords["fusions"], expected)

    def test_the_attention_input_tries_its_kernels_only_for_such_a_read(self):
        axis_sizes = sizes(tp=2)
        attention = rows(axis_sizes)
        kernel = MagicMock()
        for is_plain_norm in (True, False):
            with self.subTest(is_plain_norm=is_plain_norm):
                read = ProbeRead(is_plain_norm)
                edge = EdgeContract(
                    produced=OutputContract(
                        attention, group=SumGroup.TP, may_defer_to_next=True
                    ),
                    need=InputContract(attention, read=read),
                    residual=attention,
                    residual_to=attention,
                )
                boundary = bind_entry(edge, carried_fusions=(kernel,))
                self.assertIs(boundary.prepare.keywords["step"].keywords["read"], read)
                self.assertEqual(
                    boundary.prepare.keywords["carried_fusions"],
                    (kernel,) if is_plain_norm else (),
                )


if __name__ == "__main__":
    unittest.main()
