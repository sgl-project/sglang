"""Compute rows move independently of the source residual and finish once."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
)
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.layout import Layout, TokenAxis
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD, PlainAdd
from sglang.srt.layers.layer_boundary.row_transport import (
    ComputeTransport,
    RowTransport,
    TransportRows,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class GatheredOperand(ComputeTransport):
    def __init__(self, source, events, rows):
        self.rows = rows
        self.input = torch.cat((source, source + 7))
        self.events = events

    @contextmanager
    def scope(self):
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")

    def return_output(self, output):
        self.events.append("sum-return")
        left, right = output.chunk(2)
        return left + right


class TestRowTransport(unittest.TestCase):
    def build(self, events, *, in_token_order=False, update=PLAIN_ADD):
        parallel = fixture.parallel_of(attn_dp=2, attn_tp=1, pp_size=1)
        source = Layout.sharded_over(
            TokenAxis.ATTN_DP, axis_sizes={TokenAxis.ATTN_DP: 2}
        )
        rows = TransportRows(source, Layout(frozenset()), source)
        transport = RowTransport(
            {
                variant: rows
                for variant in BatchVariant
                if variant is not BatchVariant.SEQUENCE_PARALLEL
            },
            lambda x, batch, context, rows, overlapped: GatheredOperand(
                x, events, rows
            ),
        )
        with fixture.planning(parallel), layer_stack():
            (stage,) = append_stages(
                (
                    declare_attn(row_transport=transport, in_token_order=in_token_order)
                    if in_token_order
                    else declare_ffn(row_transport=transport, update=update),
                    fixture.Norm(),
                )
            )
        return parallel, stage

    def test_transport_rejects_unhandled_token_order_contract(self):
        with self.assertRaisesRegex(NotImplementedError, "another output contract"):
            self.build([], in_token_order=True)

    def batch(self):
        return SimpleNamespace(
            input_ids=torch.tensor([1, 2]),
            forward_mode=ForwardMode.DECODE,
            residual_stream=None,
            dp_padding_mode=None,
        )

    def test_operand_moves_after_read_and_residual_stays_on_source_rows(self):
        events = []
        parallel, stage = self.build(events)
        batch = self.batch()
        source = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        residual_batch.start(batch)
        with fixture.planning(parallel):
            operand = stage.prepare(source, batch)
            self.assertIs(batch.residual_stream.residual, source)
            torch.testing.assert_close(operand, torch.cat((source * 2, source * 2 + 7)))
            with stage.exit(batch) as output:
                hidden = output.finish(operand * 3)
                with self.assertRaisesRegex(RuntimeError, "already consumed"):
                    output.finish(operand)
        torch.testing.assert_close(hidden, source * 12 + 21)
        self.assertIs(batch.residual_stream.pending.update, PLAIN_ADD)
        self.assertIsNone(batch.residual_stream.transport)
        self.assertEqual(events, ["enter", "sum-return", "exit"])

    def test_transport_binds_update_context_once_for_finish_and_exit(self):
        for route in ("finish", "exit"):
            for at_exit in (False, True):
                bindings = []
                applied = []

                class BoundUpdate(PlainAdd):
                    applied_at_exit = at_exit

                    def __init__(self, context):
                        self.context = context

                    def update(self, value, residual):
                        applied.append(self.context)
                        return super().update(value, residual)

                class Update(PlainAdd):
                    applied_at_exit = at_exit

                    def bind(self, batch, context):
                        assert context is not None, "missing update context"
                        bindings.append(context)
                        return BoundUpdate(context)

                parallel, stage = self.build([], update=Update())
                for context in (17, 29):
                    with self.subTest(route=route, at_exit=at_exit, context=context):
                        batch = self.batch()
                        residual_batch.start(batch)
                        source = torch.ones(2, 4)
                        with fixture.planning(parallel):
                            operand = stage.prepare(source, batch)
                            if route == "finish":
                                result = stage.finish(
                                    operand * 3, batch, update_context=context
                                )
                            else:
                                with stage.exit(
                                    batch, update_context=context
                                ) as output:
                                    result = output.finish(operand * 3)
                            if at_exit:
                                self.assertEqual(applied[-1], context)
                            else:
                                self.assertEqual(
                                    batch.residual_stream.pending.update.context,
                                    context,
                                )
                            result = residual_batch.final_norm(
                                result, batch, fixture.Norm()
                            )
                        torch.testing.assert_close(result, source * 26 + 42)
                        self.assertIsNone(batch.residual_stream)
                self.assertEqual(bindings, [17, 29])

    def test_successful_scope_requires_finish_and_error_releases_invocation(self):
        for fails in (False, True):
            with self.subTest(compute_error=fails):
                events = []
                parallel, stage = self.build(events)
                batch = self.batch()
                residual_batch.start(batch)
                with fixture.planning(parallel):
                    stage.prepare(torch.ones(2, 4), batch)
                    with self.assertRaisesRegex(
                        RuntimeError, "compute failed" if fails else "without finish"
                    ):
                        with stage.exit(batch):
                            if fails:
                                raise RuntimeError("compute failed")
                self.assertIsNone(batch.residual_stream.transport)
                self.assertEqual(events, ["enter", "exit"])


if __name__ == "__main__":
    unittest.main()
