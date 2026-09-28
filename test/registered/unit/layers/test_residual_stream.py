import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.communicator import (
    BoundarySteps,
    EdgeDecl,
    LayerCommunicator,
    Layout,
    StageEntry,
    StageInput,
    StageOutput,
    make_boundary,
)
from sglang.srt.layers.communicator.layer import StageCommunicator
from sglang.srt.layers.communicator.output import UnreducedOutput
from sglang.srt.layers.communicator.residual.access import add_to_output
from sglang.srt.layers.communicator.residual.add_norm import ADD
from sglang.srt.layers.communicator.residual.stream import OwedOutput, ResidualStream
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestResidualStream(CustomTestCase):
    def setUp(self):
        self.group = SimpleNamespace(all_reduce=Mock(side_effect=lambda x: x.mul_(2)))
        self.residual = torch.full((2, 4), 3.0)
        self.partial = torch.ones(2, 4)
        self.stream = ResidualStream(self.residual)
        self.hidden = self.stream.leave(
            UnreducedOutput(self.partial, group=self.group), ADD
        )

    def test_written_input_does_not_repeat_first_stage_initialization(self):
        class Read:
            norms_plainly = False
            enters = 0

            def enter(self, value):
                self.enters += 1
                return value * 3

            def read(self, value, norm, quant_format=""):
                return value * 2, value

        read = Read()
        rows = Layout(frozenset())
        boundary = make_boundary(
            EdgeDecl(StageOutput(rows), StageInput(rows, read=read), rows, rows),
            enters_stack=True,
        )
        steps = BoundarySteps(
            StageEntry(boundary.prepare, rows), None, StageOutput(rows), None, False
        )
        stage = StageCommunicator(
            SimpleNamespace(input_layernorm=None, _context=None),
            "attention",
            "input_layernorm",
        )
        value = torch.ones(2, 4)
        batch = SimpleNamespace(residual_stream=ResidualStream())
        hidden, stream = stage.prepare(value, batch.residual_stream, batch, steps)
        self.assertEqual(read.enters, 1)
        torch.testing.assert_close(hidden, value * 6)
        hidden, stream = stage.prepare(stream.residual, stream, batch, steps)
        self.assertEqual(read.enters, 1)
        torch.testing.assert_close(hidden, value * 6)

    def test_consumer_executes_the_carried_update_not_the_bound_producers(self):
        class Update:
            adds_plainly = False
            at_producer = False

            def __init__(self, increment):
                self.increment = increment
                self.calls = 0

            def update(self, value, residual):
                self.calls += 1
                return value + residual + self.increment

        class Read:
            norms_plainly = False

            def update_and_read(self, update, value, residual, norm, **kwargs):
                residual = update.update(value, residual)
                return residual * 2, residual

        declared = Update(100)
        rows = Layout(frozenset())
        boundary = make_boundary(
            EdgeDecl(
                StageOutput(rows, update=declared),
                StageInput(rows, read=Read()),
                rows,
                rows,
            )
        )
        steps = BoundarySteps(
            StageEntry(boundary.prepare, rows), None, StageOutput(rows), None, False
        )
        stage = StageCommunicator(
            SimpleNamespace(input_layernorm=None, _context=None),
            "attention",
            "input_layernorm",
        )
        for increment in (2, 7):
            with self.subTest(increment=increment):
                actual = Update(increment)
                stream = ResidualStream(torch.full((2, 4), 3.0))
                hidden = stream.leave(torch.ones(2, 4), actual)
                batch = SimpleNamespace(residual_stream=stream)
                result, stream = stage.prepare(hidden, stream, batch, steps)
                torch.testing.assert_close(
                    result, torch.full((2, 4), 2.0 * (4 + increment))
                )
                self.assertEqual(actual.calls, 1)
                self.assertIsNone(stream.pending)
        self.assertEqual(declared.calls, 0)

    def test_unbound_update_capability_fails_before_consuming_the_sum(self):
        rows = Layout(frozenset())
        boundary = make_boundary(
            EdgeDecl(StageOutput(rows), StageInput(rows), rows, rows)
        )
        with self.assertRaisesRegex(RuntimeError, "capability"):
            boundary.prepare(
                self.partial,
                self.residual,
                None,
                None,
                None,
                update=SimpleNamespace(adds_plainly=False),
            )

    def test_snapshot_finishes_a_copy_without_consuming_main_work(self):
        snapshot = self.stream.snapshot(self.hidden)
        torch.testing.assert_close(snapshot, torch.full((2, 4), 5.0))
        torch.testing.assert_close(self.partial, torch.ones(2, 4))
        torch.testing.assert_close(self.residual, torch.full((2, 4), 3.0))
        self.assertIsInstance(self.hidden, OwedOutput)
        self.assertIsNotNone(self.stream.pending.owed)
        with self.assertRaises(TypeError):
            self.hidden + 1
        with self.assertRaises(TypeError):
            self.hidden[0]

    def test_main_capture_completes_once_and_leaves_update_pending(self):
        hidden = self.stream.complete(self.hidden)
        self.assertIs(self.stream.complete(hidden), hidden)
        self.group.all_reduce.assert_called_once()
        self.assertIs(self.stream.pending.value, hidden)
        self.assertIs(self.stream.pending.update, ADD)
        self.assertIsNone(self.stream.pending.owed)
        self.assertIs(self.stream.residual, self.residual)
        torch.testing.assert_close(hidden, torch.full((2, 4), 2.0))
        with self.assertRaises(RuntimeError):
            self.stream.input(self.hidden)
        with self.assertRaises(RuntimeError):
            self.stream.input(hidden.clone())

    def test_capture_and_compute_share_the_fused_read(self):
        boundary = LayerCommunicator.__new__(LayerCommunicator)
        events = []
        outputs = AuxHiddenStateList()

        def prepare(hidden, stream, batch, **kwargs):
            events.append("add_norm")
            self.assertIsNone(stream.pending.owed)
            # A fused read returns both the norm output and its updated residual.
            updated = hidden + stream.residual
            stream.write(updated)
            return updated * 3, stream

        def capture(value, *, owned=False):
            events.append("capture")
            outputs.capture(value, owned=owned)

        boundary.prepare_attn = Mock(side_effect=prepare)
        boundary.attn = SimpleNamespace(
            entry=lambda batch: SimpleNamespace(
                capture_move=None,
                capture_move_allocates=False,
                capture_preserves_residual=None,
            )
        )
        output, stream = boundary.prepare_attn_and_capture_last_layer_outputs(
            self.hidden, self.stream, None, capture_output=capture
        )
        self.assertEqual(events, ["add_norm", "capture"])
        self.group.all_reduce.assert_called_once()
        boundary.prepare_attn.assert_called_once()
        torch.testing.assert_close(output, torch.full((2, 4), 15.0))
        torch.testing.assert_close(outputs[0], torch.full((2, 4), 5.0))
        stream.residual.zero_()
        torch.testing.assert_close(outputs[0], torch.full((2, 4), 5.0))

    def test_deepstack_capture_precedes_the_extra_addition(self):
        boundary = LayerCommunicator.__new__(LayerCommunicator)
        extra = torch.full_like(self.partial, 7.0)
        outputs = AuxHiddenStateList()

        def prepare(hidden, stream, batch, **kwargs):
            self.assertEqual(len(outputs), 1)
            self.assertIs(kwargs["post_residual_addition"], extra)
            updated = hidden + stream.residual + extra
            stream.write(updated)
            return updated * 3, stream

        boundary.prepare_attn = Mock(side_effect=prepare)
        boundary.attn = SimpleNamespace(
            entry=lambda batch: SimpleNamespace(
                capture_move=None,
                capture_move_allocates=False,
                capture_preserves_residual=None,
            )
        )
        output, _ = boundary.prepare_attn_and_capture_last_layer_outputs(
            self.hidden,
            self.stream,
            None,
            post_residual_addition=extra,
            capture_output=outputs.capture,
        )
        torch.testing.assert_close(outputs[0], torch.full((2, 4), 5.0))
        torch.testing.assert_close(output, torch.full((2, 4), 36.0))
        self.group.all_reduce.assert_called_once()

    def test_capture_restores_the_producers_rows_after_a_scattered_read(self):
        from unittest.mock import patch

        from sglang.srt.layers.communicator import TokenAxis

        full = Layout(frozenset())
        local = Layout(frozenset({TokenAxis.ATTN_TP_SCATTER}))
        boundary = make_boundary(
            EdgeDecl(StageOutput(full), StageInput(local), full, local)
        )
        shard = torch.full((2, 4), 3.0)
        expected = torch.cat([shard, torch.full_like(shard, 7.0)])
        with patch(
            "sglang.srt.layers.communicator.ops._redistribute_from_attn_tp_shards",
            return_value=expected,
        ) as gather:
            captured = boundary.capture_move(shard, forward_batch=None)
        gather.assert_called_once_with(shard)
        torch.testing.assert_close(captured, expected)
        torch.testing.assert_close(shard, torch.full((2, 4), 3.0))

    def test_final_capture_uses_the_norms_residual_result(self):
        from sglang.srt.layers.communicator.residual.access import norm_output

        outputs = AuxHiddenStateList()
        updated = torch.full_like(self.partial, 5.0)
        normalized = torch.full_like(self.partial, 13.0)
        norm = Mock(return_value=(normalized, updated))
        result = norm_output(self.partial, self.residual, norm, outputs.capture)
        self.assertIs(result, normalized)
        norm.assert_called_once_with(self.partial, self.residual)
        updated.zero_()
        torch.testing.assert_close(outputs[0], torch.full((2, 4), 5.0))

    def test_deepstack_adds_once_after_completing_the_sum(self):
        hidden, stream = add_to_output(
            self.hidden, self.stream, torch.full((2, 4), 7.0)
        )
        self.assertIs(stream, self.stream)
        self.assertIs(stream.pending.value, hidden)
        self.assertIs(stream.residual, self.residual)
        torch.testing.assert_close(hidden, torch.full((2, 4), 9.0))
        self.group.all_reduce.assert_called_once()

    def test_handles_cannot_cross_microbatches_or_be_reused_after_take(self):
        other = ResidualStream(self.residual.clone())
        other.leave(UnreducedOutput(self.partial.clone(), group=self.group), ADD)
        with self.assertRaises(RuntimeError):
            other.input(self.hidden)
        with self.assertRaises(RuntimeError):
            self.stream.leave(torch.zeros_like(self.partial), ADD)
        self.stream.write(self.residual)
        with self.assertRaises(RuntimeError):
            self.stream.input(self.hidden)
        self.assertEqual(self.stream.input(self.residual), (self.residual, None))

    def test_local_row_move_is_owed_even_without_an_all_reduce(self):
        move = Mock(side_effect=lambda value: value[:1])
        stream = ResidualStream(self.residual[:1])
        hidden = stream.leave(
            UnreducedOutput(self.partial, reduce_and_redistribute=move), ADD
        )
        self.assertIsInstance(hidden, OwedOutput)
        completed = stream.complete(hidden)
        self.assertEqual(completed.shape[0], 1)
        self.assertIs(stream.pending.value, completed)
        move.assert_called_once_with(self.partial)


if __name__ == "__main__":
    unittest.main()
