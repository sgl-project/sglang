import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.layer_boundary import (
    EdgeContract,
    EntryPath,
    InputContract,
    Layout,
    OutputContract,
    StagePath,
    SumGroup,
    bind_entry,
)
from sglang.srt.layers.layer_boundary.contracts import BatchVariant, StageKind
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import (
    OwedOutput,
    ResidualStream,
)
from sglang.srt.layers.layer_boundary.stage import StageBoundary
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.boundary_fixtures import prepare_attention, stub_plan, stub_stage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestResidualStream(CustomTestCase):
    def setUp(self):
        self.group = SimpleNamespace(all_reduce=Mock(side_effect=lambda x: x.mul_(2)))
        self.residual = torch.full((2, 4), 3.0)
        self.partial = torch.ones(2, 4)
        self.stream = ResidualStream(self.residual)
        self.hidden = self.stream.record(
            UnreducedOutput(self.partial, group=self.group), PLAIN_ADD
        )

    def test_written_input_does_not_repeat_first_stage_initialization(self):
        class Read:
            is_plain_norm = False
            enters = 0

            def init_residual(self, value):
                self.enters += 1
                return value * 3

            def read(self, value, norm, quant_format=""):
                return value * 2, value

        read = Read()
        rows = Layout(frozenset())
        boundary = bind_entry(
            EdgeContract(
                OutputContract(rows), InputContract(rows, read=read), rows, rows
            ),
            enters_stack=True,
        )
        steps = StagePath(EntryPath(boundary.prepare, rows), OutputContract(rows), None)
        stage = StageBoundary(
            SimpleNamespace(norm=None),
            declaration=SimpleNamespace(kind=StageKind.ATTENTION),
        )
        value = torch.ones(2, 4)
        hidden, stream = stage._prepare(value, ResidualStream(), None, steps)
        self.assertEqual(read.enters, 1)
        torch.testing.assert_close(hidden, value * 6)
        hidden, stream = stage._prepare(stream.residual, stream, None, steps)
        self.assertEqual(read.enters, 1)
        torch.testing.assert_close(hidden, value * 6)

    def test_consumer_executes_the_carried_update_not_the_bound_producers(self):
        class Update:
            is_plain_add = False
            applied_at_exit = False

            def __init__(self, increment):
                self.increment = increment
                self.calls = 0

            def update(self, value, residual):
                self.calls += 1
                return value + residual + self.increment

        class Read:
            is_plain_norm = False

            def update_and_read(self, update, value, residual, norm, **kwargs):
                residual = update.update(value, residual)
                return residual * 2, residual

        declared = Update(100)
        rows = Layout(frozenset())
        boundary = bind_entry(
            EdgeContract(
                OutputContract(rows, update=declared),
                InputContract(rows, read=Read()),
                rows,
                rows,
            )
        )
        steps = StagePath(EntryPath(boundary.prepare, rows), OutputContract(rows), None)
        stage = StageBoundary(
            SimpleNamespace(norm=None),
            declaration=SimpleNamespace(kind=StageKind.ATTENTION),
        )
        for increment in (2, 7):
            with self.subTest(increment=increment):
                actual = Update(increment)
                stream = ResidualStream(torch.full((2, 4), 3.0))
                hidden = stream.record(torch.ones(2, 4), actual)
                result, stream = stage._prepare(hidden, stream, None, steps)
                torch.testing.assert_close(
                    result, torch.full((2, 4), 2.0 * (4 + increment))
                )
                self.assertEqual(actual.calls, 1)
                self.assertIsNone(stream.pending)
        self.assertEqual(declared.calls, 0)

    def test_unbound_update_capability_fails_before_consuming_the_sum(self):
        rows = Layout(frozenset())
        boundary = bind_entry(
            EdgeContract(OutputContract(rows), InputContract(rows), rows, rows)
        )
        with self.assertRaisesRegex(RuntimeError, "capability"):
            boundary.prepare(
                self.partial,
                self.residual,
                None,
                None,
                update=SimpleNamespace(is_plain_add=False),
            )

    def test_a_declared_sum_s_snapshot_keeps_the_partial_and_its_export_completes_it(
        self,
    ):
        stream = ResidualStream(self.residual)
        hidden = stream.record(self.partial, PLAIN_ADD, declared_sum=SumGroup.TP)
        self.assertIsInstance(hidden, OwedOutput)
        with patch(
            "sglang.srt.layers.layer_boundary.residual.stream._sum_group",
            return_value=self.group,
        ):
            snapshot = stream.snapshot(hidden)
            torch.testing.assert_close(snapshot, torch.full((2, 4), 5.0))
            torch.testing.assert_close(self.partial, torch.ones(2, 4))
            # A pipeline handoff sends it complete.
            wire, residual = stream.export(hidden)
        torch.testing.assert_close(wire, torch.full((2, 4), 2.0))
        self.assertIsNone(stream.pending.owed)
        received, rebuilt = ResidualStream.from_handoff(wire, residual, PLAIN_ADD)
        self.assertIs(received, wire)
        self.assertIsNone(rebuilt.pending.owed)

    def test_materialized_declared_sum_is_not_reduced_again(self):
        rows = Layout(frozenset())
        boundary = bind_entry(
            EdgeContract(
                OutputContract(rows, group=SumGroup.TP, always_partial=True),
                InputContract(rows),
                rows,
                rows,
            )
        )
        stream = ResidualStream(self.residual)
        hidden = stream.record(self.partial, PLAIN_ADD, declared_sum=SumGroup.TP)
        with patch(
            "sglang.srt.layers.layer_boundary.residual.stream._sum_group",
            return_value=self.group,
        ):
            hidden = stream.complete(hidden)

        def norm(value, residual):
            return value + residual, value + residual

        result, _ = boundary.prepare(
            hidden,
            self.residual,
            None,
            norm,
            pending=stream.pending,
            update=stream.pending.update,
        )
        torch.testing.assert_close(result, torch.full((2, 4), 5.0))
        self.group.all_reduce.assert_called_once()

    def test_declared_sum_must_match_the_consumer_contract(self):
        rows = Layout(frozenset())
        boundary = bind_entry(
            EdgeContract(OutputContract(rows), InputContract(rows), rows, rows)
        )
        stream = ResidualStream(self.residual)
        stream.record(self.partial, PLAIN_ADD, declared_sum=SumGroup.TP)
        with self.assertRaisesRegex(RuntimeError, "sum does not match"):
            boundary.prepare(
                self.partial,
                self.residual,
                None,
                None,
                pending=stream.pending,
                update=PLAIN_ADD,
            )

    def test_snapshot_rejects_stateful_update_without_touching_main_state(self):
        update = SimpleNamespace(is_plain_add=False, update=Mock())
        stream = ResidualStream(self.residual)
        hidden = stream.record(UnreducedOutput(self.partial, group=self.group), update)
        with self.assertRaisesRegex(NotImplementedError, "plain residual update"):
            stream.snapshot(hidden)
        self.group.all_reduce.assert_not_called()
        update.update.assert_not_called()
        self.assertIs(stream.pending.update, update)
        self.assertIsNotNone(stream.pending.owed)

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
        self.assertIs(self.stream.pending.update, PLAIN_ADD)
        self.assertIsNone(self.stream.pending.owed)
        self.assertIs(self.stream.residual, self.residual)
        torch.testing.assert_close(hidden, torch.full((2, 4), 2.0))
        with self.assertRaises(RuntimeError):
            self.stream.input(self.hidden)
        with self.assertRaises(RuntimeError):
            self.stream.input(hidden.clone())

    def test_capture_and_compute_share_the_fused_read(self):
        boundary = stub_plan()
        events = []
        outputs = AuxHiddenStateList()

        def prepare(hidden, stream, batch, steps, **kwargs):
            events.append("add_norm")
            self.assertIsNone(stream.pending.owed)
            # A fused read returns both the norm output and its updated residual.
            updated = hidden + stream.residual
            stream.write(updated)
            return updated * 3, stream

        def capture(value, *, owned=False):
            events.append("capture")
            outputs.capture(value, owned=owned)

        boundary.norm = None
        stub_stage(boundary, StageKind.ATTENTION)._prepare = Mock(side_effect=prepare)
        entry = SimpleNamespace(
            capture_move=None,
            capture_move_allocates=False,
            capture_preserves_residual=None,
        )
        stub_stage(boundary, StageKind.ATTENTION)._select = lambda hidden, batch: (
            hidden,
            SimpleNamespace(entry=entry),
        )
        output, stream = prepare_attention(
            stub_stage(boundary, StageKind.ATTENTION),
            self.hidden,
            self.stream,
            None,
            capture=capture,
        )
        self.assertEqual(events, ["add_norm", "capture"])
        self.group.all_reduce.assert_called_once()
        stub_stage(boundary, StageKind.ATTENTION)._prepare.assert_called_once()
        torch.testing.assert_close(output, torch.full((2, 4), 15.0))
        torch.testing.assert_close(outputs[0], torch.full((2, 4), 5.0))
        stream.residual.zero_()
        torch.testing.assert_close(outputs[0], torch.full((2, 4), 5.0))

    def test_deepstack_capture_precedes_the_extra_addition(self):
        boundary = stub_plan()
        extra = torch.full_like(self.partial, 7.0)
        outputs = AuxHiddenStateList()

        def prepare(hidden, stream, batch, steps, **kwargs):
            self.assertEqual(len(outputs), 1)
            self.assertIs(kwargs["post_residual_addition"], extra)
            updated = hidden + stream.residual + extra
            stream.write(updated)
            return updated * 3, stream

        boundary.norm = None
        stub_stage(boundary, StageKind.ATTENTION)._prepare = Mock(side_effect=prepare)
        entry = SimpleNamespace(
            capture_move=None,
            capture_move_allocates=False,
            capture_preserves_residual=None,
        )
        stub_stage(boundary, StageKind.ATTENTION)._select = lambda hidden, batch: (
            hidden,
            SimpleNamespace(entry=entry),
        )
        output, _ = prepare_attention(
            stub_stage(boundary, StageKind.ATTENTION),
            self.hidden,
            self.stream,
            None,
            post_residual_addition=extra,
            capture=outputs.capture,
        )
        torch.testing.assert_close(outputs[0], torch.full((2, 4), 5.0))
        torch.testing.assert_close(output, torch.full((2, 4), 36.0))
        self.group.all_reduce.assert_called_once()

    def test_capture_restores_the_producers_rows_after_a_scattered_read(self):
        from unittest.mock import patch

        from sglang.srt.layers.layer_boundary import TokenAxis

        full = Layout(frozenset())
        local = Layout(frozenset({TokenAxis.ATTN_TP}))
        boundary = bind_entry(
            EdgeContract(OutputContract(full), InputContract(local), full, local)
        )
        shard = torch.full((2, 4), 3.0)
        expected = torch.cat([shard, torch.full_like(shard, 7.0)])
        with patch(
            "sglang.srt.layers.layer_boundary.ops.attn_tp_gather",
            return_value=expected,
        ) as gather:
            captured = boundary.capture_move(shard, forward_batch=None)
        gather.assert_called_once_with(shard)
        torch.testing.assert_close(captured, expected)
        torch.testing.assert_close(shard, torch.full((2, 4), 3.0))

    def test_final_capture_uses_the_norms_residual_result(self):
        from sglang.srt.layers.layer_boundary.residual.access import final_norm_pair

        outputs = AuxHiddenStateList()
        updated = torch.full_like(self.partial, 5.0)
        normalized = torch.full_like(self.partial, 13.0)
        norm = Mock(return_value=(normalized, updated))
        result = final_norm_pair(self.partial, self.residual, norm, outputs.capture)
        self.assertIs(result, normalized)
        norm.assert_called_once_with(self.partial, self.residual)
        updated.zero_()
        torch.testing.assert_close(outputs[0], torch.full((2, 4), 5.0))

    def test_deepstack_adds_once_after_completing_the_sum(self):
        hidden = residual_batch.add_to_output(
            self.hidden,
            SimpleNamespace(residual_stream=self.stream),
            torch.full((2, 4), 7.0),
        )
        self.assertIs(self.stream.pending.value, hidden)
        self.assertIs(self.stream.residual, self.residual)
        torch.testing.assert_close(hidden, torch.full((2, 4), 9.0))
        self.group.all_reduce.assert_called_once()

    def test_handles_cannot_cross_microbatches_or_be_reused_after_take(self):
        other = ResidualStream(self.residual.clone())
        other.record(UnreducedOutput(self.partial.clone(), group=self.group), PLAIN_ADD)
        with self.assertRaises(RuntimeError):
            other.input(self.hidden)
        with self.assertRaises(RuntimeError):
            self.stream.record(torch.zeros_like(self.partial), PLAIN_ADD)
        self.stream.write(self.residual)
        with self.assertRaises(RuntimeError):
            self.stream.input(self.hidden)
        self.assertEqual(self.stream.input(self.residual), (self.residual, None))

    def test_local_row_move_is_owed_even_without_an_all_reduce(self):
        move = Mock(side_effect=lambda value: value[:1])
        stream = ResidualStream(self.residual[:1])
        hidden = stream.record(
            UnreducedOutput(self.partial, reduce_to_dp_local=move), PLAIN_ADD
        )
        self.assertIsInstance(hidden, OwedOutput)
        completed = stream.complete(hidden)
        self.assertEqual(completed.shape[0], 1)
        self.assertIs(stream.pending.value, completed)
        move.assert_called_once_with(self.partial)

    def test_prepare_releases_the_consumed_contribution(self):
        class Read:
            is_plain_norm = True

            def update_and_read(self, update, value, residual, norm, **kwargs):
                residual = update.update(value, residual)
                return residual * 2, residual

        rows = Layout(frozenset())
        boundary = bind_entry(
            EdgeContract(
                OutputContract(rows), InputContract(rows, read=Read()), rows, rows
            )
        )
        steps = StagePath(EntryPath(boundary.prepare, rows), OutputContract(rows), None)
        stage = StageBoundary(
            SimpleNamespace(norm=None),
            declaration=SimpleNamespace(kind=StageKind.ATTENTION),
        )
        partial = torch.ones(4, 4)
        partial_ref = weakref.ref(partial)
        stream = ResidualStream(torch.full((2, 4), 3.0))
        # The deferred DP completion: sum, then this rank's rows.
        hidden = stream.record(
            UnreducedOutput(partial, reduce_to_dp_local=lambda x: x[:2] * 2),
            PLAIN_ADD,
        )
        del partial
        result, stream = stage._prepare(hidden, stream, None, steps)
        torch.testing.assert_close(result, torch.full((2, 4), 10.0))
        # The caller still holds the handle; it no longer keeps the partial.
        self.assertIsInstance(hidden, OwedOutput)
        self.assertIsNone(partial_ref())
        with self.assertRaises(RuntimeError):
            stream.input(hidden)


class TestBatchStageOwnership(CustomTestCase):
    def test_terminal_norm_releases_layer_buffers(self):
        from sglang.srt.layers.layer_boundary.residual import batch

        fb = SimpleNamespace(residual_stream=None)
        batch.start(fb)
        residual = torch.ones(2, 4)
        contribution = torch.full_like(residual, 2)
        refs = [weakref.ref(residual), weakref.ref(contribution)]
        batch.stream_of(fb).write(residual)
        output = batch.stream_of(fb).record(contribution, PLAIN_ADD)
        result = batch.final_norm(output, fb, lambda x, r: (x + r, r))
        del output, residual, contribution
        torch.testing.assert_close(result, torch.full((2, 4), 3.0))
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertIsNone(fb.residual_stream)

    def test_pp_export_transfers_buffer_ownership(self):
        from sglang.srt.layers.layer_boundary.residual import batch

        fb = SimpleNamespace(residual_stream=None)
        batch.start(fb)
        residual = torch.ones(2, 4)
        contribution = torch.full_like(residual, 2)
        refs = [weakref.ref(residual), weakref.ref(contribution)]
        batch.stream_of(fb).write(residual)
        output = batch.stream_of(fb).record(contribution, PLAIN_ADD)
        proxy = batch.to_pp(output, fb)
        del output, residual, contribution
        self.assertTrue(all(ref() is not None for ref in refs))
        self.assertIsNone(fb.residual_stream)
        del proxy
        self.assertTrue(all(ref() is None for ref in refs))

    def test_interleaved_batches_keep_independent_streams(self):
        from sglang.srt.layers.layer_boundary.residual import batch

        class Read:
            is_plain_norm = False

            def init_residual(self, value):
                return value

            def read(self, value, norm, quant_format="", **kwargs):
                return value * 2, value

            def update_and_read(self, update, value, residual, norm, **kwargs):
                updated = update.update(value, residual)
                return updated * 2, updated

        rows = Layout(frozenset())
        boundary = bind_entry(
            EdgeContract(
                OutputContract(rows), InputContract(rows, read=Read()), rows, rows
            )
        )
        owner = stub_plan()
        owner.norm = None
        owner.paths[BatchVariant.ORDINARY] = StagePath(
            EntryPath(boundary.prepare, rows), OutputContract(rows), None
        )
        owner.paths[BatchVariant.SEQUENCE_PARALLEL] = owner.paths[
            BatchVariant.INPUT_SCATTERED
        ] = owner.paths[BatchVariant.CONTEXT_PARALLEL] = None
        stage = stub_stage(owner, StageKind.FFN)
        a, b = [SimpleNamespace(forward_mode=ForwardMode.DECODE) for _ in range(2)]
        batch.start(a)
        batch.start(b)
        first = stage.prepare(torch.ones(2, 4), a)
        second = stage.prepare(torch.full((2, 4), 10.0), b)
        first = batch.stream_of(a).record(first * 3, PLAIN_ADD)
        second = batch.stream_of(b).record(second * 5, PLAIN_ADD)
        torch.testing.assert_close(stage.prepare(first, a), torch.full((2, 4), 14.0))
        torch.testing.assert_close(stage.prepare(second, b), torch.full((2, 4), 220.0))
        self.assertIsNot(batch.stream_of(a), batch.stream_of(b))
        batch.start(a)
        self.assertIsNone(batch.stream_of(a).pending)
        self.assertIsNotNone(batch.stream_of(b).residual)


if __name__ == "__main__":
    unittest.main()
