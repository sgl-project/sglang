"""A layer stack connects the stages appended to it in order."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import test_declared_decoder_boundary as fixture
import torch
from torch import nn

from sglang.srt.layers.layer_boundary import (
    BatchVariant,
    ExitRows,
    ProducerReduction,
    ReadoutFusion,
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
)
from sglang.srt.layers.layer_boundary import prepare as boundary_prepare
from sglang.srt.layers.layer_boundary.layout import SumGroup, TokenAxis
from sglang.srt.layers.layer_boundary.ops import (
    attn_tp_gather_input,
    keep_output,
    update_attn_tp_gather_output,
)
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    PLAIN_ADD,
    REPLACE_AT_EXIT,
    NormQuantReadout,
)
from sglang.srt.layers.rotary_embedding import factory as rope_factory
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.utils.common import is_building_neighbour_layer, make_layers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def layer(sparse=False):
    """One decoder layer's stages: attention, then an FFN."""
    return append_stages(
        (declare_attn(), fixture.Norm()),
        (declare_ffn(sparse=sparse, next_layer_sparse=sparse), fixture.Norm()),
    )


def mixer():
    """A single-stage mixer whose exit depends on the stage after it."""
    return declare_attn(
        reduction=ProducerReduction.EXIT_SCOPED, gathers_attn_tp_input=False
    )


class TestAppendStages(CustomTestCase):
    def setUp(self):
        self.planning = fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2))
        self.planning.__enter__()
        self.addCleanup(self.planning.__exit__, None, None, None)

    def test_only_the_stacks_last_stage_ends_it(self):
        with layer_stack():
            stages = [s for _ in range(3) for s in layer()]
        self.assertEqual([s.plan.terminal for s in stages], [False] * 5 + [True])
        self.assertEqual([s.plan.enters_stack for s in stages], [True] + [False] * 5)

    def test_a_one_layer_stack_starts_and_ends_on_that_layer(self):
        # A NextN / MTP draft builds its single layer in a stack of its own.
        with layer_stack():
            attention, ffn = layer()
        self.assertTrue(attention.plan.enters_stack)
        self.assertTrue(ffn.plan.terminal)

    def test_every_stage_binds_when_the_stack_closes(self):
        with layer_stack():
            stages = [s for _ in range(3) for s in layer()]
            # Declarations are usable at once; plans wait for the neighbours.
            self.assertTrue(all(s.declaration is not None for s in stages))
            self.assertTrue(all(s.plan is None for s in stages))
        self.assertTrue(all(s.plan is not None for s in stages))

    def test_appending_needs_an_open_stack(self):
        with self.assertRaisesRegex(RuntimeError, "open layer stack"):
            layer()

    def test_the_previous_layer_gives_the_first_stage_its_producer(self):
        built = []

        def previous_layer():
            built.append(None)
            layer(sparse=True)

        with layer_stack(previous_layers=[previous_layer]):
            first, _ = layer()
        self.assertEqual(len(built), 1)
        self.assertTrue(first.declaration.previous.sparse)
        self.assertFalse(first.plan.enters_stack)

    def test_the_next_layer_gives_the_last_stage_its_consumer(self):
        with layer_stack(
            next_layers=[lambda: append_stages((declare_ffn(), fixture.Norm()))]
        ):
            (stage,) = append_stages((mixer(), fixture.Norm()))
        self.assertFalse(stage.plan.terminal)
        # It leaves its sum to the FFN that follows it, which completes it.
        for edge in stage.plan.edges.values():
            self.assertTrue(edge.outgoing.produced.always_partial)

    def test_a_stack_without_stages_builds_no_neighbour(self):
        def unexpected():
            self.fail("a neighbour was built for a stack without stages")

        with layer_stack(previous_layers=[unexpected], next_layers=[unexpected]):
            pass

    def test_neighbours_are_built_after_the_stacks_own_layers(self):
        order = []

        def neighbour():
            order.append("neighbour")
            layer()

        with layer_stack(previous_layers=[neighbour], next_layers=[neighbour]):
            order.append("own")
            layer()
        self.assertEqual(order, ["own", "neighbour", "neighbour"])

    def test_a_neighbour_without_stages_is_passed_over(self):
        with layer_stack(previous_layers=[lambda: None, lambda: layer(sparse=True)]):
            first, _ = layer()
        self.assertTrue(first.declaration.previous.sparse)
        with layer_stack(previous_layers=[lambda: None], next_layers=[lambda: None]):
            first, last = layer()
        # No earlier layer declares a stage: the stack starts here, and ends
        # here when no later one does.
        self.assertTrue(first.plan.enters_stack)
        self.assertTrue(last.plan.terminal)

    def test_a_neighbour_is_read_without_joining_the_stack(self):
        def previous_layer():
            # A layer with a dense FFN, then one with a MoE: the producer is
            # the nearest stage, and neither joins this stack.
            layer(sparse=False)
            layer(sparse=True)

        with layer_stack(previous_layers=[previous_layer]):
            stages = layer()
        self.assertTrue(stages[0].declaration.previous.sparse)
        self.assertIsNone(stages[0].declaration.previous.previous)

    def test_a_branch_neither_extends_the_stack_nor_waits(self):
        with layer_stack():
            attention, moe = layer(sparse=True)
            branch = append_stages(
                (declare_ffn(), fixture.Norm()),
                (declare_attn(), fixture.Norm()),
                (declare_ffn(), fixture.Norm()),
                prepared_from=moe.declaration,
            )
            following = layer()
        self.assertTrue(all(s.plan is not None for s in branch))
        # The branch reads the MoE's input; the next layer follows the main
        # line's MoE, not the branch's dense FFN.
        self.assertEqual(branch[0].declaration.prepared_from, moe.declaration)
        self.assertTrue(following[0].declaration.previous.sparse)

    def test_a_nested_stack_leaves_the_outer_one_untouched(self):
        with layer_stack():
            outer = layer()
            with layer_stack():
                inner = layer()
            later = layer()
        self.assertTrue(inner[0].plan.enters_stack and inner[1].plan.terminal)
        # The outer stack resumes from its own last stage.
        self.assertFalse(outer[1].plan.terminal)
        self.assertFalse(later[0].plan.enters_stack)
        self.assertTrue(later[1].plan.terminal)

    def test_options_a_stage_does_not_take_are_rejected_when_appended(self):
        with layer_stack():
            with self.assertRaisesRegex(TypeError, "ATTENTION stage options"):
                append_stages((declare_attn(), fixture.Norm(), {"exit_rows": None}))
            with self.assertRaisesRegex(TypeError, "FFN stage options"):
                append_stages(
                    (declare_ffn(), fixture.Norm(), {"qkv_latent_func": None})
                )

    def test_a_branch_reads_a_stage_of_its_own_stack(self):
        with self.assertRaisesRegex(ValueError, "prepared_from"):
            with layer_stack():
                layer()
                append_stages(
                    (declare_ffn(), fixture.Norm()), prepared_from=declare_attn()
                )

    def test_a_branch_reads_the_stage_it_names_when_a_declaration_is_reused(self):
        moe = declare_ffn(sparse=True, next_layer_sparse=True)
        with layer_stack():
            _, first = append_stages(
                (declare_attn(), fixture.Norm()), (moe, fixture.Norm())
            )
            _, second = append_stages(
                (declare_attn(), fixture.Norm()), (moe, fixture.Norm())
            )
            (branch,) = append_stages(
                (declare_ffn(), fixture.Norm()), prepared_from=first.declaration
            )
        self.assertIsNot(first.declaration, second.declaration)
        self.assertIs(branch.declaration.prepared_from, first.declaration)

    @unittest.skipUnless(hasattr(BaseException, "add_note"), "needs add_note")
    def test_an_error_at_close_names_the_append_it_binds(self):
        # The stack binds after the layers that appended to it have returned,
        # so the traceback does not reach the layer; the note names it.
        with self.assertRaises(ValueError) as caught:
            with layer_stack():
                layer()
                append_stages(
                    (declare_attn(), fixture.Norm()),
                    (declare_ffn(dense_tp_size=3), fixture.Norm()),
                )
                layer()
        (note,) = caught.exception.__notes__
        self.assertRegex(
            note, r"test_append_stages\.py:\d+ \(append 1 of its layer stack\)"
        )


NUM_LAYERS = 4


class TestMakeLayers(CustomTestCase):
    """make_layers builds its layers in one stack, and on a pipeline stage
    reads the stages the other stages' layers declare next to it."""

    def setUp(self):
        self.planning = fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2))
        self.planning.__enter__()
        self.addCleanup(self.planning.__exit__, None, None, None)

    def build(self, pp_rank=None, pp_size=None, *, declares=True):
        """Build the layers; layer 1 has a MoE. Returns each layer_fn call as
        (index, prefix, device, built as a neighbour) and the stages of the
        layers built here."""
        calls, stages = [], {}

        def layer_fn(idx, prefix):
            device = torch.empty(()).device.type
            calls.append((idx, prefix, device, is_building_neighbour_layer()))
            if declares:
                stages.setdefault(idx, layer(sparse=idx == 1))
            return nn.Identity()

        make_layers(
            NUM_LAYERS, layer_fn, pp_rank=pp_rank, pp_size=pp_size, prefix="layers"
        )
        return calls, stages

    def test_the_layers_form_one_stack(self):
        _, stages = self.build()
        stages = [s for idx in range(NUM_LAYERS) for s in stages[idx]]
        self.assertEqual([s.plan.terminal for s in stages], [False] * 7 + [True])
        self.assertEqual([s.plan.enters_stack for s in stages], [True] + [False] * 7)

    def test_a_pipeline_stage_builds_its_neighbours_on_meta_after_its_layers(self):
        calls, _ = self.build(pp_rank=0, pp_size=2)
        self.assertEqual(
            calls,
            [
                (0, "layers.0", "cpu", False),
                (1, "layers.1", "cpu", False),
                (2, "layers.2", "meta", True),
            ],
        )
        calls, _ = self.build(pp_rank=1, pp_size=2)
        self.assertEqual(
            calls,
            [
                (2, "layers.2", "cpu", False),
                (3, "layers.3", "cpu", False),
                (1, "layers.1", "meta", True),
            ],
        )

    def test_the_neighbours_connect_the_stack_across_pipeline_stages(self):
        _, first = self.build(pp_rank=0, pp_size=2)
        self.assertTrue(first[0][0].plan.enters_stack)
        self.assertFalse(first[1][1].plan.terminal)
        _, last = self.build(pp_rank=1, pp_size=2)
        # Layer 2's attention follows layer 1's MoE, on the other stage.
        self.assertTrue(last[2][0].declaration.previous.sparse)
        self.assertFalse(last[2][0].plan.enters_stack)
        self.assertTrue(last[3][1].plan.terminal)

    def test_layers_without_stages_build_no_neighbour(self):
        calls, _ = self.build(pp_rank=1, pp_size=2, declares=False)
        self.assertEqual([call[0] for call in calls], [2, 3])

    def test_a_neighbour_s_rope_modules_leave_the_cache(self):
        def layer_fn(idx, prefix):
            rope_factory._ROPE_DICT[("test", idx)] = None
            layer()
            return nn.Identity()

        with patch.dict(rope_factory._ROPE_DICT):
            make_layers(NUM_LAYERS, layer_fn, pp_rank=0, pp_size=2)
            added = {key for key in rope_factory._ROPE_DICT if key[0] == "test"}
        self.assertEqual(added, {("test", 0), ("test", 1)})


class TestPipelineHandoff(CustomTestCase):
    """With an a2a MoE over attention TP, an FFN leaves its output on this
    rank's slice of the rows for the next layer to gather; across a pipeline
    handoff it returns to the attention's rows, which both ranks agree on."""

    def setUp(self):
        self.planning = fixture.planning(
            fixture.parallel_of(attn_dp=1, attn_tp=2),
            a2a=True,
            boundary_reduction="ar",
        )
        self.planning.__enter__()
        self.addCleanup(self.planning.__exit__, None, None, None)

    @staticmethod
    def path(stage):
        return stage.plan.paths[BatchVariant.ORDINARY]

    def test_the_sending_rank_returns_to_the_attention_rows(self):
        with layer_stack(next_layers=[partial(layer, sparse=True)]):
            first_attn, first_ffn, second_attn, second_ffn = [
                s for _ in range(2) for s in layer(sparse=True)
            ]
        # Within the rank the output stays on the slice, and the next
        # attention gathers its input.
        self.assertIs(self.path(first_ffn).output_move, keep_output)
        self.assertIs(self.path(second_attn).entry.input_move, attn_tp_gather_input)
        self.assertIs(
            self.path(second_ffn).output_move.func, update_attn_tp_gather_output
        )
        self.assertFalse(self.path(first_ffn).writes_at_handoff)
        self.assertTrue(self.path(second_ffn).writes_at_handoff)

    def test_the_handoff_writes_what_the_move_leaves(self):
        # A move that leaves the residual, as for an FFN on an unpadded batch's
        # rows: the exit still writes the output into it.
        with layer_stack(next_layers=[partial(layer, sparse=True)]):
            ffn = [s for s in layer(sparse=True)][-1]
        steps = msgspec.structs.replace(self.path(ffn), output_move=keep_output)
        hidden, residual = torch.ones(4, 4), torch.full((4, 4), 2.0)
        want = hidden + residual
        written, left = ffn.plan.output._complete_now(
            hidden,
            residual,
            forward_batch=None,
            dp_step=None,
            steps=steps,
            output_move=keep_output,
            owes_sum=False,
        )
        self.assertIsNone(left)
        torch.testing.assert_close(written, want)

    def test_the_receiving_rank_reads_the_attention_rows(self):
        with layer_stack(previous_layers=[partial(layer, sparse=True)]):
            attention, _ = layer(sparse=True)
        self.assertIsNone(self.path(attention).entry.input_move)
        self.assertIs(attention.declaration.previous.exit_rows, ExitRows.ATTENTION)

    def test_the_receiving_rank_reads_the_stream_as_written(self):
        # A residual beside it, as a captured graph's input buffer holds, is
        # not read.
        with layer_stack(previous_layers=[partial(layer, sparse=True)]):
            attention, _ = layer(sparse=True)
        attention.plan.path_for = lambda _: self.path(attention)
        hidden, stale = torch.ones(4, 4), torch.full((4, 4), 2.0)
        for wire in (
            {"hidden_states": hidden},
            {"hidden_states": hidden, "residual": stale},
        ):
            with self.subTest(keys=sorted(wire)):
                batch = SimpleNamespace(residual_stream=None)
                attention.from_pp(PPProxyTensors(wire), batch)
                self.assertIs(batch.residual_stream.residual, hidden)
                self.assertIsNone(batch.residual_stream.pending)

    def test_a_handoff_from_the_attention_rows_carries_its_residual(self):
        # A dense FFN over the TP group computes on the attention's rows and
        # leaves the residual add to the receiver, which requires it.
        with layer_stack(previous_layers=[partial(layer, sparse=False)]):
            attention, _ = layer(sparse=False)
        self.assertFalse(attention.declaration.previous.writes_at_handoff)
        attention.plan.path_for = lambda _: self.path(attention)
        hidden, residual = torch.ones(4, 4), torch.full((4, 4), 2.0)
        batch = SimpleNamespace(residual_stream=None)
        wire = PPProxyTensors({"hidden_states": hidden, "residual": residual})
        attention.from_pp(wire, batch)
        self.assertIs(batch.residual_stream.residual, residual)
        self.assertIs(batch.residual_stream.pending.value, hidden)
        with self.assertRaises(KeyError):
            attention.from_pp(PPProxyTensors({"hidden_states": hidden}), batch)


class TestUnpaddedBatches(CustomTestCase):
    """Without attention DP, --disable-attn-tp-gather lets a batch reach the
    stages with rows that do not divide over attention TP. Such a batch keeps
    an FFN that would run on this rank's attention-TP slice on the
    attention's rows instead."""

    def build(self, parallel, *, sparse):
        with fixture.planning(parallel, a2a=True, boundary_reduction="ar"):
            with layer_stack():
                return [s for _ in range(2) for s in layer(sparse=sparse)]

    def test_an_unpadded_batch_keeps_the_attention_rows(self):
        for sparse in (True, False):
            with self.subTest(sparse=sparse):
                parallel = fixture.parallel_of(
                    attn_dp=1,
                    attn_tp=2,
                    disable_attn_tp_gather=True,
                    moe_dense_tp_size=None if sparse else 1,
                )
                _, ffn, attention, _ = self.build(parallel, sparse=sparse)
                ordinary = ffn.plan.paths[BatchVariant.ORDINARY]
                unpadded = ffn.plan.paths[BatchVariant.UNPADDED]
                # Rows that divide take the slice; the others complete the
                # attention's sum on every row and hand on a complete output.
                self.assertIn(TokenAxis.ATTN_TP, ordinary.entry.input_rows.sharded)
                self.assertEqual(unpadded.entry.input_rows.sharded, frozenset())
                self.assertIs(
                    unpadded.entry.prepare.keywords["step"].func,
                    boundary_prepare._reduce_update_read,
                )
                self.assertIsNone(unpadded.output.group)
                self.assertIs(unpadded.output_move, keep_output)
                # The next attention reads those rows as they are.
                self.assertIs(
                    attention.plan.paths[BatchVariant.ORDINARY].entry.input_move,
                    attn_tp_gather_input,
                )
                self.assertIsNone(
                    attention.plan.paths[BatchVariant.UNPADDED].entry.input_move
                )

    def test_padded_batches_have_no_unpadded_path(self):
        for attn_dp, disabled in ((2, True), (1, False)):
            with self.subTest(attn_dp=attn_dp, disable_attn_tp_gather=disabled):
                parallel = fixture.parallel_of(
                    attn_dp=attn_dp, attn_tp=2, disable_attn_tp_gather=disabled
                )
                _, ffn, _, _ = self.build(parallel, sparse=True)
                self.assertNotIn(BatchVariant.UNPADDED, ffn.plan.paths)


class TestRowsTheConsumerReads(CustomTestCase):
    """An FFN on this rank's attention-TP slice leaves its output there unless
    the stage after it needs the attention's rows: a read that needs every row
    (reads_after_attn_tp_gather) has the FFN's exit gather them, in that
    stage's declared gather. At the stack's end the model's final read takes
    that place: one that reads the slice (reads_attn_tp_slices) takes the
    last FFN's output there, and its own gather serves the last FFN's exit."""

    class EveryRowRead(NormQuantReadout):
        reads_after_attn_tp_gather = True

    def build(
        self,
        *,
        final_read=None,
        disable_attn_tp_gather=False,
        gather=None,
        read=None,
    ):
        parallel = fixture.parallel_of(
            attn_dp=1, attn_tp=2, disable_attn_tp_gather=disable_attn_tp_gather
        )
        attn_read = {} if read is None else dict(read=read)
        with fixture.planning(parallel, a2a=True, boundary_reduction="ar"):
            with layer_stack(final_read=final_read):
                return [
                    stage
                    for idx in range(2)
                    for stage in append_stages(
                        (
                            declare_attn(
                                attn_tp_gather=gather if idx else None, **attn_read
                            ),
                            fixture.Norm(),
                        ),
                        (
                            declare_ffn(sparse=True, next_layer_sparse=True),
                            fixture.Norm(),
                        ),
                    )
                ]

    def test_the_last_ffn_stays_on_its_slice_for_a_read_of_the_slice(self):
        for reads_slices in (True, False):
            with self.subTest(reads_slices=reads_slices):
                final_read = SimpleNamespace(reads_attn_tp_slices=reads_slices)
                last = self.build(final_read=final_read)[-1]
                path = last.plan.paths[BatchVariant.ORDINARY]
                if reads_slices:
                    self.assertIs(path.output_move, keep_output)
                else:
                    self.assertIs(path.output_move.func, update_attn_tp_gather_output)

    def test_the_last_ffn_gathers_in_the_final_read_gather(self):
        def gather(hidden_states):
            return None

        last = self.build(final_read=SimpleNamespace(attn_tp_gather=gather))[-1]
        move = last.plan.paths[BatchVariant.ORDINARY].output_move
        self.assertIs(move.func, update_attn_tp_gather_output)
        self.assertIs(move.keywords["gather"], gather)

    def test_an_unpadded_batch_leaves_the_last_ffn_on_the_attention_rows(self):
        last = self.build(
            final_read=SimpleNamespace(reads_attn_tp_slices=True),
            disable_attn_tp_gather=True,
        )[-1]
        unpadded = last.plan.paths[BatchVariant.UNPADDED]
        self.assertEqual(unpadded.output.layout.sharded, frozenset())
        self.assertIs(unpadded.output_move, keep_output)

    def test_the_next_attention_gathers_in_the_declared_gather(self):
        def gather(hidden_states):
            return None

        first_ffn, attention = self.build(gather=gather)[1:3]
        move = attention.plan.paths[BatchVariant.ORDINARY].entry.input_move
        self.assertIs(move.func, attn_tp_gather_input)
        self.assertIs(move.keywords["gather"], gather)
        path = first_ffn.plan.paths[BatchVariant.ORDINARY]
        self.assertIs(path.output_move, keep_output)

    def test_a_read_of_every_row_has_the_ffn_gather_in_its_gather(self):
        def gather(hidden_states):
            return None

        first_ffn, attention = self.build(gather=gather, read=self.EveryRowRead())[1:3]
        move = first_ffn.plan.paths[BatchVariant.ORDINARY].output_move
        self.assertIs(move.func, update_attn_tp_gather_output)
        self.assertIs(move.keywords["gather"], gather)
        entry = attention.plan.paths[BatchVariant.ORDINARY].entry
        self.assertIsNone(entry.input_move)
        self.assertEqual(entry.input_rows.sharded, frozenset())

    def test_a_read_that_gathers_needs_a_plain_add_or_a_written_stream(self):
        def gather_read(hidden_states, residual, norm):
            return None

        class GatheringRead(NormQuantReadout):
            gathering_reads = (gather_read,)

        class Scaled:
            """Neither a plain add nor applied at the producer's exit."""

            is_plain_add = False
            applied_at_exit = False
            outlives_layer = True

        # One pipeline rank: an update other than a plain add can't cross one.
        parallel = fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=1)
        for update, binds in (
            (PLAIN_ADD, True),
            (REPLACE_AT_EXIT, True),
            (Scaled(), False),
        ):
            with self.subTest(update=type(update).__name__):
                with (
                    fixture.planning(parallel, a2a=True, boundary_reduction="ar"),
                    layer_stack(final_read=SimpleNamespace(reads_attn_tp_slices=True)),
                ):
                    stages = [
                        stage
                        for _ in range(2)
                        for stage in append_stages(
                            (declare_attn(read=GatheringRead()), fixture.Norm()),
                            (
                                declare_ffn(
                                    sparse=True,
                                    next_layer_sparse=True,
                                    update=update,
                                ),
                                fixture.Norm(),
                            ),
                        )
                    ]
                # The stack binds its stages when it closes.
                entry = stages[2].plan.paths[BatchVariant.ORDINARY].entry
                self.assertEqual(
                    entry.prepare.keywords["step"].keywords["read_gathers"],
                    (gather_read,) if binds else (),
                )


class TestReadKernels(CustomTestCase):
    """The entry steps around a read's own kernels: one that reads and gathers
    hands its input to the entry's gather as already gathered, and one that
    completes the sum and reads returns the read."""

    def test_a_read_that_gathers_passes_through_the_entry_gather(self):
        gathered = torch.ones(4, 8)

        class Read:
            def read(self, residual, norm, quant_format=""):
                return residual * 2, residual

        def takes(hidden_states, residual, norm):
            return gathered, None

        def declines(hidden_states, residual, norm):
            return None

        shard = torch.randn(2, 8)
        common = dict(pre_move=None, enters_stack=False, read=Read())
        got, residual = boundary_prepare._update_read(
            shard, None, None, None, read_gathers=(declines, takes), **common
        )
        self.assertIs(attn_tp_gather_input(got, None), gathered)
        self.assertIsNone(residual)
        got, residual = boundary_prepare._update_read(
            shard, None, None, None, read_gathers=(declines,), **common
        )
        torch.testing.assert_close(got, shard * 2, rtol=0, atol=0)

    def test_a_kernel_that_reads_returns_the_read(self):
        read_result = (torch.zeros(2, 8), torch.ones(2, 8))

        def reads(takes):
            def run(hidden_states, residual, forward_batch, norm):
                return read_result if takes else None

            return ReadoutFusion(SumGroup.ATTN_TP, run, scatters=True, reads=True)

        class Read:
            def update_and_read(self, update, hidden_states, residual, norm):
                return "boundary", residual

        class Update:
            def slice_residual_attn_tp(self, residual):
                return residual[:2]

        call = dict(scatters_residual=True, read=Read(), update=Update())
        hidden, residual = torch.randn(4, 8), torch.randn(4, 8)
        with patch.object(boundary_prepare, "attn_tp_reduce_scatter", lambda h: h[:2]):
            got = boundary_prepare._attn_tp_reduce_scatter_update_read(
                hidden, residual, None, None, read_fusions=(reads(True),), **call
            )
            self.assertIs(got, read_result)
            got, _ = boundary_prepare._attn_tp_reduce_scatter_update_read(
                hidden, residual, None, None, read_fusions=(reads(False),), **call
            )
            self.assertEqual(got, "boundary")


class TestDenseFfnOverAttentionTp(CustomTestCase):
    """A dense FFN sharded over attention TP under attention DP computes on
    the attention's rows and sums over attention TP, with no DP move."""

    def test_the_ffn_stays_on_the_attention_rows(self):
        parallel = fixture.parallel_of(attn_dp=2, attn_tp=2)
        with fixture.planning(parallel, boundary_reduction="ar"), layer_stack():
            stages = [
                stage
                for _ in range(2)
                for stage in append_stages(
                    (declare_attn(), fixture.Norm()),
                    (declare_ffn(dense_tp_size=2), fixture.Norm()),
                )
            ]
        for ffn in stages[1::2]:
            path = ffn.plan.paths[BatchVariant.ORDINARY]
            # The attention's sum completes on this rank's rows, ungathered.
            self.assertIs(
                path.entry.prepare.keywords["step"].func,
                boundary_prepare._reduce_update_read,
            )
            self.assertEqual(path.output.layout.sharded, {TokenAxis.ATTN_DP})
            self.assertIs(path.output.group, SumGroup.ATTN_TP)
            self.assertFalse(path.returns_over_dp)
            self.assertIs(path.output_move, keep_output)


if __name__ == "__main__":
    unittest.main()
