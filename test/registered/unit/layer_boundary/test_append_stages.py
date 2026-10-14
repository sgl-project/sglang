"""A layer stack connects the stages appended to it in order."""

import threading
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
from sglang.srt.layers.layer_boundary.facts import facts_of
from sglang.srt.layers.layer_boundary.layout import SumGroup, TokenAxis
from sglang.srt.layers.layer_boundary.ops import (
    attn_tp_gather_input,
    keep_output,
    update_attn_tp_gather_output,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    PLAIN_ADD,
    REPLACE_AT_EXIT,
    NormQuantReadout,
)
from sglang.srt.layers.layer_boundary.residual.stream import DeclaredSum
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.utils.common import make_layers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def layer_stages(sparse=False):
    """The stages one decoder layer declares: attention, then an FFN. A
    pipeline rank reads them for a layer another rank holds."""
    return (
        declare_attn(),
        declare_ffn(sparse=sparse, next_layer_sparse=sparse),
    )


def layer(sparse=False):
    """One decoder layer, declaring layer_stages."""
    return append_stages(
        *((declaration, fixture.Norm()) for declaration in layer_stages(sparse))
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
        read = []

        def previous_layer():
            read.append(None)
            return layer_stages(sparse=True)

        with layer_stack(previous_layers=[previous_layer]):
            first, _ = layer()
        self.assertEqual(len(read), 1)
        self.assertTrue(first.declaration.previous.sparse)
        self.assertFalse(first.plan.enters_stack)

    def test_the_next_layer_gives_the_last_stage_its_consumer(self):
        with layer_stack(next_layers=[lambda: (declare_ffn(),)]):
            (stage,) = append_stages((mixer(), fixture.Norm()))
        self.assertFalse(stage.plan.terminal)
        # It leaves its sum to the FFN that follows it, which completes it.
        for edge in stage.plan.edges.values():
            self.assertTrue(edge.outgoing.produced.always_partial)

    def test_a_stack_without_stages_reads_no_neighbour(self):
        def unexpected():
            self.fail("a neighbour was read for a stack without stages")

        with layer_stack(previous_layers=[unexpected], next_layers=[unexpected]):
            pass

    def test_neighbours_are_read_after_the_stacks_own_layers(self):
        order = []

        def neighbour():
            order.append("neighbour")
            return layer_stages()

        with layer_stack(previous_layers=[neighbour], next_layers=[neighbour]):
            order.append("own")
            layer()
        self.assertEqual(order, ["own", "neighbour", "neighbour"])

    def test_a_neighbour_without_stages_is_passed_over(self):
        with layer_stack(
            previous_layers=[lambda: (), lambda: layer_stages(sparse=True)]
        ):
            first, _ = layer()
        self.assertTrue(first.declaration.previous.sparse)
        with layer_stack(previous_layers=[lambda: ()], next_layers=[lambda: ()]):
            first, last = layer()
        # No earlier layer declares a stage: the stack starts here, and ends
        # here when no later one does.
        self.assertTrue(first.plan.enters_stack)
        self.assertTrue(last.plan.terminal)

    def test_a_neighbour_is_read_without_joining_the_stack(self):
        def previous_layer():
            # A dense FFN, then a MoE: the producer is the nearest stage, and
            # neither joins this stack.
            return (declare_ffn(sparse=False), declare_ffn(sparse=True))

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
        self.assertEqual(branch[0].declaration.prepared_from, facts_of(moe.declaration))
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
        self.assertEqual(branch.declaration.prepared_from, facts_of(first.declaration))

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


class TestOneStagePerAppend(CustomTestCase):
    """Appends of one stage each: an attention, then an FFN, alternately."""

    def test_both_sides_of_an_attention_s_edge_agree_across_appends(self):
        # An a2a MoE returns its output on this rank's attention-TP slice,
        # so the next attention keeps its residual there; the following FFN
        # must start from that slice, not from the attention's full rows.
        with fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2), a2a=True):
            with layer_stack():
                stages = [
                    append_stages(
                        (
                            declare_attn() if i % 2 == 0 else declare_ffn(sparse=True),
                            fixture.Norm(),
                        )
                    )[0]
                    for i in range(4)
                ]
        for producer, consumer in zip(stages, stages[1:]):
            if not producer.plan.finishes_directly:
                continue
            for variant, edges in producer.plan.edges.items():
                with self.subTest(variant=variant.name):
                    incoming = consumer.plan.edges[variant].incoming
                    self.assertEqual(edges.outgoing.residual, incoming.residual)
                    self.assertEqual(edges.outgoing.residual_to, incoming.residual_to)

    def test_a_stack_that_ends_on_an_attention_leaves_the_residual_on_its_rows(self):
        # No exit moves the residual after the last attention: the next rank,
        # or the final read, takes it on the rows that attention ran on, so
        # the a2a MoE before it returns it on the attention's rows.
        def ffn_layer():
            return (declare_ffn(sparse=True),)

        for next_layers in ((), (ffn_layer,)):
            with self.subTest(hands_off=bool(next_layers)):
                with fixture.planning(
                    fixture.parallel_of(attn_dp=1, attn_tp=2), a2a=True
                ):
                    with layer_stack(next_layers=next_layers):
                        stages = [
                            append_stages(
                                (
                                    declare_attn()
                                    if i % 2 == 0
                                    else declare_ffn(sparse=True),
                                    fixture.Norm(),
                                )
                            )[0]
                            for i in range(3)
                        ]
                for variant, edges in stages[-1].plan.edges.items():
                    self.assertEqual(
                        edges.incoming.residual_to, edges.incoming.need.layout
                    )


def stage_facts(idx):
    """The shared declaration function of TestMakeLayers' layers: layer 1
    has a MoE."""
    return layer_stages(sparse=idx == 1)


class TestMakeLayers(CustomTestCase):
    """make_layers builds its layers in one stack, and on a pipeline stage
    reads the stages the other stages' layers declare next to it from the
    model's shared declaration function."""

    def setUp(self):
        self.planning = fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2))
        self.planning.__enter__()
        self.addCleanup(self.planning.__exit__, None, None, None)

    def build(self, pp_rank=None, pp_size=None, *, declares=True, facts=stage_facts):
        """Build the layers. Returns the indices layer_fn built and the stages
        of the layers built here."""
        built, stages = [], {}

        def layer_fn(idx, prefix):
            self.assertEqual(prefix, f"layers.{idx}")
            built.append(idx)
            if declares:
                stages[idx] = layer(sparse=idx == 1)
            return nn.Identity()

        make_layers(
            NUM_LAYERS,
            layer_fn,
            pp_rank=pp_rank,
            pp_size=pp_size,
            prefix="layers",
            stage_facts=facts,
        )
        return built, stages

    def test_the_layers_form_one_stack(self):
        _, stages = self.build()
        stages = [s for idx in range(NUM_LAYERS) for s in stages[idx]]
        self.assertEqual([s.plan.terminal for s in stages], [False] * 7 + [True])
        self.assertEqual([s.plan.enters_stack for s in stages], [True] + [False] * 7)

    def test_a_pipeline_stage_builds_only_its_own_layers(self):
        self.assertEqual(self.build(pp_rank=0, pp_size=2)[0], [0, 1])
        self.assertEqual(self.build(pp_rank=1, pp_size=2)[0], [2, 3])

    def test_the_declaration_function_connects_the_stack_across_stages(self):
        _, first = self.build(pp_rank=0, pp_size=2)
        self.assertTrue(first[0][0].plan.enters_stack)
        self.assertFalse(first[1][1].plan.terminal)
        _, last = self.build(pp_rank=1, pp_size=2)
        # Layer 2's attention follows layer 1's MoE, on the other stage.
        self.assertTrue(last[2][0].declaration.previous.sparse)
        self.assertFalse(last[2][0].plan.enters_stack)
        self.assertTrue(last[3][1].plan.terminal)

    def test_a_layer_must_declare_what_its_declaration_function_says(self):
        def layer_fn(idx, prefix):
            layer(sparse=idx == 2)
            return nn.Identity()

        with self.assertRaisesRegex(
            ValueError,
            r"layer 2 declares stages .*: stage 1 \(FFN\): sparse is True, "
            "the function says False",
        ):
            make_layers(
                NUM_LAYERS,
                layer_fn,
                pp_rank=1,
                pp_size=2,
                stage_facts=lambda idx: (declare_attn(), declare_ffn()),
            )

    def test_layers_with_stages_need_a_declaration_function(self):
        with self.assertRaisesRegex(ValueError, "stage_facts"):
            self.build(pp_rank=1, pp_size=2, facts=None)

    def test_layers_without_stages_need_no_declaration_function(self):
        built, _ = self.build(pp_rank=1, pp_size=2, declares=False, facts=None)
        self.assertEqual(built, [2, 3])


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
        with layer_stack(next_layers=[partial(layer_stages, sparse=True)]):
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
        with layer_stack(next_layers=[partial(layer_stages, sparse=True)]):
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
        with layer_stack(previous_layers=[partial(layer_stages, sparse=True)]):
            attention, _ = layer(sparse=True)
        self.assertIsNone(self.path(attention).entry.input_move)
        self.assertIs(attention.declaration.previous.exit_rows, ExitRows.ATTENTION)

    def test_the_receiving_rank_reads_the_stream_as_written(self):
        # A residual beside it, as a captured graph's input buffer holds, is
        # not read.
        with layer_stack(previous_layers=[partial(layer_stages, sparse=True)]):
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
        with layer_stack(previous_layers=[partial(layer_stages, sparse=False)]):
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


class TestInputScatteredHandoff(CustomTestCase):
    """On an input-scattered batch the attention keeps the residual on this
    rank's attention-TP slice; the FFN that hands off to the next pipeline
    rank runs and exits with the residual on one set of rows."""

    def test_an_ffn_s_entry_leaves_the_rows_its_exit_starts_from(self):
        parallel = fixture.parallel_of(
            attn_dp=1, attn_tp=4, enable_attn_tp_input_scattered=True
        )
        with fixture.planning(parallel):
            with layer_stack(next_layers=[layer_stages]):
                stages = [s for _ in range(2) for s in layer(sparse=True)]
        for stage in stages:
            for variant, edges in stage.plan.edges.items():
                with self.subTest(kind=stage.kind.name, variant=variant.name):
                    self.assertEqual(
                        edges.incoming.residual_to, edges.outgoing.residual
                    )


# The stages of a stack cut between pipeline ranks.
CUT_STAGES = {
    "attention": declare_attn,
    "mixer": mixer,
    "dense": declare_ffn,
    "sparse": partial(declare_ffn, sparse=True),
}
CUT_ROWS = 8
# A stack binds when it closes, in the one stack the module has open: the
# simulated ranks bind theirs one at a time.
BINDING = threading.Lock()


def bind_cut(kinds, *, before=(), after=()):
    """One rank's stages: ``kinds``, with ``before`` and ``after`` held by
    the pipeline ranks next to it."""

    def held_by_another_rank(kinds):
        return [lambda: tuple(CUT_STAGES[kind]() for kind in kinds)] if kinds else []

    with (
        BINDING,
        layer_stack(
            previous_layers=held_by_another_rank(before),
            next_layers=held_by_another_rank(after),
        ),
    ):
        return append_stages(*((CUT_STAGES[kind](), fixture.Norm()) for kind in kinds))


class Logged(fixture.Group):
    """A group that logs each collective, by kind, on the rank that runs it."""

    def all_reduce(self, x):
        fixture.state().log.append(("all_reduce", self.name))
        return super().all_reduce(x)

    def reduce_scatter_tensor(self, output, input):
        fixture.state().log.append(("reduce_scatter", self.name))
        super().reduce_scatter_tensor(output, input)

    def all_gather_into_tensor(self, output, input):
        fixture.state().log.append(("all_gather", self.name))
        super().all_gather_into_tensor(output, input)


def computed(kind, hidden, state):
    """What a stage computes on one rank: an attention or a mixer leaves its
    sum (3 * x) as this rank's attention-TP partial, a dense FFN its sum
    (5 * x) as this rank's TP partial, and an a2a MoE, on this rank's slice
    of the rows, a complete 5 * x."""
    if kind in ("attention", "mixer"):
        if hidden.shape[0] != CUT_ROWS:
            # An input-scattered attention's QKV hook gathers the rows.
            rows = hidden.new_empty(CUT_ROWS, hidden.shape[1])
            state.attention_gather.all_gather_into_tensor(rows, hidden)
            hidden = rows
        return fixture.attention(hidden, state)
    if kind == "dense":
        return fixture.dense_mlp(hidden, state)
    return 5 * hidden


def cut_reference(x, kinds):
    """The residual after ``kinds``: each stage reads Norm's 2 * residual and
    adds its output to it."""
    residual, output = x, None
    for kind in kinds:
        if output is not None:
            residual = output + residual
        output = (3 if kind in ("attention", "mixer") else 5) * (2 * residual)
    return output + residual


def cut_batch():
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_context_parallel_extend=lambda: False,
            is_decode_or_idle=lambda: False,
        ),
        dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
        global_dp_buffer_len=CUT_ROWS,
        input_ids=torch.zeros(CUT_ROWS),
        residual_stream=None,
    )


class TestValuesAcrossAPipelineCut(CustomTestCase):
    """A stack cut between two pipeline ranks computes what it does uncut, on
    every attention-TP rank, each a thread over collectives that wait for its
    whole group. The pipeline sends each tensor as one slice of its rows per
    attention-TP rank, and the next rank gathers the slices back: a sum left
    to the next rank's input would reach it as a mixture of the ranks'
    partials. So the sending rank completes, once, every sum its output still
    owes; the receiving rank sums nothing it receives, and adds the residual
    once."""

    def run_world(self, forward, *, attn_tp, scattered, a2a, use_reduce_scatter):
        world = fixture.World(attn_tp)
        fixture.WORLD[0] = world
        # Without attention DP, attention TP is the TP group.
        tp_group = Logged(world, "tp", range(attn_tp))
        groups = dict(
            # An input-scattered attention's own gather of its rows.
            attention_gather=Logged(world, "attention", range(attn_tp)),
            # The pipeline's gather of the slices it sent.
            pipeline=Logged(world, "pipeline", range(attn_tp)),
        )
        states = [
            SimpleNamespace(
                rank=rank,
                parallel=fixture.parallel_of(
                    attn_dp=1,
                    attn_tp=attn_tp,
                    tp_rank=rank,
                    attn_tp_rank=rank,
                    tp_group=tp_group,
                    attn_tp_group=tp_group,
                    enable_attn_tp_input_scattered=scattered,
                    pp_size=2,
                ),
                flags=fixture.Flags(),
                calls=[],
                log=[],
                dp=0,
                rows=CUT_ROWS,
                offset=0,
                local_rows=CUT_ROWS,
                global_rows=CUT_ROWS,
                dp_rows=[CUT_ROWS],
                **groups,
            )
            for rank in range(attn_tp)
        ]
        with (
            fixture.running(
                reduce_scatterv=False, a2a=a2a, use_reduce_scatter=use_reduce_scatter
            ),
            patch_communicator(
                "get_attn_tp_context",
                lambda: SimpleNamespace(
                    input_scattered=scattered,
                    is_dsa=False,
                    set_attn_inputs=lambda inputs: None,
                ),
            ),
        ):
            results, errors = world.run(states, lambda rank: forward(states[rank]))
        failed = [
            (rank, error) for rank, error in enumerate(errors) if error is not None
        ]
        if failed:
            # A rank that raises breaks the barriers the others wait at.
            rank, error = next(
                (
                    (rank, error)
                    for rank, error in failed
                    if not isinstance(error, threading.BrokenBarrierError)
                ),
                failed[0],
            )
            raise AssertionError(f"rank {rank} raised") from error
        # Every rank ran the same collectives.
        self.assertEqual(len({tuple(state.log) for state in states}), 1)
        return results, states

    @staticmethod
    def run_stages(stages, kinds, hidden, forward_batch, state):
        """Run ``stages`` on one rank. Records on ``state`` what each one's
        input owes as it prepares (``owed``), and where in the log of its
        collectives its prepare ends (``prepared``)."""
        for stage, kind in zip(stages, kinds):
            pending = forward_batch.residual_stream.pending
            state.owed.append(None if pending is None else pending.owed)
            hidden = stage.prepare(hidden, forward_batch)
            state.prepared.append(len(state.log))
            if stage.plan.finishes_directly:
                hidden = stage.finish(computed(kind, hidden, state), forward_batch)
            else:
                with stage.exit(forward_batch) as scope:
                    output = computed(kind, hidden, state)
                hidden = scope.finish(output)
        return hidden

    @staticmethod
    def transported(tensors, state):
        """What the next pipeline rank receives: each attention-TP rank sends
        its slice of every tensor's rows, and the receiver gathers them."""
        parallel = state.parallel
        received = {}
        for name, value in tensors.tensors.items():
            part = value.tensor_split(parallel.attn_tp_size)[parallel.attn_tp_rank]
            received[name] = value.new_empty(value.shape)
            state.pipeline.all_gather_into_tensor(received[name], part.contiguous())
        return PPProxyTensors(received)

    def run_cut(
        self,
        kinds,
        cut,
        *,
        attn_tp,
        scattered=False,
        a2a=False,
        use_reduce_scatter=True,
    ):
        """Run ``kinds`` on every rank uncut, then cut before ``kinds[cut]``.
        Returns the input, and each run's per-rank (output, residual) and
        states. A cut run's state also holds what the sending rank's output
        owed as it was sent (``sent``), the tensors sent (``keys``), and the
        collectives that sent them (``sending``) and that read them on the
        receiving rank, up to the end of its first prepare (``receiving``)."""
        x = torch.arange(1.0, CUT_ROWS * fixture.HIDDEN + 1).double()
        x = x.view(CUT_ROWS, fixture.HIDDEN)

        def embedded(state):
            if scattered:
                # The vocabulary-parallel embedding leaves its TP sum.
                return x * fixture.WEIGHTS[attn_tp][state.parallel.tp_rank]
            return x.clone()

        def uncut(state):
            state.owed, state.prepared = [], []
            stages = bind_cut(kinds)
            forward_batch = cut_batch()
            residual_batch.start(forward_batch)
            hidden = self.run_stages(
                stages, kinds, embedded(state), forward_batch, state
            )
            return forward_batch.residual_stream.export(hidden)

        def cut_at(state):
            state.owed, state.prepared = [], []
            sender = bind_cut(kinds[:cut], after=kinds[cut:])
            receiver = bind_cut(kinds[cut:], before=kinds[:cut])
            forward_batch = cut_batch()
            residual_batch.start(forward_batch)
            hidden = self.run_stages(
                sender, kinds[:cut], embedded(state), forward_batch, state
            )
            pending = forward_batch.residual_stream.pending
            state.sent = None if pending is None else pending.owed
            start = len(state.log)
            tensors = residual_batch.to_pp(hidden, forward_batch)
            state.sending = state.log[start:]
            state.keys = sorted(tensors.tensors)
            received = self.transported(tensors, state)
            start = len(state.log)
            forward_batch = cut_batch()
            hidden = receiver[0].from_pp(received, forward_batch)
            state.prepared = []
            hidden = self.run_stages(
                receiver, kinds[cut:], hidden, forward_batch, state
            )
            state.receiving = state.log[start : state.prepared[0]]
            return forward_batch.residual_stream.export(hidden)

        world = dict(
            attn_tp=attn_tp,
            scattered=scattered,
            a2a=a2a,
            use_reduce_scatter=use_reduce_scatter,
        )
        uncut_results, uncut_states = self.run_world(uncut, **world)
        cut_results, cut_states = self.run_world(cut_at, **world)
        return SimpleNamespace(
            x=x,
            kinds=kinds,
            uncut=uncut_results,
            uncut_states=uncut_states,
            cut=cut_results,
            cut_states=cut_states,
        )

    def assert_values(self, run):
        """Every rank gets the output and residual it gets uncut, and their
        sum is the reference's residual."""
        want = cut_reference(run.x, run.kinds)
        for rank, ((hidden, residual), (cut_hidden, cut_residual)) in enumerate(
            zip(run.uncut, run.cut)
        ):
            with self.subTest(rank=rank):
                torch.testing.assert_close(cut_hidden, hidden, rtol=0, atol=0)
                self.assertIs(cut_residual is None, residual is None)
                if residual is not None:
                    torch.testing.assert_close(cut_residual, residual, rtol=0, atol=0)
                    hidden = hidden + residual
                torch.testing.assert_close(hidden, want, rtol=0, atol=0)

    def assert_sums_completed_once(self, run):
        """The sending rank completes what its output owes as it sends it,
        once; the receiving rank sums nothing of what it receives."""
        reductions = ("all_reduce", "reduce_scatter")
        for state in run.cut_states:
            with self.subTest(rank=state.rank, side="sending"):
                sent = [call for call in state.sending if call[0] in reductions]
                self.assertEqual(len(sent), 0 if state.sent is None else 1)
            with self.subTest(rank=state.rank, side="receiving"):
                received = [call for call in state.receiving if call[0] in reductions]
                self.assertEqual(received, [])

    def test_a_mixer_s_sum_left_to_the_next_ffn(self):
        run = self.run_cut(("mixer", "dense"), 1, attn_tp=2)
        # Uncut, the mixer's exit leaves its sum to the FFN's input; cut
        # between them, the sending rank still owes it.
        for state in run.uncut_states:
            self.assertEqual(state.owed[1], DeclaredSum(SumGroup.ATTN_TP))
        for state in run.cut_states:
            self.assertEqual(state.sent, DeclaredSum(SumGroup.ATTN_TP))
        self.assert_values(run)
        self.assert_sums_completed_once(run)

    def test_an_input_scattered_ffn_s_sum_left_to_a_reduce_scatter(self):
        kinds = ("attention", "dense", "attention", "dense")
        run = self.run_cut(kinds, 2, attn_tp=4, scattered=True)
        # The FFN leaves its TP sum to the next attention's reduce-scatter
        # onto this rank's slice of the rows.
        for state in run.uncut_states:
            self.assertEqual(state.owed[2], DeclaredSum(SumGroup.TP))
        for state in run.cut_states:
            self.assertEqual(state.sent, DeclaredSum(SumGroup.TP))
        self.assert_values(run)
        self.assert_sums_completed_once(run)

    def test_an_input_scattered_ffn_s_completed_sum(self):
        kinds = ("attention", "dense", "attention", "dense")
        run = self.run_cut(
            kinds, 2, attn_tp=4, scattered=True, use_reduce_scatter=False
        )
        # Without a reduce-scatter the FFN completes its own sum, though the
        # next attention's entry also binds for one left to it.
        for state in run.uncut_states:
            self.assertIsNone(state.owed[2])
        for state in run.cut_states:
            self.assertIsNone(state.sent)
        self.assert_values(run)
        self.assert_sums_completed_once(run)

    def test_an_ffn_on_its_own_rows_hands_on_the_stream_it_writes(self):
        kinds = ("attention", "sparse", "attention", "sparse")
        run = self.run_cut(kinds, 2, attn_tp=2, a2a=True)
        # The a2a MoE's output is complete; it writes it into the residual
        # and sends the stream alone.
        for state in run.cut_states:
            self.assertIsNone(state.sent)
            self.assertEqual(state.keys, ["hidden_states"])
        self.assert_values(run)
        self.assert_sums_completed_once(run)


class TestInputScatteredAttentionInput(CustomTestCase):
    """On an input-scattered batch an attention without a QKV hook, which
    projects the rows it is given, gets every row of its input."""

    def test_an_attention_without_a_qkv_hook_gets_every_row(self):
        parallel = fixture.parallel_of(
            attn_dp=1, attn_tp=2, enable_attn_tp_input_scattered=True
        )
        with fixture.planning(parallel):
            with layer_stack():
                attention, _ = layer()
        entry = attention.plan.paths[BatchVariant.INPUT_SCATTERED].entry
        rows = torch.ones(2, 4)
        with (
            patch.object(
                boundary_prepare,
                "get_attn_tp_context",
                return_value=SimpleNamespace(is_dsa=False),
            ),
            patch.object(
                boundary_prepare, "tp_gather", lambda h, fb: torch.cat([h, h])
            ),
        ):
            hidden = entry.attn_input_adapter(rows, SimpleNamespace(), None)
        self.assertEqual(hidden.shape[0], 4)


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
            writes_stream = False
            quantized_sum = False

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
                        for i in range(2)
                        for stage in append_stages(
                            (declare_attn(read=GatheringRead()), fixture.Norm()),
                            (
                                declare_ffn(
                                    sparse=True,
                                    next_layer_sparse=True,
                                    # The final read adds the last output as a
                                    # plain add.
                                    update=update if i == 0 else PLAIN_ADD,
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
