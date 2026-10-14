"""Independent stages use adjacent declarations without a decoder-layer plan."""

import itertools
import unittest
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.layer_boundary import (
    ProducerReduction,
    StageKind,
    append_stages,
    declare_attn,
    declare_ffn,
    factories,
    layer_stack,
)
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.facts import facts_of
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual import batch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.boundary_fixtures import build_stages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def next_layer():
    """The next layer's attention, which a stack that does not end the
    model hands its output to."""
    return (declare_attn(),)


class TestIndependentStageConstruction(CustomTestCase):
    def test_active_variant_without_a_bound_path_never_falls_back(self):
        from sglang.srt.layers.layer_boundary import construction
        from sglang.test.boundary_fixtures import stub_plan

        for variant in BatchVariant:
            plan = stub_plan()
            plan._unpadded_attn_tp_size = 2
            ordinary = object()
            plan.paths[BatchVariant.ORDINARY] = ordinary
            with (
                self.subTest(variant=variant),
                patch.object(
                    construction,
                    "get_forward",
                    return_value=SimpleNamespace(
                        sp_active=variant is BatchVariant.SEQUENCE_PARALLEL
                    ),
                ),
                patch.object(
                    construction,
                    "get_attn_tp_context",
                    return_value=SimpleNamespace(
                        input_scattered=variant is BatchVariant.INPUT_SCATTERED
                    ),
                ),
                patch.object(
                    construction,
                    "_batch_shards_over_cp",
                    return_value=variant is BatchVariant.CONTEXT_PARALLEL,
                ),
                patch.object(
                    construction,
                    "_rows_indivisible_over_attn_tp",
                    return_value=variant is BatchVariant.UNPADDED,
                ),
            ):
                if variant is BatchVariant.ORDINARY:
                    self.assertIs(plan.path_for(None), ordinary)
                else:
                    with self.assertRaisesRegex(NotImplementedError, variant.name):
                        plan.path_for(None)
                    selected = object()
                    plan.paths[variant] = selected
                    self.assertIs(plan.path_for(None), selected)

    def test_sequences_of_one_to_four_stages(self):
        for count in range(1, 5):
            for kinds in itertools.product(StageKind, repeat=count):
                with (
                    self.subTest(kinds=kinds),
                    fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=1)),
                ):
                    declarations = [
                        declare_attn() if kind is StageKind.ATTENTION else declare_ffn()
                        for kind in kinds
                    ]
                    boundaries = build_stages(
                        *((d, fixture.Norm()) for d in declarations), terminal=True
                    )
                    self.assertEqual(len(boundaries), count)
                    fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)
                    batch.start(fb)
                    hidden = torch.ones(2, 4)

                    for boundary in boundaries:
                        hidden = boundary.prepare(hidden, fb)
                        if boundary.kind is StageKind.ATTENTION:
                            hidden = boundary.finish(hidden, fb)
                        else:
                            with boundary.exit(fb) as output:
                                pass
                            hidden = output.finish(hidden)
                    # Norm doubles its input, each later stage adds the prior
                    # residual. Snapshot adds the final contribution.
                    torch.testing.assert_close(
                        batch.snapshot(hidden, fb), torch.full((2, 4), float(3**count))
                    )

    def test_grouping_stages_into_appends_binds_the_same_plans(self):
        # A stage binds once its producer and consumer are both known, so one
        # append per stage binds what one append of the whole sequence does.
        for dp, tp, terminal in itertools.product((1, 2), (1, 2), (False, True)):
            for kinds in itertools.product(StageKind, repeat=3):
                with (
                    self.subTest(dp=dp, tp=tp, terminal=terminal, kinds=kinds),
                    fixture.planning(fixture.parallel_of(attn_dp=dp, attn_tp=tp)),
                ):

                    def stages():
                        return [
                            (
                                (
                                    declare_attn()
                                    if kind is StageKind.ATTENTION
                                    else declare_ffn()
                                ),
                                fixture.Norm(),
                            )
                            for kind in kinds
                        ]

                    together = build_stages(*stages(), terminal=terminal)
                    with layer_stack(next_layers=[] if terminal else [next_layer]):
                        apart = [b for stage in stages() for b in append_stages(stage)]
                    for index, (combined, single) in enumerate(zip(together, apart)):
                        ends = terminal and index == len(kinds) - 1
                        self.assertEqual(combined.plan.terminal, ends)
                        self.assertEqual(single.plan.terminal, ends)
                        self.assertEqual(combined.plan.edges, single.plan.edges)
                        for variant, steps in combined.plan.paths.items():
                            self.assertEqual(
                                steps.output, single.plan.paths[variant].output
                            )

    def test_sequence_branch_keeps_source_lineage(self):
        for dp, tp, local_dense in itertools.product((1, 2), (1, 2), (False, True)):
            with (
                self.subTest(dp=dp, tp=tp, local_dense=local_dense),
                fixture.planning(
                    fixture.parallel_of(
                        attn_dp=dp,
                        attn_tp=tp,
                        moe_dense_tp_size=1 if local_dense else None,
                    )
                ),
            ):

                def source_layer():
                    return append_stages(
                        (declare_attn(), fixture.Norm()),
                        (declare_ffn(sparse=True), fixture.Norm()),
                    )[1]

                def previous_layer():
                    return (declare_ffn(sparse=True),)

                neighbours = dict(
                    previous_layers=[previous_layer], next_layers=[next_layer]
                )
                with layer_stack(**neighbours):
                    alone = source_layer()
                with layer_stack(**neighbours):
                    source = source_layer()
                    first, attn, last = append_stages(
                        (declare_ffn(), fixture.Norm()),
                        (declare_attn(), fixture.Norm()),
                        (declare_ffn(), fixture.Norm()),
                        prepared_from=source.declaration,
                    )
                # The branch leaves the stage it reads from as it was.
                self.assertEqual(source.plan.edges, alone.plan.edges)
                # Each records the stage before it by what binding reads of it.
                self.assertEqual(
                    first.declaration.prepared_from, facts_of(source.declaration)
                )
                self.assertEqual(attn.declaration.previous, facts_of(first.declaration))
                self.assertEqual(last.declaration.previous, facts_of(attn.declaration))
                for v, edge in first.plan.edges.items():
                    source_input = source.plan.edges[v].incoming
                    self.assertEqual(
                        edge.incoming.produced.layout, source_input.need.layout
                    )
                    self.assertEqual(edge.incoming.residual, source_input.residual_to)

    def test_an_ffn_without_a_fusion_object_may_defer_or_reduce_scatter(self):
        with fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2)):
            _, boundary = build_stages(
                (declare_attn(), fixture.Norm()),
                (declare_ffn(sparse=True), fixture.Norm()),
            )
        self.assertIsNone(boundary.plan.fusions)
        produced = boundary.plan.paths[BatchVariant.ORDINARY].output
        self.assertTrue(produced.may_defer_to_next)
        self.assertTrue(produced.may_reduce_scatter)

    def test_mixer_successor_comes_from_the_local_sequence(self):
        for following in (
            declare_attn(
                reduction=ProducerReduction.EXIT_SCOPED, gathers_attn_tp_input=False
            ),
            declare_ffn(),
        ):
            with (
                self.subTest(kind=following.kind),
                fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2)),
            ):
                mixer, consumer = build_stages(
                    (
                        declare_attn(
                            reduction=ProducerReduction.EXIT_SCOPED,
                            gathers_attn_tp_input=False,
                        ),
                        fixture.Norm(),
                    ),
                    (following, fixture.Norm()),
                    terminal=True,
                )
                for variant, edge in mixer.plan.edges.items():
                    output = edge.outgoing.produced
                    self.assertEqual(
                        output.always_partial, following.kind is StageKind.FFN
                    )
                    self.assertEqual(
                        output.may_defer_to_next,
                        following.kind is StageKind.ATTENTION,
                    )
                    incoming = consumer.plan.edges[variant].incoming.produced
                    self.assertEqual(
                        incoming.group,
                        output.group if output.always_partial else None,
                    )


# The kinds of stage a stack is cut between.
CUT_STAGES = {
    # Always leaves its sum to the next stage's input, and has no exit.
    "attention": declare_attn,
    # A mixer, whose exit completes its sum or leaves it to an FFN's input.
    "mixer": partial(
        declare_attn,
        reduction=ProducerReduction.EXIT_SCOPED,
        gathers_attn_tp_input=False,
    ),
    "dense": declare_ffn,
    "sparse": partial(declare_ffn, sparse=True),
}

# (parallel state, a2a MoE backend)
CUT_CONFIGS = {
    "attention TP2": (fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=2), False),
    "attention TP2, a2a MoE": (
        fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=2),
        True,
    ),
    "attention DP2 x TP2, dense FFN fully DP": (
        fixture.parallel_of(attn_dp=2, attn_tp=2, moe_dense_tp_size=1, pp_size=2),
        False,
    ),
    "attention TP4, input-scattered": (
        fixture.parallel_of(
            attn_dp=1, attn_tp=4, enable_attn_tp_input_scattered=True, pp_size=2
        ),
        False,
    ),
}


def cut_stages(kinds):
    return [(CUT_STAGES[kind](), fixture.Norm()) for kind in kinds]


def held_by_another_rank(kinds):
    """The layers another pipeline rank holds, declaring ``kinds``: the
    neighbours of a stack cut next to them."""
    return [lambda: tuple(CUT_STAGES[kind]() for kind in kinds)] if kinds else []


def bind_rank(kinds, *, before=(), after=()):
    """Bind ``kinds`` as one pipeline rank's stack, with ``before`` and
    ``after`` on the ranks next to it. Returns the stages, or the error that
    rejects them."""
    try:
        with layer_stack(
            previous_layers=held_by_another_rank(before),
            next_layers=held_by_another_rank(after),
        ):
            return append_stages(*cut_stages(kinds)), None
    except (NotImplementedError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def described(value):
    """``value`` as data that compares by value. A plan holds partials, which
    compare by identity: each is described as its function's name and its
    arguments, described in turn."""
    if isinstance(value, partial):
        return (described(value.func), described(value.args), described(value.keywords))
    if isinstance(value, msgspec.Struct):
        return (
            type(value).__name__,
            *(described(getattr(value, name)) for name in value.__struct_fields__),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__,
            *(described(getattr(value, field.name)) for field in fields(value)),
        )
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((repr(key), described(item)) for key, item in value.items()),
                key=lambda pair: pair[0],
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(described(item) for item in value)
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def binding(stage):
    """What binding gave ``stage``: its declaration, without the stages it is
    chained to, and its plan's edges and paths."""
    plan = stage.plan
    return described(
        (
            replace(stage.declaration, previous=None, prepared_from=None),
            plan.terminal,
            plan.enters_stack,
            plan.edges,
            plan.paths,
        )
    )


def attention_rows(variant):
    return factories._row_layouts(variant)[1]


class TestPipelineCuts(CustomTestCase):
    """A stack cut between two pipeline ranks: each rank binds its stages with
    the stage next to them on the other rank as their neighbour. Each rank's
    stages agree on the rows the residual is on, the handoff carries it on the
    attention's rows, and the stages away from the cut bind as they do uncut.

    The receiving stage may still bind for a sum its producer leaves, next to
    the path for one already completed: which one runs is decided per batch
    (see test_append_stages.TestValuesAcrossAPipelineCut)."""

    def test_both_ranks_bind_the_rows_the_handoff_carries(self):
        for (name, (parallel, a2a)), reduction in itertools.product(
            CUT_CONFIGS.items(), ("ar", "rs")
        ):
            bound = rejected_cuts = 0
            with fixture.planning(parallel, a2a=a2a, boundary_reduction=reduction):
                for count in range(2, 5):
                    for kinds in itertools.product(CUT_STAGES, repeat=count):
                        uncut, rejected = bind_rank(kinds)
                        for cut in range(1, count):
                            # Input-scattered attention runs only in models
                            # whose layers end with an FFN, so there a rank's
                            # stages end on one: after an attention the
                            # residual stays on this rank's attention-TP slice,
                            # which is not what a handoff carries.
                            if parallel.enable_attn_tp_input_scattered and (
                                kinds[cut - 1] not in ("dense", "sparse")
                            ):
                                continue
                            with self.subTest(
                                config=name,
                                boundary_reduction=reduction,
                                sender=kinds[:cut],
                                receiver=kinds[cut:],
                            ):
                                if self.check_cut(kinds, cut, uncut, rejected):
                                    bound += 1
                                else:
                                    rejected_cuts += 1
            # Most cuts bind: what they are checked for is not vacuous.
            self.assertGreater(bound, rejected_cuts, (name, reduction))

    def check_cut(self, kinds, cut, uncut, rejected):
        """Check one cut; returns whether both of its ranks bind."""
        sender, sender_error = bind_rank(kinds[:cut], after=kinds[cut:])
        receiver, receiver_error = bind_rank(kinds[cut:], before=kinds[:cut])
        if sender_error or receiver_error:
            # A cut rejects only what the stack rejects uncut, and alike.
            for error in (sender_error, receiver_error):
                if error is not None:
                    self.assertEqual(error, rejected)
            return False
        self.assert_rows_follow(sender, "sender")
        self.assert_rows_follow(receiver, "receiver")
        last = sender[-1]
        for variant, edges in last.plan.edges.items():
            # An attention that always leaves its sum has no exit: the
            # residual stays on the rows it ran with.
            handed_on = (
                edges.incoming.residual_to
                if last.plan.finishes_directly
                else edges.outgoing.residual_to
            )
            self.assertEqual(
                handed_on, attention_rows(variant), f"sender's last, {variant.name}"
            )
        for variant, edges in receiver[0].plan.edges.items():
            self.assertEqual(
                edges.incoming.residual,
                attention_rows(variant),
                f"receiver's first, {variant.name}",
            )
        if rejected is not None:
            # An FFN that hands off returns its output on the attention's
            # rows, so a cut may bind a stack that is rejected uncut, where
            # the next stage cannot read the FFN's own rows.
            return True
        # Up to the sender's last FFN, which returns the residual for the
        # handoff, and after the receiver's first, which reads it, the stages
        # bind as they do uncut.
        ffns = [i for i, stage in enumerate(sender) if stage.kind is StageKind.FFN]
        for index in range(ffns[-1] if ffns else 0):
            self.assertEqual(
                binding(sender[index]), binding(uncut[index]), f"sender's {index}"
            )
        ffns = [i for i, stage in enumerate(receiver) if stage.kind is StageKind.FFN]
        for index in range(ffns[0] + 1 if ffns else len(receiver), len(receiver)):
            self.assertEqual(
                binding(receiver[index]),
                binding(uncut[cut + index]),
                f"receiver's {index}",
            )
        return True

    def test_a_rank_after_an_attention_that_transforms_its_output_is_rejected(self):
        # The attention leaves its sum, and its transform runs at the next
        # stage's input: on the receiving rank, which does not hold it.
        transformed = partial(declare_attn, output_transform=OutputTransform(abs))
        for (name, (parallel, a2a)), kind in itertools.product(
            CUT_CONFIGS.items(), CUT_STAGES
        ):
            with (
                self.subTest(config=name, receiver=kind),
                fixture.planning(parallel, a2a=a2a),
            ):
                # Without the transform, the same rank binds.
                self.assertIsNone(bind_rank((kind,), before=("attention",))[1])
                with (
                    self.assertRaisesRegex(
                        NotImplementedError,
                        "a pipeline rank that ends on an attention transforming "
                        "the output whose sum it leaves",
                    ),
                    layer_stack(previous_layers=[lambda: (transformed(),)]),
                ):
                    append_stages(*cut_stages((kind,)))

    def assert_rows_follow(self, stages, rank):
        """Within one rank's stages, each side of a boundary takes the
        residual on the rows the other side leaves it on."""
        for index, stage in enumerate(stages):
            for variant, edges in stage.plan.edges.items():
                # The exit starts from where the entry left the residual.
                self.assertEqual(
                    edges.incoming.residual_to,
                    edges.outgoing.residual,
                    f"{rank}'s {index}, {variant.name}",
                )
        for index, (producer, consumer) in enumerate(zip(stages, stages[1:])):
            for variant, edges in producer.plan.edges.items():
                incoming = consumer.plan.edges[variant].incoming
                where = f"{rank}'s {index} to {index + 1}, {variant.name}"
                if producer.plan.finishes_directly:
                    # No exit: the attention and its consumer bind one edge.
                    self.assertEqual(
                        (edges.outgoing.residual, edges.outgoing.residual_to),
                        (incoming.residual, incoming.residual_to),
                        where,
                    )
                else:
                    self.assertEqual(
                        edges.outgoing.residual_to, incoming.residual, where
                    )


if __name__ == "__main__":
    unittest.main()
