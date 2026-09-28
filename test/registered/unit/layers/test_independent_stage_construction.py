"""Independent stages use adjacent declarations without a decoder-layer plan."""

import itertools
import unittest
from types import SimpleNamespace

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.communicator import (
    EdgeDecl,
    Layout,
    StageDecl,
    StageInput,
    StageKind,
    StageOutput,
    declare_attn,
    declare_ffn,
    make_attn_stage,
    make_ffn_stage,
    make_stage,
    make_stages,
)
from sglang.srt.layers.communicator.residual import batch
from sglang.srt.layers.communicator.residual.add_norm import ADD
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIndependentStageConstruction(CustomTestCase):
    def make(self, kind, *, first=False, update=ADD, read=None):
        rows = Layout(frozenset())
        need = StageInput(rows) if read is None else StageInput(rows, read=read)
        produced = StageOutput(rows, update=update)
        return make_stage(
            StageDecl(need, produced),
            kind=kind,
            norm=fixture.Norm(),
            incoming=EdgeDecl(StageOutput(rows), need, rows, rows),
            outgoing=EdgeDecl(produced, StageInput(rows), rows, rows),
            enters_stack=first,
            terminal=True,
            fixed_output=kind is StageKind.ATTENTION,
        )

    def test_any_two_stage_kinds_share_the_same_construction(self):
        for first_kind in StageKind:
            for second_kind in StageKind:
                with (
                    self.subTest(first=first_kind, second=second_kind),
                    fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=1)),
                ):
                    first = self.make(first_kind, first=True)
                    second = self.make(second_kind)
                    self.assertIsNot(first.plan, second.plan)
                    self.assertFalse(hasattr(first.plan, "layer_facts"))
                    fb = SimpleNamespace()
                    batch.start(fb)
                    hidden = torch.ones(2, 4)
                    for stage in (first, second):
                        hidden = stage.prepare(hidden, fb)
                        if stage.kind is StageKind.ATTENTION:
                            hidden = stage.finish(hidden, fb)
                        else:
                            with stage.exit(fb) as output:
                                pass
                            hidden = output.finish(hidden)
                    torch.testing.assert_close(
                        batch.snapshot(hidden, fb), torch.full((2, 4), 9.0)
                    )

    def test_binding_order_does_not_mutate_the_other_stage(self):
        with fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=1)):
            attn = declare_attn(terminal=True)
            ffn = declare_ffn(previous=attn, terminal=True)
            for reverse in (False, True):
                a = lambda: make_attn_stage(
                    declaration=attn, norm=fixture.Norm(), following=ffn
                )
                f = lambda: make_ffn_stage(declaration=ffn, norm=fixture.Norm())
                first, second = (f(), a()) if reverse else (a(), f())
                self.assertIsNot(first.plan, second.plan)
                self.assertIsNot(first.norm, second.norm)
                self.assertFalse(hasattr(first.plan, "layer_facts"))
                self.assertFalse(hasattr(first.plan, "previous"))
            with self.assertRaises(TypeError):
                declare_ffn(previous=first)
            with self.assertRaisesRegex(ValueError, "following"):
                make_attn_stage(declaration=attn, norm=None, following=declare_ffn())

    def test_public_factories_connect_all_stage_pairs(self):
        for first_kind in StageKind:
            for second_kind in StageKind:
                with (
                    self.subTest(first=first_kind, second=second_kind),
                    fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=1)),
                ):
                    declare = {
                        StageKind.ATTENTION: declare_attn,
                        StageKind.FFN: declare_ffn,
                    }
                    build = {
                        StageKind.ATTENTION: make_attn_stage,
                        StageKind.FFN: make_ffn_stage,
                    }
                    first = declare[first_kind]()
                    second = declare[second_kind](previous=first)
                    stages = [
                        build[first_kind](
                            declaration=first, norm=fixture.Norm(), following=second
                        ),
                        build[second_kind](declaration=second, norm=fixture.Norm()),
                    ]
                    fb = SimpleNamespace()
                    batch.start(fb)
                    hidden = torch.ones(2, 4)
                    for stage in stages:
                        hidden = stage.prepare(hidden, fb)
                        if stage.kind is StageKind.ATTENTION:
                            hidden = stage.finish(hidden, fb)
                        else:
                            with stage.exit(fb) as output:
                                pass
                            hidden = output.finish(hidden)
                    torch.testing.assert_close(
                        batch.snapshot(hidden, fb), torch.full((2, 4), 9.0)
                    )

    def test_prepared_branch_reuses_the_same_placements(self):
        from sglang.srt.layers.communicator.factories import (
            _connect,
            _connections,
            _fork_input,
        )

        for dp in (1, 2):
            for tp in (1, 2):
                for local_dense in (False, True):
                    parallel = fixture.parallel_of(
                        attn_dp=dp,
                        attn_tp=tp,
                        moe_dense_tp_size=1 if local_dense else None,
                    )
                    with (
                        self.subTest(dp=dp, tp=tp, local_dense=local_dense),
                        fixture.planning(parallel),
                    ):
                        attn = declare_attn(previous=declare_ffn(sparse=True))
                        moe = declare_ffn(previous=attn, sparse=True)
                        dense = declare_ffn(
                            prepared_from=moe,
                            ordinary_only=True,
                        )
                        branch_attn = declare_attn(previous=dense, ordinary_only=True)
                        last_dense = declare_ffn(
                            previous=branch_attn,
                            ordinary_only=True,
                        )
                        _, into_moe = _connections(attn, moe)
                        fork = _fork_input(into_moe, dense)
                        middle = _connect(dense, branch_attn)
                        last = _connect(branch_attn, last_dense, residual_from=middle)
                        for declaration, following, expected in (
                            (dense, branch_attn, fork),
                            (branch_attn, last_dense, middle),
                            (last_dense, None, last),
                        ):
                            make = (
                                make_attn_stage
                                if declaration.kind is StageKind.ATTENTION
                                else make_ffn_stage
                            )
                            stage = make(
                                declaration=declaration,
                                norm=fixture.Norm(),
                                following=following,
                            )
                            self.assertEqual(
                                {
                                    v: edge.incoming
                                    for v, edge in stage.plan.variants.items()
                                },
                                expected.entries,
                            )
                            self.assertFalse(stage.plan.is_first_layer)
                            if declaration is dense:
                                with self.assertRaisesRegex(
                                    RuntimeError, "branch_input"
                                ):
                                    stage.plan._steps.ffn.prepare(None, None, None)

        with self.assertRaisesRegex(ValueError, "choose"):
            declare_ffn(previous=attn, prepared_from=moe)

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
                    boundaries = make_stages(
                        *((d, fixture.Norm()) for d in declarations), terminal=True
                    )
                    self.assertEqual(len(boundaries), count)
                    fb = SimpleNamespace()
                    batch.start(fb)
                    hidden = torch.ones(2, 4)
                    # The direct factories provide an independent construction
                    # reference, including repeated kinds and terminal placement.
                    from dataclasses import replace

                    previous = None
                    linked = []
                    for i, d in enumerate(declarations):
                        previous = replace(
                            d, previous=previous, terminal=i == count - 1
                        )
                        linked.append(previous)
                    for i, boundary in enumerate(boundaries):
                        build = (
                            make_attn_stage
                            if kinds[i] is StageKind.ATTENTION
                            else make_ffn_stage
                        )
                        reference = build(
                            declaration=linked[i],
                            norm=boundary.norm,
                            following=linked[i + 1] if i + 1 < count else None,
                        )
                        self.assertEqual(
                            boundary.plan.variants, reference.plan.variants
                        )
                        self.assertEqual(boundary.plan.is_last_layer, i == count - 1)
                        self.assertIsNone(declarations[i].previous)
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
                _, source = make_stages(
                    (declare_attn(), fixture.Norm()),
                    (declare_ffn(sparse=True), fixture.Norm()),
                    previous=declare_ffn(sparse=True),
                )
                before = dict(source.plan.variants)
                first, attn, last = make_stages(
                    (
                        declare_ffn(ordinary_only=True),
                        fixture.Norm(),
                    ),
                    (declare_attn(ordinary_only=True), fixture.Norm()),
                    (
                        declare_ffn(ordinary_only=True),
                        fixture.Norm(),
                    ),
                    prepared_from=source.declaration,
                )
                self.assertEqual(source.plan.variants, before)
                self.assertIs(first.declaration.prepared_from, source.declaration)
                self.assertIs(attn.declaration.previous, first.declaration)
                self.assertIs(last.declaration.previous, attn.declaration)
                for v, edge in first.plan.variants.items():
                    source_input = source.plan.variants[v].incoming
                    self.assertEqual(
                        edge.incoming.produced.layout, source_input.need.layout
                    )
                    self.assertEqual(edge.incoming.residual, source_input.residual_to)
                with self.assertRaisesRegex(RuntimeError, "branch_input"):
                    first.plan._steps.ffn.prepare(None, None, None)

    def test_mixer_successor_comes_from_the_local_sequence(self):
        for following in (declare_attn(mixer_exit=True), declare_ffn()):
            with (
                self.subTest(kind=following.kind),
                fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2)),
            ):
                mixer, consumer = make_stages(
                    (declare_attn(mixer_exit=True), fixture.Norm()),
                    (following, fixture.Norm()),
                    terminal=True,
                )
                self.assertIs(mixer.declaration.next_kind, following.kind)
                for variant, edge in mixer.plan.variants.items():
                    output = edge.outgoing.produced
                    self.assertEqual(
                        output.always_leaves, following.kind is StageKind.FFN
                    )
                    self.assertEqual(
                        output.leaves_for_next_layer,
                        following.kind is StageKind.ATTENTION,
                    )
                    incoming = consumer.plan.variants[variant].incoming.produced
                    self.assertEqual(incoming.group, output.group)

    def test_sequence_requires_one_source_and_external_terminal(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            make_stages()
        with self.assertRaisesRegex(ValueError, "choose"):
            make_stages(
                (declare_attn(), None),
                previous=declare_ffn(),
                prepared_from=declare_ffn(),
            )
        with self.assertRaisesRegex(ValueError, "input sources"):
            make_stages((declare_attn(previous=declare_ffn()), None))
        with self.assertRaisesRegex(ValueError, "terminal"):
            make_stages((declare_ffn(terminal=True), None))

    def test_mismatched_declared_edge_is_rejected(self):
        rows = Layout(frozenset())
        declaration = StageDecl(StageInput(rows), StageOutput(rows))
        wrong_input = StageInput(rows, read=object())
        with self.assertRaisesRegex(ValueError, "disagrees"):
            make_stage(
                declaration,
                kind=StageKind.FFN,
                norm=None,
                incoming=EdgeDecl(StageOutput(rows), wrong_input, rows, rows),
                outgoing=EdgeDecl(declaration.output, declaration.input, rows, rows),
            )


if __name__ == "__main__":
    unittest.main()
