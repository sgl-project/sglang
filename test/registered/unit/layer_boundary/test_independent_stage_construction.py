"""Independent stages use adjacent declarations without a decoder-layer plan."""

import itertools
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.layer_boundary import (
    ProducerReduction,
    StageKind,
    declare_attn,
    declare_ffn,
    make_attn_stage,
    make_ffn_stage,
    make_stages,
)
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.residual import batch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIndependentStageConstruction(CustomTestCase):
    def test_active_variant_without_a_bound_path_never_falls_back(self):
        from sglang.srt.layers.layer_boundary import construction
        from sglang.test.boundary_fixtures import stub_plan

        for variant in BatchVariant:
            plan = stub_plan()
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
                    boundaries = make_stages(
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

    def test_sequence_and_independent_factories_agree_on_terminal_placement(self):
        for dp, tp, terminal in itertools.product((1, 2), (1, 2), (False, True)):
            for kinds in itertools.product(StageKind, repeat=3):
                with (
                    self.subTest(dp=dp, tp=tp, terminal=terminal, kinds=kinds),
                    fixture.planning(fixture.parallel_of(attn_dp=dp, attn_tp=tp)),
                ):
                    declarations = [
                        declare_attn() if kind is StageKind.ATTENTION else declare_ffn()
                        for kind in kinds
                    ]
                    sequence = make_stages(
                        *(
                            (declaration, fixture.Norm())
                            for declaration in declarations
                        ),
                        terminal=terminal,
                    )
                    linked = []
                    for index, declaration in enumerate(declarations):
                        linked.append(
                            replace(
                                declaration,
                                previous=linked[-1] if linked else None,
                                terminal=terminal and index == len(declarations) - 1,
                            )
                        )
                    for index, (declaration, combined) in enumerate(
                        zip(linked, sequence)
                    ):
                        make = (
                            make_attn_stage
                            if declaration.kind is StageKind.ATTENTION
                            else make_ffn_stage
                        )
                        independent = make(
                            declaration=declaration,
                            norm=fixture.Norm(),
                            following=linked[index + 1]
                            if index + 1 < len(linked)
                            else None,
                        )
                        self.assertEqual(combined.plan.terminal, declaration.terminal)
                        self.assertEqual(
                            independent.plan.terminal, declaration.terminal
                        )
                        self.assertEqual(combined.plan.edges, independent.plan.edges)
                        for variant, steps in combined.plan.paths.items():
                            self.assertEqual(
                                steps.output, independent.plan.paths[variant].output
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
                before = dict(source.plan.edges)
                first, attn, last = make_stages(
                    (
                        declare_ffn(),
                        fixture.Norm(),
                    ),
                    (declare_attn(), fixture.Norm()),
                    (
                        declare_ffn(),
                        fixture.Norm(),
                    ),
                    prepared_from=source.declaration,
                )
                self.assertEqual(source.plan.edges, before)
                self.assertIs(first.declaration.prepared_from, source.declaration)
                self.assertIs(attn.declaration.previous, first.declaration)
                self.assertIs(last.declaration.previous, attn.declaration)
                for v, edge in first.plan.edges.items():
                    source_input = source.plan.edges[v].incoming
                    self.assertEqual(
                        edge.incoming.produced.layout, source_input.need.layout
                    )
                    self.assertEqual(edge.incoming.residual, source_input.residual_to)

    def test_a_replicated_tail_stays_in_compute_without_a_fusion_object(self):
        for local_tail in (False, True):
            with fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2)):
                boundary = make_ffn_stage(
                    declaration=declare_ffn(
                        previous=declare_attn(),
                        sparse=True,
                        reduction=ProducerReduction.TAIL_AFTER_SUM
                        if local_tail
                        else ProducerReduction.EXIT_SCOPED,
                    ),
                    norm=fixture.Norm(),
                )
                self.assertIsNone(boundary.plan.fusions)
                produced = boundary.plan.paths[BatchVariant.ORDINARY].output
                self.assertEqual(produced.may_defer_to_next, not local_tail)
                # A local compute tail does not change the existing RS alternative.
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
                mixer, consumer = make_stages(
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
                    self.assertEqual(incoming.group, output.group)


if __name__ == "__main__":
    unittest.main()
