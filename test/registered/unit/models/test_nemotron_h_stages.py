import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.boundary_layout import (
    Layout,
    StageDecl,
    StageInput,
    StageOutput,
    SumGroup,
    TokenAxis,
    stage_edges,
)
from sglang.srt.layers.communicator import MixerExit, UnreducedOutput
from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
from sglang.srt.models import nemotron_h_utils as utils
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def sizes(*, dp=1, tp=1):
    return {
        TokenAxis.ATTN_DP: dp,
        TokenAxis.ATTN_CP: 1,
        TokenAxis.ATTN_TP_SCATTER: tp,
    }


def stages(pattern, *, dp=1, tp=1, a2a=False):
    backend = SimpleNamespace(is_none=lambda: not a2a)
    with (
        patch.object(utils, "token_axis_sizes", return_value=sizes(dp=dp, tp=tp)),
        patch.object(utils, "get_moe_a2a_backend", return_value=backend),
    ):
        return [utils.layer_stage(pattern, i) for i in range(len(pattern))]


class TestStageEdges(CustomTestCase):
    """Each layer's two boundaries come from its own declaration and the
    previous stage's; a boundary two layers share is seen the same way by both."""

    PATTERNS = ("M-M*E", "MEMEM*E", "*E", "-*", "*--", "--", "-M", "E*E-", "MM*")

    def test_adjacent_layers_agree_on_their_shared_boundary(self):
        for pattern, dp, tp, a2a in itertools.product(
            self.PATTERNS, (1, 2), (1, 2), (False, True)
        ):
            with self.subTest(pattern=pattern, dp=dp, tp=tp, a2a=a2a):
                layers = stages(pattern, dp=dp, tp=tp, a2a=a2a)
                rows = Layout.sharded_over(
                    TokenAxis.ATTN_DP, axis_sizes=sizes(dp=dp, tp=tp)
                )
                for before, after in zip(layers, layers[1:]):
                    out_of, into = before.edges[1], after.edges[0]
                    # Both ends of the boundary are on the rows every layer
                    # hands on, residual included.
                    self.assertEqual(out_of.need.layout, rows)
                    self.assertEqual(out_of.residual_to, rows)
                    self.assertEqual(into.residual, rows)
                    self.assertEqual(into.produced.layout, rows)
                    # What the value may carry is what the producer may leave.
                    produced = out_of.produced
                    may_leave = produced.always_leaves or produced.leaves_for_next_layer
                    self.assertEqual(
                        into.produced.group, produced.group if may_leave else None
                    )
                    self.assertEqual(
                        into.produced.always_leaves, produced.always_leaves
                    )
                    self.assertEqual(
                        into.produced.leaves_for_next_layer,
                        produced.leaves_for_next_layer,
                    )
                # The layer stack starts with a complete value; the last layer
                # leaves nothing to a next one.
                self.assertEqual(layers[0].edges[0].produced, StageOutput(rows))
                self.assertTrue(layers[0].enters_stack)
                self.assertFalse(any(layer.enters_stack for layer in layers[1:]))
                last = layers[-1].edges[1].produced
                self.assertFalse(last.always_leaves or last.leaves_for_next_layer)

    def test_what_each_kind_of_boundary_carries(self):
        # (pattern, boundary after layer 0): group, always_leaves, leaves_for_next_layer
        cases = {
            "M-": (SumGroup.ATTN_TP, True, False),
            "*E": (SumGroup.ATTN_TP, True, False),
            "MM": (SumGroup.ATTN_TP, False, True),
            "-M": (SumGroup.TP, False, True),
            "EM": (SumGroup.MOE_OUTPUT, False, True),
            "--": (None, False, False),
            "E-": (None, False, False),
        }
        for pattern, expected in cases.items():
            with self.subTest(pattern=pattern):
                into = stages(pattern, tp=2)[1].edges[0].produced
                self.assertEqual(
                    (into.group, into.always_leaves, into.leaves_for_next_layer),
                    expected,
                )
        # Without attention TP a mixer's output is complete.
        into = stages("M-", tp=1)[1].edges[0].produced
        self.assertEqual(into, StageOutput(into.layout))
        # A MoE on this rank's own rows hands on a complete output; an a2a
        # backend dispatches only the MoE, so an MLP still sums over TP.
        into = stages("EM", tp=2, a2a=True)[1].edges[0].produced
        self.assertEqual(into, StageOutput(into.layout))
        into = stages("-M", tp=2, a2a=True)[1].edges[0].produced
        self.assertEqual(
            (into.group, into.always_leaves, into.leaves_for_next_layer),
            (SumGroup.TP, False, True),
        )

    def test_the_residual_follows_the_input_onto_a_finer_slice(self):
        axis_sizes = sizes(dp=2, tp=2)
        attention = Layout.sharded_over(TokenAxis.ATTN_DP, axis_sizes=axis_sizes)
        local = Layout.sharded_over(
            TokenAxis.ATTN_DP, TokenAxis.ATTN_TP_SCATTER, axis_sizes=axis_sizes
        )
        full = Layout.sharded_over(axis_sizes=axis_sizes)
        for need, during in ((local, local), (full, attention)):
            stage = StageDecl(StageInput(need), StageOutput(need))
            into, out_of = stage_edges(previous=None, stage=stage, rows=attention)
            self.assertEqual((into.residual, into.residual_to), (attention, during))
            self.assertEqual((out_of.residual, out_of.residual_to), (during, attention))


class TestMixerExit(CustomTestCase):
    """A mixer skips its output all-reduce when its output always leaves the sum
    (to an FFN stage), and when it may leave it and the fused kernel takes it;
    what it hands on says which."""

    def test_decision_table(self):
        tp_group = object()
        attention = Layout.sharded_over(TokenAxis.ATTN_DP, axis_sizes=sizes(tp=2))
        for always, may, fuses in itertools.product((False, True), repeat=3):
            if always and may:
                continue
            with self.subTest(
                always_leaves=always, leaves_for_next_layer=may, fuses=fuses
            ):
                produced = StageOutput(
                    attention,
                    group=SumGroup.ATTN_TP if always or may else None,
                    always_leaves=always,
                    leaves_for_next_layer=may,
                )
                communicator = SimpleNamespace(
                    _batch_steps=lambda batch: SimpleNamespace(ffn_output=produced),
                    should_fuse_mlp_allreduce_with_next_layer=MagicMock(
                        return_value=fuses
                    ),
                )
                hidden = torch.ones(2, 4)
                with get_parallel().override(tp_group=tp_group):
                    with MixerExit(communicator, None) as mixer_exit:
                        skipped = should_skip_mlp_all_reduce()
                    output = mixer_exit.finish(hidden)
                self.assertFalse(should_skip_mlp_all_reduce())
                hands_on = may and fuses
                self.assertEqual(mixer_exit.skips_reduction, always or hands_on)
                self.assertEqual(skipped, always or hands_on)
                if hands_on:
                    self.assertIsInstance(output, UnreducedOutput)
                    self.assertIs(output.group, tp_group)
                else:
                    self.assertIs(output, hidden)


if __name__ == "__main__":
    unittest.main()
