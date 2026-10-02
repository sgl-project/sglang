import itertools
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.layer_boundary import (
    Layout,
    MixerExit,
    OutputContract,
    SumGroup,
    TokenAxis,
    UnreducedOutput,
)
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
from sglang.srt.models import nemotron_h_utils as utils
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def layer_stage(pattern, index):
    from sglang.srt.layers.layer_boundary.construction import BatchVariant
    from sglang.srt.layers.layer_boundary.factories import _connections

    previous = utils._declaration(pattern, index - 1) if index else None
    declaration = replace(
        utils._declaration(pattern, index),
        previous=previous,
        terminal=index == len(pattern) - 1,
    )
    following = (
        replace(utils._declaration(pattern, index + 1), previous=declaration)
        if index + 1 < len(pattern)
        else None
    )
    incoming, outgoing = _connections(declaration, following)
    return SimpleNamespace(
        kind=declaration.kind,
        edges=(
            incoming.entries[BatchVariant.ORDINARY],
            outgoing.exits[BatchVariant.ORDINARY],
        ),
        enters_stack=index == 0,
    )


def sizes(*, dp=1, tp=1):
    return {
        TokenAxis.ATTN_DP: dp,
        TokenAxis.ATTN_CP: 1,
        TokenAxis.ATTN_TP: tp,
    }


def stages(pattern, *, dp=1, tp=1, a2a=False):
    from sglang.test.communicator_patch import patch_communicator

    parallel = SimpleNamespace(
        attn_dp_size=dp,
        attn_tp_size=tp,
        attn_cp_size=1,
        tp_size=dp * tp,
        moe_dp_size=1,
        moe_dense_tp_size=None,
        enable_attn_tp_input_scattered=False,
        enable_prefill_cp=False,
    )
    with (
        patch(
            "sglang.srt.layers.layernorm_sp.layernorm_sp_enabled", return_value=False
        ),
        patch_communicator(
            "get_exec",
            return_value=SimpleNamespace(
                comm=SimpleNamespace(boundary_reduction="rs+rsv"),
                overlap=SimpleNamespace(enable_two_batch_overlap=False),
            ),
        ),
        patch_communicator("token_axis_sizes", return_value=sizes(dp=dp, tp=tp)),
        patch_communicator("get_parallel", return_value=parallel),
        patch.object(utils, "get_parallel", return_value=parallel),
        patch_communicator("is_moe_input_scattered_across_dp_ranks", return_value=a2a),
        patch_communicator("is_dense_ffn_fully_dp", return_value=False),
        patch_communicator("_prefill_cp_shards_tokens", return_value=False),
    ):
        return [layer_stage(pattern, i) for i in range(len(pattern))]


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
                    self.assertEqual(out_of.need.layout, rows)
                    self.assertEqual(out_of.residual_to, rows)
                    self.assertEqual(into.residual, rows)
                    self.assertEqual(into.produced.layout, rows)
                    produced = out_of.produced
                    may_leave = produced.always_partial or produced.may_defer_to_next
                    self.assertEqual(
                        into.produced.group, produced.group if may_leave else None
                    )
                    self.assertEqual(
                        into.produced.always_partial, produced.always_partial
                    )
                    self.assertEqual(
                        into.produced.may_defer_to_next,
                        produced.may_defer_to_next,
                    )
                self.assertEqual(
                    layers[0].edges[0].produced, OutputContract(rows, update=None)
                )
                self.assertTrue(layers[0].enters_stack)
                self.assertFalse(any(layer.enters_stack for layer in layers[1:]))
                last = layers[-1].edges[1].produced
                self.assertFalse(last.always_partial or last.may_defer_to_next)

    def test_what_each_kind_of_boundary_carries(self):
        # (pattern, boundary after layer 0): group, always_partial, may_defer_to_next
        cases = {
            "M-": (SumGroup.ATTN_TP, True, False),
            "*E": (SumGroup.ATTN_TP, True, False),
            "MM": (SumGroup.ATTN_TP, False, True),
            "-M": (SumGroup.TP, False, True),
            "EM": (SumGroup.MOE_OUTPUT, False, True),
            "--": (SumGroup.TP, False, True),
            "E-": (SumGroup.MOE_OUTPUT, False, True),
        }
        for pattern, expected in cases.items():
            with self.subTest(pattern=pattern):
                into = stages(pattern, tp=2)[1].edges[0].produced
                self.assertEqual(
                    (into.group, into.always_partial, into.may_defer_to_next),
                    expected,
                )
        # Without attention TP a mixer's output is complete.
        into = stages("M-", tp=1)[1].edges[0].produced
        self.assertEqual(into, OutputContract(into.layout, update=None))
        # A MoE on this rank's own rows hands on a complete output; an a2a
        # backend dispatches only the MoE, so an MLP still sums over TP.
        into = stages("EM", tp=2, a2a=True)[1].edges[0].produced
        self.assertEqual(into, OutputContract(into.layout, update=None))
        into = stages("-M", tp=2, a2a=True)[1].edges[0].produced
        self.assertEqual(
            (into.group, into.always_partial, into.may_defer_to_next),
            (SumGroup.TP, False, True),
        )


class TestMixerExit(CustomTestCase):
    """A mixer skips its output all-reduce when its output always leaves the sum
    (to an FFN stage), and when it may leave it and the fused kernel takes it;
    what it hands on says which."""

    def test_decision_table(self):
        tp_group = object()
        attention = Layout.sharded_over(TokenAxis.ATTN_DP, axis_sizes=sizes(tp=2))
        for always, may, movable in itertools.product((False, True), repeat=3):
            if always and may:
                continue
            with self.subTest(
                always_partial=always, may_defer_to_next=may, movable=movable
            ):
                produced = OutputContract(
                    attention,
                    group=SumGroup.ATTN_TP if always or may else None,
                    always_partial=always,
                    may_defer_to_next=may,
                )
                communicator = SimpleNamespace(
                    plan=SimpleNamespace(
                        path_for=lambda batch: SimpleNamespace(output=produced),
                    ),
                    _sum_deferral_allowed=MagicMock(return_value=movable),
                )
                hidden = torch.ones(2, 4)
                with get_parallel().override(tp_group=tp_group, tp_size=2):
                    with MixerExit(
                        communicator, None, stream=ResidualStream()
                    ) as mixer_exit:
                        skipped = should_skip_mlp_all_reduce()
                    output = mixer_exit.finish(hidden)
                    output, _ = mixer_exit._stream.input(output)
                self.assertFalse(should_skip_mlp_all_reduce())
                hands_on = may and movable
                self.assertEqual(mixer_exit.skips_reduction, always or hands_on)
                self.assertEqual(skipped, always or hands_on)
                if hands_on:
                    self.assertIsInstance(output, UnreducedOutput)
                    self.assertIs(output.group, tp_group)
                else:
                    self.assertIs(output, hidden)


if __name__ == "__main__":
    unittest.main()
