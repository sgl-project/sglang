import types
import unittest
from functools import partial
from unittest.mock import MagicMock

import torch

from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers.layer_boundary import (
    Layout,
    OutputContract,
    SumGroup,
    UnreducedOutput,
    complete_owed,
)
from sglang.srt.layers.layer_boundary import ops as transport_ops
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.ops import keep_output
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.runtime_context import get_forward
from sglang.test.boundary_fixtures import (
    finish_exit,
    identity_input,
    sp_region_steps,
    stub_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_group(scale=3):
    """The group a deferred sum is owed over, with a stubbed all-reduce."""
    return types.SimpleNamespace(all_reduce=MagicMock(side_effect=lambda h: h * scale))


def ordinary_steps(output, *, returns_over_dp=False):
    """A layer's ordinary steps, carrying the FFN output declaration the exit
    reads and whether the output goes back over attention DP."""
    return comm.StagePath(
        entry=comm.EntryPath(
            prepare=partial(
                comm_ops._run_entry,
                step=partial(
                    comm_ops._update_read,
                    pre_move=None,
                    enters_stack=False,
                    read=comm.NORM_QUANT_READOUT,
                    update=comm.PLAIN_ADD,
                ),
                carried_fusions=(),
            ),
            input_rows=comm.Layout(frozenset()),
            input_move=identity_input,
            attn_input_adapter=comm_ops._attn_input_default,
        ),
        output=output,
        returns_over_dp=returns_over_dp,
        output_move=None if returns_over_dp else keep_output,
    )


def make_communicator(
    *,
    fuse,
    reduce_scatter,
    reduce_scatter_step=None,
    scatters_to_local_tokens=False,
    group=None,
    fusions=None,
    leaves_sum=True,
):
    """A communicator whose decisions and postprocess are stubbed, built
    without the process-wide parallel state."""
    communicator = stub_plan()
    communicator.fusions = fusions
    communicator.paths[BatchVariant.ORDINARY] = ordinary_steps(
        OutputContract(
            Layout(frozenset()),
            group=SumGroup.TP,
            may_defer_to_next=leaves_sum,
        ),
        returns_over_dp=scatters_to_local_tokens,
    )
    if fusions is None:
        communicator.output._defers_sum = MagicMock(return_value=fuse)
    else:
        communicator.output._defers_sum = MagicMock(return_value=False)
    communicator.output._sum_in_reduce_scatter = MagicMock(return_value=reduce_scatter)
    communicator.terminal = False
    communicator.paths[BatchVariant.SEQUENCE_PARALLEL] = None
    communicator.paths[BatchVariant.INPUT_SCATTERED] = None
    communicator.paths[BatchVariant.CONTEXT_PARALLEL] = None
    communicator.output._dp_reduce_scatter_step = MagicMock(
        return_value=reduce_scatter_step
    )
    communicator.output.ffn_reduction_group = MagicMock(
        return_value=group or make_group()
    )
    communicator.output._complete_now = MagicMock(
        side_effect=lambda hidden_states, residual, **_: (hidden_states + 1, residual)
    )
    return communicator


def published_flags():
    """The skip flag the FFN sees; the boundary completes the sum, so the FFN
    never leaves it out."""
    return (get_forward().mlp_reduce_scatter,)


class TestFfnExit(CustomTestCase):
    def setUp(self):
        patches = patch_communicator("_batch_shards_over_cp", return_value=False)
        patches.__enter__()
        self.addCleanup(patches.__exit__, None, None, None)
        self.hidden_states = torch.ones(3, 4)
        self.residual = torch.zeros(3, 4)
        self.forward_batch = object()

    def run_exit(self, communicator):
        with communicator.output.ffn_exit(
            self.forward_batch, stream=ResidualStream()
        ) as ffn_exit:
            seen = published_flags()
            hidden_states = self.hidden_states * 2
        return seen, finish_exit(ffn_exit, hidden_states, self.residual)

    def test_reduction_left_to_next_layer_declares_its_group(self):
        group = make_group()
        communicator = make_communicator(fuse=True, reduce_scatter=False, group=group)
        _, (hidden_states, _) = self.run_exit(communicator)
        self.assertIs(hidden_states.group, group)
        self.assertIsNone(hidden_states.reduce_to_dp_local)

    def test_a_reduce_scatter_is_left_to_the_next_layer(self):
        """Under attention DP the reduce-scatter postprocess would run goes to the
        next layer with the partial sum."""
        step = MagicMock()
        communicator = make_communicator(
            fuse=False,
            reduce_scatter=True,
            reduce_scatter_step=step,
            scatters_to_local_tokens=True,
        )
        seen, (hidden_states, residual) = self.run_exit(communicator)
        self.assertEqual(seen, (False,))
        self.assertIsInstance(hidden_states, UnreducedOutput)
        bound = hidden_states.reduce_to_dp_local
        self.assertIs(bound.func, transport_ops.to_dp_local)
        self.assertEqual(bound.args, (step, self.forward_batch))
        self.assertIs(residual, self.residual)
        communicator.output._complete_now.assert_not_called()
        step.assert_not_called()

    def test_a_deferred_sum_carries_the_scatter_back_under_attention_dp(self):
        group = make_group()
        communicator = make_communicator(
            fuse=True, reduce_scatter=False, scatters_to_local_tokens=True, group=group
        )
        _, (hidden_states, _) = self.run_exit(communicator)
        self.assertIsInstance(hidden_states, UnreducedOutput)
        bound = hidden_states.reduce_to_dp_local
        self.assertIs(bound.func, transport_ops.all_reduce_to_dp_local)
        self.assertEqual(bound.args, (group, self.forward_batch))
        communicator.output._complete_now.assert_not_called()
        group.all_reduce.assert_not_called()

    def test_postprocess_completes_other_exits(self):
        for reduce_scatter in (False, True):
            with self.subTest(reduce_scatter=reduce_scatter):
                communicator = make_communicator(
                    fuse=False, reduce_scatter=reduce_scatter
                )
                seen, (hidden_states, residual) = self.run_exit(communicator)
                self.assertEqual(seen, (False,))
                self.assertNotIsInstance(hidden_states, UnreducedOutput)
                communicator.output._complete_now.assert_called_once()
                # A reduce-scatter on the way back completes the sum; otherwise
                # the exit completes it.
                self.assertEqual(
                    communicator.output._complete_now.call_args.kwargs["owes_sum"],
                    not reduce_scatter,
                )
                torch.testing.assert_close(hidden_states, self.hidden_states * 2 + 1)
                self.assertIs(residual, self.residual)

    def test_a_layer_that_declares_no_deferral_runs_postprocess(self):
        communicator = make_communicator(
            fuse=True, reduce_scatter=True, leaves_sum=False
        )
        seen, (hidden_states, residual) = self.run_exit(communicator)

        self.assertEqual(seen, (False,))
        communicator.output._defers_sum.assert_not_called()
        communicator.output._complete_now.assert_called_once()
        torch.testing.assert_close(hidden_states, self.hidden_states * 2 + 1)

    def test_finish_applies_the_decision_the_ffn_saw(self):
        """finish() applies the decision published to the FFN, even when the
        decision methods would answer differently by then."""
        for fuse in (True, False):
            with self.subTest(fuse=fuse):
                communicator = make_communicator(fuse=fuse, reduce_scatter=False)
                with communicator.output.ffn_exit(
                    self.forward_batch, stream=ResidualStream()
                ) as ffn_exit:
                    seen = published_flags()
                    decide = communicator.output._defers_sum
                    decide.return_value = not fuse
                    hidden_states = self.hidden_states * 2
                hidden_states, _ = finish_exit(ffn_exit, hidden_states, self.residual)
                self.assertEqual(seen, (False,))
                self.assertEqual(isinstance(hidden_states, UnreducedOutput), fuse)
                self.assertEqual(communicator.output._complete_now.called, not fuse)

    def test_exit_selects_its_batch_path_once(self):
        communicator = make_communicator(fuse=False, reduce_scatter=False)
        selected = communicator.paths.get(BatchVariant.ORDINARY)
        lookup = MagicMock(return_value=selected)
        communicator.path_for = lookup
        self.run_exit(communicator)
        lookup.assert_called_once_with(self.forward_batch)
        self.assertIs(
            communicator.output._complete_now.call_args.kwargs["steps"],
            selected,
        )

    def test_flags_are_restored_after_the_ffn(self):
        before = published_flags()
        for fuse, scatter in ((True, False), (False, True)):
            self.run_exit(make_communicator(fuse=fuse, reduce_scatter=scatter))
            self.assertEqual(published_flags(), before)
        self.assertEqual(published_flags(), before)

    def test_deferral_implies_fusion_and_passes_the_handoff_through(self):
        """A deferring communicator publishes the finalize choice, and a
        non-tensor handoff leaves finish() untouched for the next layer's input
        norm."""

        fusions = types.SimpleNamespace(can_defer_finalize=lambda layer, fb: True)
        communicator = make_communicator(
            fuse=False, reduce_scatter=False, fusions=fusions
        )
        handoff = object()
        with communicator.output.ffn_exit(
            self.forward_batch, stream=ResidualStream()
        ) as ffn_exit:
            seen = published_flags() + (get_forward().defer_moe_finalize,)
        self.assertEqual(seen, (False, True))
        self.assertEqual(
            finish_exit(ffn_exit, handoff, self.residual), (handoff, self.residual)
        )
        communicator.output._complete_now.assert_not_called()
        self.assertFalse(get_forward().defer_moe_finalize)

        # The MoE may decline per forward; a tensor then takes the fused exit.
        with communicator.output.ffn_exit(
            self.forward_batch, stream=ResidualStream()
        ) as fallback_exit:
            hidden_states, _ = finish_exit(
                fallback_exit, self.hidden_states * 2, self.residual
            )
        self.assertIsInstance(hidden_states, UnreducedOutput)

    def test_compiles_without_graph_breaks(self):
        """The exit and the output it leaves trace under fullgraph torch.compile."""
        group = types.SimpleNamespace(all_reduce=lambda h: h * 3)
        for fuse in (False, True):
            with self.subTest(fuse=fuse):
                communicator = stub_plan()
                communicator.terminal = False
                communicator.paths[BatchVariant.ORDINARY] = ordinary_steps(
                    OutputContract(
                        Layout(frozenset()),
                        group=SumGroup.TP,
                        may_defer_to_next=True,
                    )
                )
                output = communicator.output
                output._defers_sum = lambda fb, steps, **_: fuse
                output._skips_sum_for_reduce_scatter = lambda steps, dp: False
                output.ffn_reduction_group = lambda steps: group
                output._complete_now = lambda h, r, **_: (h + 1, r)

                def layer(hidden_states, residual):
                    stream = ResidualStream(residual)
                    with output.ffn_exit(None, stream=stream) as ffn_exit:
                        if get_forward().fuse_mlp_allreduce:
                            hidden_states = hidden_states * 2
                    return stream.complete(ffn_exit.finish(hidden_states))

                torch._dynamo.reset()
                compiled = torch.compile(layer, backend="eager", fullgraph=True)
                with patch_communicator("_batch_shards_over_cp", lambda fb: False):
                    hidden_states = compiled(self.hidden_states, self.residual)
                expected = self.hidden_states * 6 if fuse else self.hidden_states + 1
                torch.testing.assert_close(hidden_states, expected)


class TestReduceOutput(CustomTestCase):
    def setUp(self):
        patches = patch_communicator("_batch_shards_over_cp", return_value=False)
        patches.__enter__()
        self.addCleanup(patches.__exit__, None, None, None)
        self.group = make_group()
        self.all_reduce = self.group.all_reduce

    def communicator(self, *, fuse):
        return make_communicator(fuse=fuse, reduce_scatter=False, group=self.group)

    def test_reduction_left_by_the_last_layer_runs_once(self):
        communicator = self.communicator(fuse=True)
        with communicator.output.ffn_exit(
            object(), stream=ResidualStream()
        ) as ffn_exit:
            hidden_states = torch.ones(3, 4)
        hidden_states, _ = finish_exit(ffn_exit, hidden_states, torch.zeros(3, 4))

        hidden_states = complete_owed(hidden_states)
        self.all_reduce.assert_called_once()
        torch.testing.assert_close(hidden_states, torch.full((3, 4), 3.0))
        self.assertNotIsInstance(hidden_states, UnreducedOutput)

        complete_owed(hidden_states)
        self.all_reduce.assert_called_once()

    def test_a_reduce_scatter_left_by_the_last_layer_runs_once(self):
        local = torch.full((1, 4), 7.0)
        step = MagicMock(return_value=local)
        partial = torch.ones(3, 4)
        hidden_states = complete_owed(UnreducedOutput(partial, reduce_to_dp_local=step))
        step.assert_called_once_with(partial)
        self.assertIs(hidden_states, local)
        self.all_reduce.assert_not_called()

    def test_finish_layer_stack_completes_the_last_layer(self):
        for fuse in (True, False):
            with self.subTest(fuse=fuse):
                self.all_reduce.reset_mock()
                communicator = self.communicator(fuse=fuse)
                residual = torch.zeros(3, 4)
                with communicator.output.ffn_exit(
                    object(), stream=ResidualStream()
                ) as ffn_exit:
                    hidden_states = torch.ones(3, 4)
                hidden_states, _ = finish_exit(ffn_exit, hidden_states, residual)

                hidden_states = complete_owed(hidden_states)
                self.assertEqual(self.all_reduce.call_count, int(fuse))
                expected = 3.0 if fuse else 2.0  # all-reduce stub / postprocess stub
                torch.testing.assert_close(hidden_states, torch.full((3, 4), expected))


class TestSelectFfnCompletion(CustomTestCase):
    """What the next layer's input runs in place of postprocess, chosen before
    the FFN runs."""

    def setUp(self):
        patches = patch_communicator("_batch_shards_over_cp", return_value=False)
        patches.__enter__()
        self.addCleanup(patches.__exit__, None, None, None)

    def communicator(
        self, *, fuse=False, is_last_layer=False, scatters=True, sp_region=False
    ):
        communicator = stub_plan()
        communicator.terminal = is_last_layer
        communicator.paths[BatchVariant.SEQUENCE_PARALLEL] = (
            sp_region_steps() if sp_region else None
        )
        communicator.paths[BatchVariant.INPUT_SCATTERED] = None
        communicator.paths[BatchVariant.CONTEXT_PARALLEL] = None
        communicator.paths[BatchVariant.ORDINARY] = ordinary_steps(
            OutputContract(
                Layout(frozenset()),
                group=SumGroup.MOE_OUTPUT,
                may_defer_to_next=True,
                may_reduce_scatter=True,
                may_reduce_scatterv=True,
            ),
            returns_over_dp=scatters,
        )
        communicator.output._defers_sum = lambda forward_batch, steps, **_: fuse
        communicator.output._sum_in_reduce_scatter = lambda forward_batch, dp_step: (
            not fuse
        )
        communicator.output._complete_now = lambda h, r, **_: ("now", r)
        self.group = make_group()
        communicator.output.ffn_reduction_group = lambda forward_batch: self.group
        return communicator

    def left(self, communicator, step, forward_batch=None):
        with patch_communicator("_select_dp_reduce_scatter", return_value=step):
            completion = communicator.output._decide(
                forward_batch, communicator.output.plan.path_for(forward_batch)
            )
        hidden_states, _ = completion.complete(torch.ones(3, 4), None)
        return None if hidden_states == "now" else hidden_states

    def test_a_reduce_scatter_is_bound_for_the_next_layer(self):
        forward_batch = object()
        for step in (
            transport_ops.dp_reduce_scatterv,
            transport_ops.dp_reduce_scatter,
        ):
            with self.subTest(step=step.__name__):
                left = self.left(self.communicator(), step, forward_batch)
                bound = left.reduce_to_dp_local
                self.assertIs(bound.func, transport_ops.to_dp_local)
                self.assertEqual(bound.args, (step, forward_batch))

    def test_postprocess_keeps_everything_else(self):
        reduce_scatter = transport_ops.dp_reduce_scatterv
        for name, communicator, step in (
            ("scatter only", self.communicator(), None),
            ("last layer", self.communicator(is_last_layer=True), reduce_scatter),
            ("other postprocess", self.communicator(scatters=False), reduce_scatter),
        ):
            with self.subTest(name):
                self.assertIsNone(self.left(communicator, step))

    def test_postprocess_keeps_an_active_layernorm_sp_region(self):
        communicator = self.communicator(sp_region=True)
        with get_forward().scoped(sp_active=True):
            self.assertIsNone(self.left(communicator, transport_ops.dp_reduce_scatterv))

    def test_a_deferred_sum_keeps_its_layout_or_scatters_back(self):
        forward_batch = object()
        kept = self.left(self.communicator(fuse=True, scatters=False), None)
        self.assertIs(kept.group, self.group)
        self.assertIsNone(kept.reduce_to_dp_local)
        moved = self.left(self.communicator(fuse=True), None, forward_batch)
        self.assertIsNone(moved.group)
        bound = moved.reduce_to_dp_local
        self.assertIs(bound.func, transport_ops.all_reduce_to_dp_local)
        self.assertEqual(bound.args, (self.group, forward_batch))

    def test_the_all_reduce_runs_before_the_scatter(self):
        calls = []
        partial = torch.ones(3, 4)
        local = torch.empty(1, 4)
        group = types.SimpleNamespace(
            all_reduce=lambda x: calls.append(("all_reduce", x)) or x * 2
        )
        with (
            patch_communicator("_dp_scatter_group", return_value="group"),
            patch_communicator(
                "get_local_dp_buffer",
                side_effect=lambda group: calls.append(("buffer", group)) or local,
            ),
            patch_communicator(
                "dp_scatter",
                side_effect=lambda out, full, fb: calls.append(("scatter", out)),
            ),
        ):
            result = transport_ops.all_reduce_to_dp_local(group, object(), partial)
        self.assertEqual([c[0] for c in calls], ["all_reduce", "buffer", "scatter"])
        self.assertIs(calls[0][1], partial)
        self.assertIs(result, local)


if __name__ == "__main__":
    unittest.main()
