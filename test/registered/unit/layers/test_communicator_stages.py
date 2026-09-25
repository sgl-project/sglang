"""A hybrid stack's stages: the input of an FFN that is a stage of its own, and
the decision of a mixer (Mamba / attention) stage to leave its output all-reduce
to the next layer's input."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def recording_group(scale):
    return SimpleNamespace(all_reduce=MagicMock(side_effect=lambda x: x * scale))


class TestFfnStageInput(CustomTestCase):
    def setUp(self):
        self.attn_tp_group = recording_group(2)
        self.hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.residual = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        self.steps = MagicMock(side_effect=lambda h, r, fb, norm, ctx: (h, r))
        patcher = patch.object(
            comm,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_group=self.attn_tp_group),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_input(self, hidden_states, residual):
        return comm._ffn_stage_input(
            hidden_states, residual, "fb", "norm", "ctx", steps=self.steps
        )

    def test_a_mixer_partial_takes_the_standard_steps(self):
        owed = comm.UnreducedOutput(self.hidden, group=self.attn_tp_group)
        hidden, residual = self.run_input(owed, self.residual)
        self.assertIs(hidden, self.hidden)
        self.assertIs(residual, self.residual)
        self.attn_tp_group.all_reduce.assert_not_called()
        self.steps.assert_called_once()

    def test_a_complete_input_is_folded_into_the_residual(self):
        hidden, residual = self.run_input(self.hidden, self.residual)
        torch.testing.assert_close(hidden, torch.zeros_like(self.hidden))
        torch.testing.assert_close(residual, self.hidden + self.residual)

    def test_the_first_stage_takes_its_input_as_the_residual(self):
        hidden, residual = self.run_input(self.hidden, None)
        torch.testing.assert_close(hidden, torch.zeros_like(self.hidden))
        self.assertIs(residual, self.hidden)

    def test_a_sum_owed_over_another_group_is_completed_once_first(self):
        ffn_group = recording_group(4)
        owed = comm.UnreducedOutput(self.hidden, group=ffn_group)
        hidden, residual = self.run_input(owed, self.residual)
        ffn_group.all_reduce.assert_called_once()
        self.attn_tp_group.all_reduce.assert_not_called()
        torch.testing.assert_close(hidden, torch.zeros_like(self.hidden))
        torch.testing.assert_close(residual, self.hidden * 4 + self.residual)

    def test_a_reduction_that_redistributes_is_completed_once_first(self):
        redistribute = MagicMock(side_effect=lambda x: x[:1] * 3)
        owed = comm.UnreducedOutput(self.hidden, reduce_and_redistribute=redistribute)
        hidden, residual = self.run_input(owed, self.residual[:1])
        redistribute.assert_called_once()
        torch.testing.assert_close(residual, self.hidden[:1] * 3 + self.residual[:1])


class TestMixerExit(CustomTestCase):
    """The mixer skips its all-reduce before an FFN stage, when the fused kernel
    takes it, and otherwise only when the next input runs the same sum."""

    def communicator(self, *, next_ffn=False, fuse=False, last=False, tp=2, attn_tp=2):
        communicator = comm.LayerCommunicator.__new__(comm.LayerCommunicator)
        communicator.next_takes_attention_partial = next_ffn
        communicator.is_last_layer = last
        communicator._context = SimpleNamespace(tp_size=tp, attn_tp_size=attn_tp)
        communicator.should_fuse_mlp_allreduce_with_next_layer = MagicMock(
            return_value=fuse
        )
        return communicator

    def decide(
        self,
        communicator,
        *,
        dp=False,
        quant=False,
        lora=False,
        scattered=False,
    ):
        with (
            patch.object(comm, "is_dp_attention_enabled", return_value=dp),
            patch.object(
                comm,
                "get_exec",
                return_value=SimpleNamespace(
                    comm=SimpleNamespace(enable_quant_communications=quant)
                ),
            ),
            patch.object(
                comm, "get_lora", return_value=SimpleNamespace(enable_lora=lora)
            ),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=scattered),
            ),
        ):
            return communicator._mixer_sum_moves_to_next_layer(object())

    def test_before_an_ffn_stage_the_mixer_always_skips(self):
        for kwargs in ({}, {"dp": True}, {"quant": True}, {"lora": True}):
            with self.subTest(**kwargs):
                communicator = self.communicator(next_ffn=True, last=True, tp=1)
                self.assertTrue(self.decide(communicator, **kwargs))
                communicator.should_fuse_mlp_allreduce_with_next_layer.assert_not_called()

    def test_the_fused_kernel_takes_it(self):
        communicator = self.communicator(fuse=True, last=True)
        self.assertTrue(self.decide(communicator, dp=True))

    def test_the_next_input_runs_the_same_sum(self):
        self.assertTrue(self.decide(self.communicator()))

    def test_each_condition_of_the_unfused_skip(self):
        cases = {
            "last layer": (dict(last=True), {}),
            "no TP": (dict(tp=1, attn_tp=1), {}),
            "attention TP is not TP": (dict(tp=4, attn_tp=2), {}),
            "attention DP": ({}, dict(dp=True)),
            "quantized all-reduce": ({}, dict(quant=True)),
            "LoRA": ({}, dict(lora=True)),
            "input scattered": ({}, dict(scattered=True)),
        }
        for name, (built, env) in cases.items():
            with self.subTest(name):
                self.assertFalse(self.decide(self.communicator(**built), **env))

    def test_the_exit_publishes_the_skip_and_declares_the_group(self):
        group = object()
        for skips in (True, False):
            with self.subTest(skips=skips):
                communicator = self.communicator()
                communicator._mixer_sum_moves_to_next_layer = MagicMock(
                    return_value=skips
                )
                partial = torch.ones(2, 2)
                with patch.object(
                    comm,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_tp_group=group),
                ):
                    with communicator.mixer_exit(object()) as mixer_exit:
                        self.assertEqual(get_forward().fuse_mlp_allreduce, skips)
                        self.assertEqual(mixer_exit.skips_reduction, skips)
                    self.assertFalse(get_forward().fuse_mlp_allreduce)
                    out = mixer_exit.finish(partial)
                communicator._mixer_sum_moves_to_next_layer.assert_called_once()
                if skips:
                    self.assertIsInstance(out, comm.UnreducedOutput)
                    self.assertIs(out.partial, partial)
                    self.assertIs(out.group, group)
                else:
                    self.assertIs(out, partial)


class TestStackEnds(CustomTestCase):
    """Across a pipeline boundary a mixer before an FFN stage sends its attention
    partial sum, and that stage takes what it receives as that partial."""

    def communicator(self, **flags):
        communicator = comm.LayerCommunicator.__new__(comm.LayerCommunicator)
        communicator.next_takes_attention_partial = flags.get("next", False)
        communicator.previous_leaves_attention_partial = flags.get("previous", False)
        return communicator

    def test_a_mixer_before_an_ffn_stage_sends_its_partial(self):
        group = recording_group(2)
        partial, residual = torch.ones(2, 2), torch.zeros(2, 2)
        owed = comm.UnreducedOutput(partial, group=group)
        sent, sent_residual = self.communicator(next=True).finish_layer_stack(
            owed, residual, object()
        )
        self.assertIs(sent, partial)
        self.assertIs(sent_residual, residual)
        group.all_reduce.assert_not_called()

    def test_any_other_layer_completes_what_it_leaves(self):
        group = recording_group(2)
        owed = comm.UnreducedOutput(torch.ones(2, 2), group=group)
        out, _ = self.communicator().finish_layer_stack(owed, None, object())
        group.all_reduce.assert_called_once()
        torch.testing.assert_close(out, torch.full((2, 2), 2.0))

    def test_a_complete_output_passes_through(self):
        complete = torch.ones(2, 2)
        out, _ = self.communicator(next=True).finish_layer_stack(
            complete, None, object()
        )
        self.assertIs(out, complete)

    def test_the_ffn_stage_takes_what_it_receives_as_that_partial(self):
        group = object()
        received = torch.ones(2, 2)
        with patch.object(
            comm, "get_parallel", return_value=SimpleNamespace(attn_tp_group=group)
        ):
            taken = self.communicator(previous=True).start_layer_stack(received)
            passed = self.communicator().start_layer_stack(received)
        self.assertIsInstance(taken, comm.UnreducedOutput)
        self.assertIs(taken.partial, received)
        self.assertIs(taken.group, group)
        self.assertIs(passed, received)


if __name__ == "__main__":
    unittest.main()
