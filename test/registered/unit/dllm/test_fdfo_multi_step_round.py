"""Unit tests for the multi-step FDFO round loop in ``DllmAlgorithm._run_fdfo``.

``SGLANG_DLLM_FDFO_STEPS_PER_ROUND=N`` keeps denoising the same frozen batch
for N forwards before handing control back to the scheduler. The contract that
makes that safe is that a round of N inner steps has to leave exactly the state
N single-step rounds would have left -- the only thing amortized is the
scheduler's per-round host work, not any of the denoise math. Two things can
break that silently:

* the batched-state path gathers the per-request states once per round instead
  of once per step, so a step that reads or writes state through the wrong
  object diverges only after the second inner step;
* algorithms with no batched path fall back to a per-step ``step`` loop (this
  is what non-NPU dLLM runs today), which is a second, independent
  implementation of the same round.

Both paths are covered here, plus the early exit that stops a round once every
row has resolved.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.algorithm.joint_threshold import JointThreshold
from sglang.srt.dllm.config import DllmConfig

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BATCH, BLOCK, VOCAB = 3, 8, 32
MASK_ID = VOCAB - 1


class _ScriptedModelRunner:
    """Replays a fixed logits sequence, one entry per forward."""

    def __init__(self, logits_seq):
        self.logits_seq = logits_seq
        self.calls = 0

    def forward(self, forward_batch, pp_proxy_tensors=None):
        logits = self.logits_seq[self.calls]
        self.calls += 1
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=logits),
            can_run_graph=False,
        )


def _forward_batch(input_ids: torch.Tensor) -> SimpleNamespace:
    # _run_fdfo touches only these three members of ForwardBatch.
    return SimpleNamespace(
        batch_size=BATCH,
        input_ids=input_ids,
        mark_forward_metadata_ready=lambda *a, **k: None,
    )


def _logits_sequence(n: int):
    torch.manual_seed(1234)
    # Unscaled logits keep every confidence below the 0.5 unmask threshold, so
    # each step resolves exactly one position per row (the forced one). A block
    # of BLOCK masks therefore needs BLOCK steps and stays unresolved across a
    # shorter round -- which is the case the state carry-over has to get right.
    return [torch.randn(BATCH * BLOCK, VOCAB) for _ in range(n)]


def _algorithm(steps_per_round: int, *, vectorized: bool, max_post_edit_steps: int):
    config = DllmConfig(
        algorithm="JointThreshold",
        algorithm_config={
            "threshold": 0.5,
            # p is a probability, so an edit threshold above 1 disables T2T and
            # leaves a pure mask-to-token trajectory.
            "edit_threshold": 2.0,
            "max_post_edit_steps": max_post_edit_steps,
            "vectorized_decoding": vectorized,
        },
        block_size=BLOCK,
        mask_id=MASK_ID,
        max_running_requests=BATCH,
        first_done_first_out_mode=True,
    )
    # fdfo_steps_per_round is read once, in the constructor.
    with envs.SGLANG_DLLM_FDFO_STEPS_PER_ROUND.override(steps_per_round):
        return JointThreshold(config)


def _drive(steps_per_round, total_forwards, *, vectorized, max_post_edit_steps=16):
    algorithm = _algorithm(
        steps_per_round,
        vectorized=vectorized,
        max_post_edit_steps=max_post_edit_steps,
    )
    input_ids = torch.full((BATCH * BLOCK,), MASK_ID, dtype=torch.int64)
    forward_batch = _forward_batch(input_ids)
    model_runner = _ScriptedModelRunner(_logits_sequence(total_forwards))

    states, accept_lengths = None, None
    for _ in range(total_forwards // steps_per_round):
        _, _, accept_lengths, states, _ = algorithm.run(
            model_runner, forward_batch, states
        )
    return input_ids, accept_lengths, model_runner.calls


# The round loop is device-agnostic; pin the portable denoise math so the
# result does not depend on whether the host happens to offer a fused kernel.
# create=True because that hook is introduced by a sibling PR (#33161) and this
# test must hold both before and after it lands; when absent there is no fused
# path to pin and the patch is inert.
@patch("sglang.srt.dllm.algorithm.base._argmax_softmax_prob_fused", None, create=True)
class TestFdfoMultiStepRound(CustomTestCase):
    def test_batched_path_multi_step_round_matches_single_step_rounds(self):
        # Four inner steps in one round vs four rounds of one step: same
        # forwards, same tokens, same accept lengths.
        one = _drive(1, 4, vectorized=True)
        four = _drive(4, 4, vectorized=True)
        self.assertEqual(one[2], 4)
        self.assertEqual(four[2], 4)
        self.assertTrue(torch.equal(one[0], four[0]))
        self.assertEqual(one[1], four[1])
        # The window must actually exercise unresolved carry-over: if every
        # row had finished, the comparison above would be vacuous.
        self.assertEqual(four[1], [0] * BATCH)
        self.assertTrue((four[0] == MASK_ID).any())

    def test_fallback_path_multi_step_round_matches_single_step_rounds(self):
        # Same contract for algorithms with no batched path (fdfo_batched_begin
        # returns None): this is the loop non-NPU dLLM runs.
        one = _drive(1, 4, vectorized=False)
        four = _drive(4, 4, vectorized=False)
        self.assertEqual(four[2], 4)
        self.assertTrue(torch.equal(one[0], four[0]))
        self.assertEqual(one[1], four[1])
        self.assertEqual(four[1], [0] * BATCH)

    def test_round_stops_once_every_row_has_resolved(self):
        # A block with no masked position and no post-edit budget resolves on
        # the first step; the remaining inner forwards must not be issued.
        for vectorized in (True, False):
            with self.subTest(vectorized=vectorized):
                algorithm = _algorithm(4, vectorized=vectorized, max_post_edit_steps=0)
                input_ids = torch.zeros(BATCH * BLOCK, dtype=torch.int64)
                model_runner = _ScriptedModelRunner(_logits_sequence(4))
                _, _, accept_lengths, states, _ = algorithm.run(
                    model_runner, _forward_batch(input_ids), None
                )
                self.assertEqual(model_runner.calls, 1)
                self.assertEqual(accept_lengths, [BLOCK] * BATCH)
                self.assertEqual(states, [None] * BATCH)


if __name__ == "__main__":
    unittest.main()
