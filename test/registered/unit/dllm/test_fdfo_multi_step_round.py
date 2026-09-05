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


def _peaked_logits(token_id: int):
    """A near-one-hot row: the peak's probability is ~1, so it clears both the
    unmask threshold and any edit threshold. Scripting the winning token per
    forward makes the T2T trajectory exact instead of sampled."""
    logits = torch.full((BATCH * BLOCK, VOCAB), -10.0)
    logits[:, token_id] = 10.0
    return logits


def _algorithm(
    steps_per_round: int,
    *,
    vectorized: bool,
    max_post_edit_steps: int,
    edit_threshold: float = 2.0,
):
    config = DllmConfig(
        algorithm="JointThreshold",
        algorithm_config={
            "threshold": 0.5,
            # p is a probability, so the default edit threshold above 1 disables
            # T2T and leaves a pure mask-to-token trajectory. The post-edit test
            # lowers it to drive edits instead.
            "edit_threshold": edit_threshold,
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


def _drive(
    steps_per_round,
    total_forwards,
    *,
    vectorized,
    max_post_edit_steps=16,
    edit_threshold=2.0,
    logits_seq=None,
):
    algorithm = _algorithm(
        steps_per_round,
        vectorized=vectorized,
        max_post_edit_steps=max_post_edit_steps,
        edit_threshold=edit_threshold,
    )
    input_ids = torch.full((BATCH * BLOCK,), MASK_ID, dtype=torch.int64)
    forward_batch = _forward_batch(input_ids)
    model_runner = _ScriptedModelRunner(
        logits_seq if logits_seq is not None else _logits_sequence(total_forwards)
    )

    states, accept_lengths = None, None
    for _ in range(total_forwards // steps_per_round):
        _, _, accept_lengths, states, _ = algorithm.run(
            model_runner, forward_batch, states
        )
    return input_ids, accept_lengths, model_runner.calls


# The round loop is device-agnostic; pin the portable denoise math so the
# result does not depend on whether the host happens to offer a fused kernel.
# create=True: the fused hook is optional and may not exist on this build.
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

    def test_t2t_edit_budget_carries_and_exhausts_across_rounds(self):
        # The other cases disable T2T, which also pins post_edit_steps at 0:
        # a block that still holds masks never takes the no_mask_active branch.
        # This one drives the state that actually crosses a round boundary.
        #
        # Scripted trajectory (peak token per forward, edit threshold 0):
        #   1: every position unmasks to token 0   (has_mask -> False)
        #   2: post_edit_steps 0->1, T2T edits to token 1
        #   3: post_edit_steps 1->2, T2T edits to token 2
        #   4: post_edit_steps 2->3 > budget 2 -> row finishes, no edit
        #
        # Token 2 is the whole assertion: if the budget were re-initialised per
        # round instead of carried, K=2 would still have headroom at forward 4
        # and would edit on to token 3.
        seq = [_peaked_logits(i) for i in range(4)]
        for vectorized in (True, False):
            with self.subTest(vectorized=vectorized):
                runs = [
                    _drive(
                        k,
                        4,
                        vectorized=vectorized,
                        max_post_edit_steps=2,
                        edit_threshold=0.0,
                        logits_seq=seq,
                    )
                    for k in (1, 2, 4)
                ]
                for ids, accept, calls in runs:
                    self.assertEqual(calls, 4)
                    self.assertEqual(accept, [BLOCK] * BATCH)
                    self.assertTrue(
                        torch.equal(ids, torch.full_like(ids, 2)), ids.tolist()
                    )

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
