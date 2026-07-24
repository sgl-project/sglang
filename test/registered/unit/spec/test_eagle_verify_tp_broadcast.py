"""Regression test for EAGLE verify TP-consistency (issue #31071).

The greedy verify branch computed accepted tokens from a per-rank local
``torch.argmax`` and did *not* broadcast the result across TP ranks, while the
sampling branch did. When per-rank logits differ (FP non-determinism from a
non-deterministic all-reduce, e.g. AMD ``--enable-aiter-allreduce-fusion``) a
near-tie makes argmax pick a different token per rank, ranks accept a different
number of drafts, committed seq_lens/batch shapes diverge, and the next TP
collective deadlocks.

The fix hoists the rank-0 broadcast to a single spot after the accept decision
is finalized so it covers the greedy path, the sampling path, and SIMULATE. This
test drives the greedy path (forced via the HIP backend flag, as in production)
with world_size>1 and asserts both the computed verify decision and that it is
broadcast from rank 0 — through the plain TP group and, when DP-attention is on,
through ``attn_tp_group``. It also confirms world_size==1 is a no-op.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.speculative.eagle_utils as eagle_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _RecordingTPGroup:
    """Fake TP group that records broadcast(tensor, src) calls."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.broadcasts = []

    def broadcast(self, tensor, src):
        self.broadcasts.append((tensor, src))


def _greedy_kernel_stub(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
    topk,
):
    """Stand in for the sgl_kernel greedy verify: accept one draft on a chain."""
    predicts.copy_(target_predict.flatten().to(torch.int32))
    accept_index[0, 0] = 0
    accept_token_num.fill_(1)
    return predicts, accept_index, accept_token_num


def _make_greedy_inputs():
    # argmax over the rows is [1, 0, 2]; near-ties here are what diverge per rank.
    logits = torch.tensor(
        [[1.0, 3.0, 2.0], [5.0, 4.0, 1.0], [0.0, 2.0, 6.0]], dtype=torch.float32
    )
    batch = SimpleNamespace(
        device=torch.device("cpu"),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        sampling_info=SimpleNamespace(
            # False + HIP backend still routes to the greedy (no-broadcast) path.
            is_all_greedy=False,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
        ),
    )
    verify_input = SimpleNamespace(
        draft_token=torch.tensor([10, 11, 12], dtype=torch.int64),
        draft_token_num=3,
        max_tree_depth=3,
        retrieve_index=torch.zeros((1, 3), dtype=torch.int32),
        retrieve_next_token=torch.full((3,), -1, dtype=torch.int32),
        retrieve_next_sibling=torch.full((3,), -1, dtype=torch.int32),
        tree_topk=1,
        grammar=None,
    )
    logits_output = SimpleNamespace(next_token_logits=logits)
    return verify_input, batch, logits_output


class TestEagleVerifyTPBroadcast(unittest.TestCase):
    def _run(self, world_size, dp_attention_enabled=False):
        verify_input, batch, logits_output = _make_greedy_inputs()
        tp_group = _RecordingTPGroup(world_size)
        # eagle_sample selects attn_tp_group when DP-attention is on, else the
        # plain TP group; point both at the recorder so either branch is caught.
        parallel = SimpleNamespace(attn_tp_group=tp_group)
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(eagle_utils, "_is_hip", True)
            )  # force greedy path
            stack.enter_context(patch.object(eagle_utils, "_is_cuda", False))
            stack.enter_context(
                patch.object(eagle_utils, "get_parallel", lambda: parallel)
            )
            stack.enter_context(
                patch("sglang.srt.distributed.get_tp_group", return_value=tp_group)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                    return_value=dp_attention_enabled,
                )
            )
            stack.enter_context(
                patch("sglang.srt.utils.async_probe.sanitize_nan_logits")
            )
            stack.enter_context(
                patch.object(
                    eagle_utils,
                    "verify_tree_greedy_func",
                    side_effect=_greedy_kernel_stub,
                )
            )
            predict, num_correct_tokens, accept_index = eagle_utils.eagle_sample(
                verify_input, batch, logits_output
            )
        return tp_group, predict, num_correct_tokens, accept_index

    def _assert_decision(self, predict, num_correct_tokens, accept_index):
        # Verified decision is computed from the (broadcast) greedy result.
        self.assertEqual(predict.tolist(), [1, 0, 2])
        self.assertEqual(accept_index.tolist(), [[0, -1, -1]])
        self.assertEqual(num_correct_tokens.tolist(), [2])  # drafts + bonus token

    def test_greedy_path_broadcasts_when_tp_gt_1(self):
        tp_group, predict, num_correct_tokens, accept_index = self._run(world_size=2)
        self._assert_decision(predict, num_correct_tokens, accept_index)
        # predict, accept_index, num_correct_drafts (pre-bonus) broadcast from rank 0.
        self.assertEqual([src for _, src in tp_group.broadcasts], [0, 0, 0])
        self.assertIs(tp_group.broadcasts[0][0], predict)
        self.assertIs(tp_group.broadcasts[1][0], accept_index)
        self.assertEqual(tp_group.broadcasts[2][0].tolist(), [1])

    def test_greedy_path_broadcasts_via_attn_tp_group_with_dp_attention(self):
        tp_group, predict, num_correct_tokens, accept_index = self._run(
            world_size=2, dp_attention_enabled=True
        )
        self._assert_decision(predict, num_correct_tokens, accept_index)
        # DP-attention routes the broadcast through attn_tp_group.
        self.assertEqual([src for _, src in tp_group.broadcasts], [0, 0, 0])

    def test_greedy_path_no_broadcast_when_single_rank(self):
        tp_group, predict, num_correct_tokens, accept_index = self._run(world_size=1)
        self._assert_decision(predict, num_correct_tokens, accept_index)
        self.assertEqual(tp_group.broadcasts, [])


if __name__ == "__main__":
    unittest.main()
