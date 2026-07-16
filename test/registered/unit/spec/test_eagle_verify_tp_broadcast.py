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
test drives the greedy path with world_size>1 and asserts the finalized decision
is broadcast (and is a no-op when world_size==1).
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative import eagle_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _RecordingTPGroup:
    """Fake TP group that records broadcast(tensor, src) calls."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.broadcasts = []

    def broadcast(self, tensor, src):
        self.broadcasts.append((tensor, src))


def _make_greedy_inputs(bs=2, draft_token_num=4, vocab=8):
    sampling_info = MagicMock()
    sampling_info.is_all_greedy = True
    sampling_info.acc_additive_penalties = None
    sampling_info.acc_scaling_penalties = None
    sampling_info.logit_bias = None

    batch = MagicMock()
    batch.device = "cpu"
    batch.forward_mode.is_idle.return_value = False
    batch.seq_lens = torch.arange(bs)
    batch.sampling_info = sampling_info

    logits_output = MagicMock()
    logits_output.next_token_logits = torch.randn(bs * draft_token_num, vocab)

    verify_input = MagicMock()
    verify_input.draft_token = torch.zeros(bs * draft_token_num, dtype=torch.int64)
    verify_input.draft_token_num = draft_token_num
    verify_input.max_tree_depth = draft_token_num
    verify_input.tree_topk = 1
    verify_input.grammar = None
    return verify_input, batch, logits_output


def _greedy_kernel_stub(*_, predicts, accept_index, accept_token_num, **__):
    """Stand in for the sgl_kernel greedy verify; return the mutable buffers."""
    return predicts, accept_index, accept_token_num


class TestEagleVerifyTPBroadcast(unittest.TestCase):
    def _run(self, world_size):
        verify_input, batch, logits_output = _make_greedy_inputs()
        tp_group = _RecordingTPGroup(world_size)
        with patch("sglang.srt.distributed.get_tp_group", return_value=tp_group), patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
            return_value=False,
        ), patch("sglang.srt.utils.async_probe.sanitize_nan_logits"), patch.object(
            eagle_utils, "verify_tree_greedy_func", side_effect=_greedy_kernel_stub
        ):
            predict, _, accept_index = eagle_utils.eagle_sample(
                verify_input, batch, logits_output
            )
        return tp_group, predict, accept_index

    def test_greedy_path_broadcasts_when_tp_gt_1(self):
        tp_group, predict, accept_index = self._run(world_size=2)
        # predict, accept_index, num_correct_drafts must all be broadcast from rank 0.
        self.assertEqual(len(tp_group.broadcasts), 3)
        self.assertTrue(all(src == 0 for _, src in tp_group.broadcasts))
        broadcast_tensors = [t for t, _ in tp_group.broadcasts]
        self.assertIn(predict, broadcast_tensors)
        self.assertIn(accept_index, broadcast_tensors)

    def test_greedy_path_no_broadcast_when_single_rank(self):
        tp_group, _, _ = self._run(world_size=1)
        self.assertEqual(tp_group.broadcasts, [])


if __name__ == "__main__":
    unittest.main()
