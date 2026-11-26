"""HIP prefill-graph gate and DP vote under decoder SWA bounded replay.

The HIP prefill graph runs the late layers on every row, so a batch past 2048
extend tokens, or one that hits a cached prefix, stays eager. The DP vote applies
the same rule, so no rank replays the graph while a peer runs the step eagerly.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=5, suite="base-b-test-1-gpu-small-amd-mi35x")

MAX_EXTEND_TOKENS = 2048


def _backend(*, bounded_replay=True):
    backend = object.__new__(DeepseekV4HipRadixBackend)
    backend.enable_decoder_swa_bounded_replay = bounded_replay
    return backend


def _forward_batch(extend_lens, prefix_lens):
    seq_lens = [e + p for e, p in zip(extend_lens, prefix_lens)]
    return SimpleNamespace(
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        extend_seq_lens_cpu=extend_lens,
        extend_prefix_lens_cpu=prefix_lens,
    )


def _schedule_batch(extend_lens, prefix_lens):
    return SimpleNamespace(extend_lens=extend_lens, prefix_lens=prefix_lens)


class TestHipPrefillGraphGate(CustomTestCase):
    def test_small_batch_without_a_prefix_hit_runs_the_graph(self):
        batch = _forward_batch([1000, 500], [0, 0])
        self.assertTrue(_backend().can_run_prefill_cuda_graph(batch))

    def test_cutoff_is_inclusive_at_2048_extend_tokens(self):
        backend = _backend()
        at_limit = _forward_batch([MAX_EXTEND_TOKENS - 1000, 1000], [0, 0])
        over_limit = _forward_batch([MAX_EXTEND_TOKENS - 999, 1000], [0, 0])
        self.assertTrue(backend.can_run_prefill_cuda_graph(at_limit))
        self.assertFalse(backend.can_run_prefill_cuda_graph(over_limit))

    def test_any_prefix_hit_stays_eager(self):
        batch = _forward_batch([100, 100], [0, 64])
        self.assertFalse(_backend().can_run_prefill_cuda_graph(batch))

    def test_without_bounded_replay_neither_rule_applies(self):
        backend = _backend(bounded_replay=False)
        long_batch = _forward_batch([MAX_EXTEND_TOKENS + 1], [0])
        prefix_hit = _forward_batch([100], [64])
        self.assertTrue(backend.can_run_prefill_cuda_graph(long_batch))
        self.assertTrue(backend.can_run_prefill_cuda_graph(prefix_hit))

    def test_empty_batch_runs_the_graph(self):
        batch = SimpleNamespace(
            seq_lens_cpu=torch.zeros(0, dtype=torch.int64),
            extend_seq_lens_cpu=[],
            extend_prefix_lens_cpu=[],
        )
        self.assertTrue(_backend().can_run_prefill_cuda_graph(batch))


class TestHipPrefillGraphVote(CustomTestCase):
    CASES = (
        ([1000, 500], [0, 0]),
        ([MAX_EXTEND_TOKENS], [0]),
        ([MAX_EXTEND_TOKENS + 1], [0]),
        ([100, 100], [0, 64]),
        ([4096], [0]),
    )

    def test_vote_agrees_with_the_graph_gate(self):
        """Both read one rule, so a rank never replays while a peer runs eagerly."""
        for bounded_replay in (True, False):
            backend = _backend(bounded_replay=bounded_replay)
            for extend_lens, prefix_lens in self.CASES:
                with self.subTest(
                    bounded_replay=bounded_replay,
                    extend_lens=extend_lens,
                    prefix_lens=prefix_lens,
                ):
                    gate = backend.can_run_prefill_cuda_graph(
                        _forward_batch(extend_lens, prefix_lens)
                    )
                    vote = backend.vote_prefill_cuda_graph(
                        _schedule_batch(extend_lens, prefix_lens)
                    )
                    self.assertEqual(gate, vote)

    def test_rank_over_the_cutoff_sends_its_peer_eager(self):
        """One DP rank at 2048 tokens and one at 4096 both vote on the same rule, so
        the min-reduced vote is eager."""
        backend = _backend()
        votes = [
            backend.vote_prefill_cuda_graph(_schedule_batch([2048], [0])),
            backend.vote_prefill_cuda_graph(_schedule_batch([4096], [0])),
        ]
        self.assertEqual(votes, [True, False])
        self.assertFalse(all(votes))

    def test_converted_decode_batch_votes_eager(self):
        """`convert_decode_to_extend` makes each decode request a 1-token extend with
        `seq_len - 1` as its prefix. That is a prefix hit, so the gate refuses it and
        the vote must too."""
        backend = _backend()
        seq_lens = [300, 41]
        extend_lens = [1] * len(seq_lens)
        prefix_lens = [n - 1 for n in seq_lens]
        self.assertFalse(
            backend.can_run_prefill_cuda_graph(_forward_batch(extend_lens, prefix_lens))
        )
        self.assertFalse(
            backend.vote_prefill_cuda_graph(_schedule_batch(extend_lens, prefix_lens))
        )


if __name__ == "__main__":
    unittest.main()
