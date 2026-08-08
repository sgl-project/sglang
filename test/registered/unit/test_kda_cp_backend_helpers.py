"""Unit tests for srt/layers/attention/linear/kda_cp.py.

Derived-property tests for the KDA CP shard metadata and conv-halo assembly:
the halo/global-tail windows are reconstructed from per-rank tails only
(one all-gather of `window` tokens per sequence per rank), and the math that
makes this exact — right-aligned rolling windows, multi-rank reach-back when
shards are shorter than the window, zero-padding for fresh sequences, prior
pool-window chaining, empty-shard compaction — is what these cases pin down.
The reference computes every window directly on the full token stream.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kda_cp import (
    KDACPPrefillMetadata,
    build_kda_cp_prefill_metadata,
    exchange_kda_conv_halo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DIM = 4
WINDOW = 3


class _StackGather:
    """all_gather_into_tensor double fed with precomputed per-rank tails."""

    def __init__(self, stacked: torch.Tensor):
        self.stacked = stacked

    def __call__(self, out, inp, group=None):
        out.copy_(self.stacked)


def _token(seq_idx: int, pos: int) -> torch.Tensor:
    """A distinguishable token embedding: value = 100*seq + pos + 1."""
    return torch.full((DIM,), float(100 * seq_idx + pos + 1))


def _build_streams(seq_lens):
    return [
        (
            torch.stack([_token(n, p) for p in range(length)])
            if length > 0
            else torch.zeros(0, DIM)
        )
        for n, length in enumerate(seq_lens)
    ]


def _ref_window(stream: torch.Tensor, prior: torch.Tensor, upto: int):
    """Last WINDOW tokens of (prior ++ stream[:upto]), right-aligned."""
    virtual = torch.cat([prior, stream[:upto]], dim=0)
    return virtual[-WINDOW:]


class TestKDACPConvHalo(CustomTestCase):
    def _run_all_ranks(self, seq_lens, world_size, has_prior_flags):
        streams = _build_streams(seq_lens)
        n_global = len(seq_lens)
        prior = torch.stack(
            [
                (
                    torch.stack([_token(n, -(WINDOW - j)) - 50 for j in range(WINDOW)])
                    if has_prior_flags[n]
                    else torch.zeros(WINDOW, DIM)
                )
                for n in range(n_global)
            ]
        )
        has_prior = torch.tensor(has_prior_flags, dtype=torch.bool)
        prior_or_zero = [
            prior[n] if has_prior_flags[n] else torch.zeros(WINDOW, DIM)
            for n in range(n_global)
        ]

        metas = [
            build_kda_cp_prefill_metadata(
                seq_lens, world_size=world_size, rank=r, device="cpu"
            )
            for r in range(world_size)
        ]
        # Precompute every rank's tails to feed the gather double.
        from sglang.srt.layers.attention.linear.kda_cp import _collect_local_tails

        conv_inputs = []
        for r in range(world_size):
            pieces = []
            offset = [0] * n_global
            for j in range(r):
                for n in range(n_global):
                    offset[n] += metas[j].shard_lens[j][n]
            for n in metas[r].local_seq_ids_list:
                lo = offset[n]
                hi = lo + metas[r].shard_lens[r][n]
                pieces.append(streams[n][lo:hi])
            conv_inputs.append(
                torch.cat(pieces, dim=0) if pieces else torch.zeros(0, DIM)
            )
        stacked = torch.stack(
            [
                _collect_local_tails(conv_inputs[r], metas[r], WINDOW)
                for r in range(world_size)
            ]
        )

        results = []
        with patch(
            "torch.distributed.all_gather_into_tensor", new=_StackGather(stacked)
        ):
            for r in range(world_size):
                results.append(
                    exchange_kda_conv_halo(
                        conv_input=conv_inputs[r],
                        metadata=metas[r],
                        prior_conv_windows=prior,
                        has_prior=has_prior,
                        group=object(),
                    )
                )

        # Verify against the direct full-stream reference.
        for r in range(world_size):
            halo, halo_has_initial, global_tails = results[r]
            offset = [0] * n_global
            for j in range(r):
                for n in range(n_global):
                    offset[n] += metas[j].shard_lens[j][n]
            for i, n in enumerate(metas[r].local_seq_ids_list):
                expect = _ref_window(streams[n], prior_or_zero[n], upto=offset[n])
                torch.testing.assert_close(
                    halo[i], expect, rtol=0, atol=0, msg=f"rank {r} seq {n} halo"
                )
                self.assertEqual(
                    bool(halo_has_initial[i]),
                    bool(has_prior_flags[n] or offset[n] > 0),
                    f"rank {r} seq {n} has_initial",
                )
            for n in range(n_global):
                expect = _ref_window(streams[n], prior_or_zero[n], upto=seq_lens[n])
                torch.testing.assert_close(
                    global_tails[n],
                    expect,
                    rtol=0,
                    atol=0,
                    msg=f"rank {r} seq {n} global tail",
                )
        # Global tails must be identical across ranks (pool replicas).
        for r in range(1, world_size):
            torch.testing.assert_close(results[0][2], results[r][2], rtol=0, atol=0)

    def test_basic_two_seqs(self):
        self._run_all_ranks([10, 7], world_size=2, has_prior_flags=[False, False])

    def test_multi_rank_reach_back(self):
        # seq of 5 over CP4 -> shards [1, 1, 1, 2]: rank 3's halo spans THREE
        # earlier ranks' single-token shards.
        self._run_all_ranks([5], world_size=4, has_prior_flags=[False])

    def test_empty_shards_and_zero_pad(self):
        # seq of 2 over CP4 -> shards [0, 1, 0, 1]: empty shards are dropped
        # from the local batch but must not break window assembly.
        self._run_all_ranks([2, 9], world_size=4, has_prior_flags=[False, False])

    def test_prior_window_chaining(self):
        # Chunked-prefill continuation: rank 0's halo IS the carried pool
        # window, and a 1-token sequence's global tail combines prior + new.
        self._run_all_ranks([1, 6], world_size=2, has_prior_flags=[True, True])

    def test_mixed_prior(self):
        self._run_all_ranks([4, 4], world_size=4, has_prior_flags=[True, False])


class TestKDACPPrefillMetadata(CustomTestCase):
    def test_compaction_and_shard_lens(self):
        meta = build_kda_cp_prefill_metadata(
            [5, 640], world_size=8, rank=0, device="cpu"
        )
        # rank 0 of a 5-token seq over CP8 holds zero tokens -> compacted out.
        self.assertEqual(meta.local_seq_ids_list, [1])
        self.assertEqual(meta.query_start_loc.tolist(), [0, 80])
        # Shard length tables cover every rank and sum to the sequence.
        for n, total in enumerate([5, 640]):
            self.assertEqual(sum(meta.shard_lens[r][n] for r in range(8)), total)
        self.assertIsInstance(meta, KDACPPrefillMetadata)

    def test_cp_context_carries_compaction(self):
        meta = build_kda_cp_prefill_metadata(
            [5, 640], world_size=8, rank=3, device="cpu"
        )
        ctx = meta.to_cp_context(group=object())
        self.assertEqual(ctx.num_global_seqs, 2)
        self.assertEqual(ctx.local_seq_ids.tolist(), meta.local_seq_ids_list)


if __name__ == "__main__":
    unittest.main(verbosity=3)
