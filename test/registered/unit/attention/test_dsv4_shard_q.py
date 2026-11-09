# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the DeepSeek-V4 ShardQ split and exchange index math.

These cover the parts that are easy to get subtly wrong without a GPU: the
zigzag ownership must partition the batch exactly once, each rank's runs must
keep their own causal lengths, every rank must make the same fallback
decision, and the two all-to-alls must be exact inverses. The collectives are
simulated on CPU; kernel numerics live in the GPU suites.
"""

import unittest

import torch

from sglang.srt.layers.attention.dsv4.shard_q import (
    ShardQ,
    split_runs,
    step_amortizes_shard_q,
    zigzag_owner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _Group:
    def __init__(self, world: int, rank: int) -> None:
        self.world_size = world
        self.rank_in_group = rank


class _SimulatedGroup(_Group):
    """Equal-split all-to-all over pre-created per-rank send buffers."""

    def __init__(self, world: int, rank: int, rows: int, buffers: dict) -> None:
        super().__init__(world, rank)
        self.rows = rows
        self.buffers = buffers

    def all_to_all_single(self, recv: torch.Tensor, send: torch.Tensor) -> None:
        self.buffers[self.rank_in_group] = send.clone()
        if send.shape[0] == self.world_size:
            # gather_o: send is [world, rows, ...]; chunk r goes to rank r.
            chunks = [
                self.buffers[s][self.rank_in_group] for s in range(self.world_size)
            ]
            recv.copy_(torch.stack(chunks, dim=0))
        else:
            # scatter_q: send is [world * rows, ...]; chunk r is rows[r*rows:].
            start = self.rank_in_group * self.rows
            chunks = [
                self.buffers[s][start : start + self.rows]
                for s in range(self.world_size)
            ]
            recv.copy_(torch.cat(chunks, dim=0))


def _plans(extend, seq, world, buffers=None):
    rows = None
    plans = []
    for rank in range(world):
        group = (
            _Group(world, rank)
            if buffers is None
            else _SimulatedGroup(world, rank, rows, buffers)
        )
        plans.append(
            ShardQ.build(
                extend_lens=extend,
                seq_lens=seq,
                group=group,
                device=torch.device("cpu"),
            )
        )
    return plans


def _build_with_rows(extend, seq, world, buffers):
    plans = _plans(extend, seq, world)
    assert all(plan is not None for plan in plans)
    rows = plans[0].rows_per_rank
    out = []
    for rank, plan in enumerate(plans):
        group = _SimulatedGroup(world, rank, rows, buffers)
        # Keep the exchange plan, but attach the group the methods will use.
        out.append(
            ShardQ(
                group=group,
                rows_per_rank=plan.rows_per_rank,
                num_tokens_per_rank=plan.num_tokens_per_rank,
                token_ids=plan.token_ids,
                token_ids_cpu=plan.token_ids_cpu,
                rows_per_request=plan.rows_per_request,
                send_ids=plan.send_ids,
                recv_ids=plan.recv_ids,
            )
        )
    return out


class TestZigzagSplit(CustomTestCase):
    def test_single_request_balance(self):
        owner = zigzag_owner([1024], 8)
        self.assertEqual(owner.size, 1024)
        self.assertEqual(
            sorted(torch.bincount(torch.tensor(owner), minlength=8).tolist()),
            [128] * 8,
        )
        # Each rank owns one early and one late chunk: ranks 0..7 then 7..0.
        chunks = owner.reshape(16, 64)
        self.assertTrue(all((chunk == chunk[0]).all() for chunk in chunks))
        self.assertEqual(
            chunks[:, 0].tolist(),
            [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
        )

    def test_multi_request_partition(self):
        extend = [1000, 300, 7]
        owner = zigzag_owner(extend, 8)
        self.assertEqual(owner.size, sum(extend))
        # Every token is owned exactly once, and no rank is starved.
        counts = torch.bincount(torch.tensor(owner), minlength=8)
        self.assertGreater(int(counts.min()), 0)

    def test_split_runs_keep_run_local_causal_lengths(self):
        extend, seq = [1024], [1024]
        owner = zigzag_owner(extend, 8)
        local = sorted(torch.nonzero(torch.tensor(owner) == 0).squeeze(1).tolist())
        req, run_qlen, run_req, run_seq = split_runs(local, extend, seq)
        self.assertEqual(run_qlen.tolist(), [64, 64])
        self.assertEqual(run_req.tolist(), [0, 0])
        self.assertEqual(run_seq.tolist(), [64, 1024])

    def test_split_runs_include_prefix(self):
        extend, seq = [1000], [1200]
        owner = zigzag_owner(extend, 8)
        local = sorted(torch.nonzero(torch.tensor(owner) == 0).squeeze(1).tolist())
        _, run_qlen, _, run_seq = split_runs(local, extend, seq)
        # The first run ends at local token 62: prefix 200 + 63.
        self.assertEqual(run_qlen.tolist(), [62, 63])
        self.assertEqual(run_seq[0], 200 + run_qlen[0])


class TestStepGate(CustomTestCase):
    def test_short_step_stays_head_sharded(self):
        self.assertFalse(step_amortizes_shard_q([4096], [4096]))

    def test_step_at_min_tokens_enables(self):
        self.assertTrue(step_amortizes_shard_q([8192], [8192]))

    def test_long_context_tail_chunk_enables(self):
        self.assertTrue(step_amortizes_shard_q([4096], [32768]))

    def test_long_context_short_extend_does_not_drag_short_prefills(self):
        # One long-context request contributes a single token while short
        # prefills bring the step to 1024 tokens: the step is not a tail chunk.
        self.assertFalse(step_amortizes_shard_q([1, 1023], [40000, 1024]))
        self.assertTrue(step_amortizes_shard_q([1024, 1023], [40000, 1024]))


class TestShardQBuild(CustomTestCase):
    def test_partition_and_permutations(self):
        extend, seq = [1000, 300, 7], [1200, 300, 7]
        world = 8
        plans = _plans(extend, seq, world)
        self.assertTrue(all(plan is not None for plan in plans))
        ids = sorted(sum([plan.token_ids_cpu for plan in plans], []))
        self.assertEqual(ids, list(range(sum(extend))))
        for plan in plans:
            self.assertEqual(plan.send_ids.numel(), world * plan.rows_per_rank)
            torch.testing.assert_close(
                plan.send_ids[plan.recv_ids],
                torch.arange(sum(extend)),
            )
            self.assertEqual(sum(plan.rows_per_request), len(plan.token_ids_cpu))

    def test_empty_rank_falls_back_on_every_rank(self):
        # world=8, 3 tokens: several ranks own no token.
        self.assertTrue(all(plan is None for plan in _plans([3], [3], 8)))
        # world=2, 1 token: the other rank owns no token.
        self.assertTrue(all(plan is None for plan in _plans([1], [1], 2)))
        self.assertTrue(all(plan is None for plan in _plans([8], [8], 1)))
        # A one-token-per-rank split is structurally valid; the backend's
        # MIN_TOKENS gate keeps it off the ShardQ path.
        self.assertTrue(all(plan is not None for plan in _plans([8], [8], 8)))

    def test_scatter_gather_round_trip(self):
        world, heads, dim, length = 8, 4, 3, 1024
        buffers: dict[int, torch.Tensor] = {}
        plans = _build_with_rows([length], [length], world, buffers)
        rows = plans[0].rows_per_rank
        q_global = torch.arange(
            length * world * heads * dim, dtype=torch.float32
        ).reshape(length, world * heads, dim)
        for rank, plan in enumerate(plans):
            q_rank = q_global[:, rank * heads : (rank + 1) * heads, :]
            buffers[rank] = q_rank.index_select(0, plan.send_ids)
        for rank, plan in enumerate(plans):
            q_rank = q_global[:, rank * heads : (rank + 1) * heads, :]
            out = plan.scatter_q(q_rank)
            expect = q_global.index_select(
                0, torch.tensor(plan.token_ids_cpu, dtype=torch.int64)
            )
            torch.testing.assert_close(out, expect)

        owner = {}
        for rank, plan in enumerate(plans):
            for row, token in enumerate(plan.token_ids_cpu):
                owner[token] = (rank, row)
        o = [
            torch.arange(
                len(plan.token_ids_cpu) * world * heads * dim,
                dtype=torch.float32,
            ).reshape(len(plan.token_ids_cpu), world * heads, dim)
            + rank * 100000
            for rank, plan in enumerate(plans)
        ]
        for rank, plan in enumerate(plans):
            n = len(plan.token_ids_cpu)
            send = torch.zeros(world, rows, heads, dim)
            send[:, :n] = o[rank].view(n, world, heads, dim).transpose(0, 1)
            buffers[rank] = send
        for rank, plan in enumerate(plans):
            restored = plan.gather_o(o[rank])
            head_slice = slice(rank * heads, (rank + 1) * heads)
            expect = torch.stack(
                [
                    o[owner[token][0]][owner[token][1], head_slice, :]
                    for token in range(length)
                ]
            )
            torch.testing.assert_close(restored, expect)

    def test_scatter_gather_round_trip_uneven(self):
        extend, seq = [1000, 300, 7], [1200, 300, 7]
        heads, dim = 4, 3
        num_tokens = sum(extend)
        for world in (2, 4, 8):
            with self.subTest(world=world):
                buffers: dict[int, torch.Tensor] = {}
                plans = _build_with_rows(extend, seq, world, buffers)
                rows = plans[0].rows_per_rank
                q_global = torch.arange(
                    num_tokens * world * heads * dim, dtype=torch.float32
                ).reshape(num_tokens, world * heads, dim)
                for rank, plan in enumerate(plans):
                    q_rank = q_global[:, rank * heads : (rank + 1) * heads, :]
                    buffers[rank] = q_rank.index_select(0, plan.send_ids)
                for rank, plan in enumerate(plans):
                    q_rank = q_global[:, rank * heads : (rank + 1) * heads, :]
                    out = plan.scatter_q(q_rank)
                    expect = q_global.index_select(
                        0, torch.tensor(plan.token_ids_cpu, dtype=torch.int64)
                    )
                    torch.testing.assert_close(out, expect)

                owner = {}
                for rank, plan in enumerate(plans):
                    for row, token in enumerate(plan.token_ids_cpu):
                        owner[token] = (rank, row)
                o = [
                    torch.arange(
                        len(plan.token_ids_cpu) * world * heads * dim,
                        dtype=torch.float32,
                    ).reshape(len(plan.token_ids_cpu), world * heads, dim)
                    + rank * 100000
                    for rank, plan in enumerate(plans)
                ]
                for rank, plan in enumerate(plans):
                    n = len(plan.token_ids_cpu)
                    send = torch.zeros(world, rows, heads, dim)
                    send[:, :n] = o[rank].view(n, world, heads, dim).transpose(0, 1)
                    buffers[rank] = send
                for rank, plan in enumerate(plans):
                    restored = plan.gather_o(o[rank])
                    head_slice = slice(rank * heads, (rank + 1) * heads)
                    expect = torch.stack(
                        [
                            o[owner[token][0]][owner[token][1], head_slice, :]
                            for token in range(num_tokens)
                        ]
                    )
                    torch.testing.assert_close(restored, expect)


if __name__ == "__main__":
    unittest.main()
