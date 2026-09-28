"""Broadcasting a DP shard's requests from its (attn-TP 0, attn-CP 0) rank, rank by rank.

Only that rank holds the data; every other rank passes None and must end with it.
A rank acting as the source of a broadcast must already hold the data.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed import communication_op
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAttnCpTpBroadcast(CustomTestCase):
    def run_shard(self, cp_size, tp_size):
        data = ["req-a", "req-b"]
        # Global rank of (cp, tp) within one DP shard.
        rank_of = {
            (cp, tp): cp * tp_size + tp
            for cp in range(cp_size)
            for tp in range(tp_size)
        }
        sent = {}

        def broadcast_pyobj(obj, rank, group, src):
            # Stand-in for the collective: the source sends what it holds and
            # every other member receives it.
            if rank == src:
                self.assertIsNotNone(obj, f"rank {rank} is a source without the data")
                sent[group] = obj
                return obj
            self.assertIn(group, sent, "a receiver ran before its source")
            return sent[group]

        results = {}
        # Collectives order the ranks for real; here each source runs before
        # the ranks it sends to.
        for tp in range(tp_size):
            for cp in range(cp_size):
                rank = rank_of[(cp, tp)]
                tp_group = SimpleNamespace(
                    world_size=tp_size,
                    rank_in_group=tp,
                    rank=rank,
                    ranks=[rank_of[(cp, t)] for t in range(tp_size)],
                    cpu_group=("attn_tp", cp),
                )
                cp_group = SimpleNamespace(
                    world_size=cp_size,
                    rank_in_group=cp,
                    rank=rank,
                    ranks=[rank_of[(c, tp)] for c in range(cp_size)],
                    cpu_group=("attn_cp", tp),
                )
                with (
                    patch.object(
                        communication_op, "get_attn_tp_group", return_value=tp_group
                    ),
                    patch.object(
                        communication_op, "get_attn_cp_group", return_value=cp_group
                    ),
                    patch.object(communication_op, "broadcast_pyobj", broadcast_pyobj),
                ):
                    results[(cp, tp)] = communication_op.attn_cp_tp_broadcast_pyobj(
                        data if (cp, tp) == (0, 0) else None
                    )
        self.assertEqual(results, {key: data for key in rank_of})

    def test_every_rank_of_the_shard_receives_the_data(self):
        for cp_size, tp_size in [(2, 2), (2, 1), (1, 2), (4, 2), (1, 1)]:
            with self.subTest(cp_size=cp_size, tp_size=tp_size):
                self.run_shard(cp_size, tp_size)


if __name__ == "__main__":
    unittest.main()
