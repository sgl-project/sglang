"""P2P shard reconstruction must scope all consumers of runtime topology."""

import unittest

import torch

from sglang.srt.distributed.parallel_state import (
    ParallelismContext,
    RankParallelismConfig,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestParallelismContext(unittest.TestCase):
    def setUp(self):
        config = get_context().override_server_args()
        config.install()
        self.addCleanup(config.restore)

    def test_round_trip_and_nested_exception_restore_topology(self):
        config = RankParallelismConfig(
            tp_size=2,
            tp_rank=1,
            ep_size=2,
            ep_rank=1,
            attn_dp_size=2,
            attn_dp_rank=1,
            world_size=2,
            global_rank=1,
            local_rank=1,
        )
        with ParallelismContext(config):
            self.assertEqual(RankParallelismConfig.from_parallel_state(1), config)
            outer_group = get_parallel().tp_group
            self.assertTrue(get_parallel().enable_dp_attention)
            with self.assertRaisesRegex(RuntimeError, "restore"):
                with ParallelismContext(RankParallelismConfig()):
                    self.assertEqual(get_parallel().tp_size, 1)
                    self.assertFalse(get_parallel().enable_dp_attention)
                    raise RuntimeError("restore")
            self.assertIs(get_parallel().tp_group, outer_group)
            self.assertEqual(get_parallel().tp_rank, 1)
        self.assertEqual(get_parallel().tp_size, 1)

    def test_invalid_topology_does_not_leak(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            with ParallelismContext(RankParallelismConfig(tp_size=2)):
                self.fail("invalid topology accepted")
        self.assertEqual(get_parallel().tp_size, 1)

    def test_real_linear_parameters_follow_requested_shard(self):
        from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

        config = RankParallelismConfig(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
            world_size=2,
            global_rank=1,
        )
        with ParallelismContext(config), torch.device("cpu"):
            row = RowParallelLinear(128, 64)
            column = ColumnParallelLinear(128, 64)
            self.assertEqual(row.weight.shape, (64, 64))
            self.assertEqual(column.weight.shape, (32, 128))
            self.assertEqual((row.tp_rank, column.tp_rank), (1, 1))
        with ParallelismContext(RankParallelismConfig()), torch.device("cpu"):
            self.assertEqual(RowParallelLinear(128, 64).weight.shape, (64, 128))


if __name__ == "__main__":
    unittest.main()
