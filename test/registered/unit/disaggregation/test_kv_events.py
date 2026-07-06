"""Unit tests for srt/disaggregation/kv_events KV-event publisher rank selection.

Covers the data-parallel rank used to offset each scheduler's KV-event
publisher port, across pure DP, DP-attention, and single-replica modes. The
port offset must make every independent KV cache publish on a distinct port so
the router can subscribe per replica (the `dp_size` it reads from
`/server_info`).
"""

import unittest

from sglang.srt.disaggregation.kv_events import (
    ZmqEventPublisher,
    select_kv_publisher_dp_rank,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestSelectKvPublisherDpRank(CustomTestCase):
    def test_select_rank_across_modes(self):
        # (label, attn_dp_size, attn_dp_rank, dp_rank, expected)
        cases = [
            # Pure DP (no dp-attention): attn_dp_rank is 0 for every worker,
            # so the replica is distinguished by dp_rank.
            ("pure_dp_worker0", 1, 0, 0, 0),
            ("pure_dp_worker1", 1, 0, 1, 1),
            ("pure_dp_worker3", 1, 0, 3, 3),
            # DP-attention: each attn-dp rank owns a KV shard; distinguish by
            # attn_dp_rank. dp_rank is ignored entirely in this mode.
            ("dp_attention_rank0", 2, 0, None, 0),
            ("dp_attention_rank1", 2, 1, None, 1),
            ("dp_attention_ignores_dp_rank", 2, 1, 99, 1),
            # Single replica / no DP.
            ("single_dp_rank_none", 1, 0, None, 0),
            ("single_dp_rank_zero", 1, 0, 0, 0),
        ]
        for label, attn_dp_size, attn_dp_rank, dp_rank, expected in cases:
            with self.subTest(label):
                self.assertEqual(
                    select_kv_publisher_dp_rank(attn_dp_size, attn_dp_rank, dp_rank),
                    expected,
                )

    def test_workers_bind_sequential_ports_per_replica(self):
        # Each replica r must publish on port_base + r, since the router opens
        # one SUB socket per rank at port_base + r. Regression: pre-fix every
        # pure-DP worker offset by attn_dp_rank == 0, so all collapsed onto the
        # single port tcp://*:5557 -> the 2nd worker crashed binding an
        # already-bound port.
        endpoint = "tcp://*:5557"
        expected = [f"tcp://*:{5557 + r}" for r in range(4)]

        # Pure DP: replica index is dp_rank (attn_dp_rank is 0 for all).
        pure_dp = [
            ZmqEventPublisher.offset_endpoint_port(
                endpoint, select_kv_publisher_dp_rank(1, 0, r)
            )
            for r in range(4)
        ]
        self.assertEqual(pure_dp, expected)

        # DP-attention: replica index is attn_dp_rank.
        dp_attention = [
            ZmqEventPublisher.offset_endpoint_port(
                endpoint, select_kv_publisher_dp_rank(4, a, None)
            )
            for a in range(4)
        ]
        self.assertEqual(dp_attention, expected)

    def test_publisher_rank_count_matches_advertised_dp_size(self):
        # The router subscribes to `dp_size` per-rank ports (from /server_info).
        # The engine must produce exactly `dp_size` distinct publisher ranks in
        # both modes, otherwise some subscribed ports get no data.
        for dp_size in (1, 2, 4):
            with self.subTest(f"pure_dp_{dp_size}"):
                ranks = {
                    select_kv_publisher_dp_rank(
                        attn_dp_size=1, attn_dp_rank=0, dp_rank=r
                    )
                    for r in range(dp_size)
                }
                self.assertEqual(len(ranks), dp_size)
            with self.subTest(f"dp_attention_{dp_size}"):
                ranks = {
                    select_kv_publisher_dp_rank(
                        attn_dp_size=dp_size, attn_dp_rank=a, dp_rank=None
                    )
                    for a in range(dp_size)
                }
                self.assertEqual(len(ranks), dp_size)


if __name__ == "__main__":
    unittest.main()
