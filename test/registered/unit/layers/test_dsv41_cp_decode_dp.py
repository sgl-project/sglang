"""Decode query shards must reconstruct the batch, including empty ranks."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.cp_decode_attn_dp import CPDecodeQueryLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCPDecodeDP(CustomTestCase):
    def test_uneven_verify_and_ghost_shards_round_trip_without_reordering(self):
        size = 4
        for rows in (1, 17, 144):
            source = torch.arange(rows * 3).view(rows, 3)
            shards = []
            for rank in range(size):
                layout = CPDecodeQueryLayout(rows, rank, size)
                shard = layout.select(source, value=-1)
                shards.append(shard)
            with patch(
                "sglang.srt.layers.cp.interleave.attn_cp_interleave_gather",
                return_value=torch.cat(shards),
            ):
                for rank, shard in enumerate(shards):
                    layout = CPDecodeQueryLayout(rows, rank, size)
                    torch.testing.assert_close(layout.gather(shard), source)


if __name__ == "__main__":
    unittest.main()
