"""CPU checks for the Kimi-K3 sequence-parallel residual stream over the CP group.

The rank-major zigzag block layout that `_sp_cp_rank_major_blocks` produces for a
reduce-scatter must equal, per rank, what `ZigzagCPStrategy.shard_hidden_states`
gives that rank (plus its zero padding), so that RS(blocks)[rank] lands exactly
the rows the rank's residual stream holds."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.cp.padding import pad_local_rows, pad_logical_token_to_physical
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.models.kimi_k3 import _sp_cp_rank_major_blocks
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSPResidualRankMajorBlocks(CustomTestCase):
    def _check(self, cp_size, extend_lens):
        num_tokens = sum(extend_lens)
        hidden = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(
            1
        ) * torch.ones(1, 3)
        with get_parallel().override(attn_cp_rank=0, attn_cp_size=cp_size):
            strategy = ZigzagCPStrategy(cp_size=cp_size)
            metadata = strategy.build_metadata(num_tokens, extend_lens)
            pad_logical_token_to_physical(metadata)  # as prepare_cp_forward does
            blocks = _sp_cp_rank_major_blocks(hidden, metadata, cp_size)
        phys = metadata.per_rank_actual_token[0]
        self.assertEqual(tuple(blocks.shape), (phys * cp_size, 3))
        for rank in range(cp_size):
            with get_parallel().override(attn_cp_rank=rank, attn_cp_size=cp_size):
                strategy_r = ZigzagCPStrategy(cp_size=cp_size)
                meta_r = strategy_r.build_metadata(num_tokens, extend_lens)
                pad_logical_token_to_physical(meta_r)
                forward_batch = SimpleNamespace(attn_cp_metadata=meta_r)
                local = strategy_r.shard_hidden_states(hidden, forward_batch)
                local = pad_local_rows(local, meta_r, dim=0)
            with self.subTest(rank=rank):
                torch.testing.assert_close(
                    blocks[rank * phys : (rank + 1) * phys], local, rtol=0, atol=0
                )

    def test_single_balanced_request(self):
        self._check(8, [32768 // 64])

    def test_ragged_requests_with_padding(self):
        self._check(8, [100, 33])
        self._check(4, [257, 16, 4096 // 16])

    def test_cp2(self):
        self._check(2, [7, 9, 24])


if __name__ == "__main__":
    unittest.main()
