import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.attention.dmtp_attention import (
    AscendDMTPAttention,
)
from sglang.srt.speculative.dmtp_eager import pack_eager_commit


def run_attention(*, draft_block, batch, mapping, values, num_queries):
    # Zero Q/K makes the visible V rows contribute with uniform weights.
    values = values.reshape(-1, 2, 1, 1)
    pool = SimpleNamespace(get_kv_buffer=lambda _: (torch.zeros_like(values), values))
    backend = SimpleNamespace(
        token_to_kv_pool=pool, req_to_token_pool=SimpleNamespace(req_to_token=mapping)
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=1,
        v_head_dim=1,
        scaling=1.0,
    )
    attention = AscendDMTPAttention(backend, draft_block=draft_block)
    return attention(
        torch.zeros(num_queries, 2), None, None, layer, batch, save_kv_cache=False
    )


class TestAscendDMTPEager(unittest.TestCase):
    def test_dflash_keeps_last_4096_rows_and_all_block_slots(self):
        values = torch.zeros(4110)
        values[[0, 1, 4098, 4099]] = 50000
        # The last block row must be visible even to the anchor query.
        values[-1] = 4104
        batch = SimpleNamespace(
            seq_lens=torch.tensor([4098]),
            seq_lens_cpu=torch.tensor([4106]),
            req_pool_indices=torch.tensor([0]),
            batch_size=1,
            out_cache_loc=torch.arange(4102, 4110),
            spec_info=SimpleNamespace(draft_token_num=8),
        )
        out = run_attention(
            draft_block=True,
            batch=batch,
            mapping=torch.arange(4110).reshape(1, -1),
            values=values,
            num_queries=8,
        )
        torch.testing.assert_close(out, torch.ones(8, 2))

    def test_native_mtp_uses_token_mapping_and_bottom_right_causal_mask(self):
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 0]),
            extend_prefix_lens_cpu=[2, 1],
            extend_seq_lens_cpu=[2, 1],
        )
        out = run_attention(
            draft_block=False,
            batch=batch,
            mapping=torch.tensor([[8, 2, 0, 0], [6, 0, 3, 7]]),
            values=torch.arange(10, dtype=torch.float32),
            num_queries=3,
        )
        torch.testing.assert_close(
            out, torch.tensor([3.0, 4.0, 5.0])[:, None].expand(-1, 2)
        )

    def test_commit_packs_previous_token_rows_and_appends_next_anchor(self):
        candidates = torch.arange(24).reshape(3, 8)
        bonus = torch.tensor([30, 31, 32])
        hidden = candidates.unsqueeze(-1).expand(-1, -1, 2).float()
        slots = candidates + 100
        tokens, rows, locations, tails = pack_eager_commit(
            candidates, bonus, hidden, slots, [1, 4, 8]
        )
        expected_rows = torch.tensor([0, 8, 9, 10, 11, *range(16, 24)])
        self.assertEqual(tokens.tolist(), [30, 9, 10, 11, 31, *range(17, 24), 32])
        torch.testing.assert_close(rows, expected_rows[:, None].expand(-1, 2).float())
        torch.testing.assert_close(locations, expected_rows + 100)
        self.assertEqual(tails, [0, 4, 12])
        with self.assertRaises(ValueError):
            pack_eager_commit(candidates, bonus, hidden, slots, [0, 4, 8])


if __name__ == "__main__":
    unittest.main()
