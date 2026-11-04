from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.spec_utils import generate_tree_mask_func
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, suite="base-b-kernel-unit-1-gpu-large")


def _make_batch(seq_lens: list[int]):
    return SimpleNamespace(
        reqs=[object() for _ in seq_lens],
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
    )


def _make_req_masks(seq_lens: list[int], draft_token_num: int) -> torch.Tensor:
    req_masks = []
    for req_idx, seq_len in enumerate(seq_lens):
        values = torch.arange(
            draft_token_num * seq_len,
            dtype=torch.int64,
            device="cuda",
        )
        req_mask = (values + req_idx) % 3 != 0
        req_masks.append(req_mask.reshape(draft_token_num, seq_len).flatten())
    return torch.cat(req_masks)


def _make_tree_masks(batch_size: int, draft_token_num: int) -> torch.Tensor:
    values = torch.arange(
        batch_size * draft_token_num * draft_token_num,
        dtype=torch.int64,
        device="cuda",
    )
    return ((values + 1) % 4 != 0).reshape(batch_size, draft_token_num, draft_token_num)


def _reference_generate_tree_mask(
    req_masks: torch.Tensor,
    tree_masks: torch.Tensor,
    seq_lens: list[int],
    draft_token_num: int,
) -> torch.Tensor:
    expected = []
    req_mask_offset = 0
    for req_idx, seq_len in enumerate(seq_lens):
        req_mask_size = draft_token_num * seq_len
        req_mask = req_masks[req_mask_offset : req_mask_offset + req_mask_size]
        req_mask = req_mask.reshape(draft_token_num, seq_len)
        tree_mask = tree_masks[req_idx].reshape(draft_token_num, draft_token_num)
        expected.append(torch.cat((req_mask, tree_mask), dim=1).flatten())
        req_mask_offset += req_mask_size
    return torch.cat(expected)


class TestNgramGenerateTreeMask(CustomTestCase):
    def test_generate_tree_mask_matches_reference(self):
        for seq_lens, draft_token_num in [
            ([1], 1),
            ([1, 3], 4),
            ([1, 255, 256, 257], 8),
            ([2, 17, 257], 12),
            ([1, 128, 300, 1025], 24),
            ([1, 1024, 4099, 8192, 32768, 65536], 128),
        ]:
            with self.subTest(seq_lens=seq_lens, draft_token_num=draft_token_num):
                batch = _make_batch(seq_lens)
                req_masks = _make_req_masks(seq_lens, draft_token_num)
                tree_masks = _make_tree_masks(len(seq_lens), draft_token_num)
                output = torch.empty(
                    draft_token_num * (sum(seq_lens) + len(seq_lens) * draft_token_num),
                    dtype=torch.bool,
                    device="cuda",
                )

                generate_tree_mask_func(
                    req_masks, tree_masks, batch, draft_token_num, output
                )

                expected = _reference_generate_tree_mask(
                    req_masks, tree_masks, seq_lens, draft_token_num
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(output, expected))


if __name__ == "__main__":
    unittest.main()
