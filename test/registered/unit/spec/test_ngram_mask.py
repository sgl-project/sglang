import unittest

import torch

from sglang.srt.speculative.spec_utils import (
    generate_tree_mask_func,
    pack_ngram_full_mask,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=3, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestNgramMask(CustomTestCase):
    def test_generate_tree_mask_with_explicit_context_mask(self):
        cases = [
            ([1], 1),
            ([1, 3], 4),
            ([1, 255, 256, 257], 8),
            ([2, 17, 257], 12),
            ([1, 128, 300, 1025], 24),
        ]
        for seq_lens_cpu, draft_token_num in cases:
            with self.subTest(seq_lens=seq_lens_cpu, draft_token_num=draft_token_num):
                seq_lens = torch.tensor(seq_lens_cpu, dtype=torch.int32, device="cuda")
                kv_lens = seq_lens + draft_token_num
                kv_indptr = torch.zeros(
                    len(seq_lens_cpu) + 1, dtype=torch.int32, device="cuda"
                )
                kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)

                req_mask_values = torch.arange(
                    draft_token_num * sum(seq_lens_cpu), device="cuda"
                )
                req_masks = req_mask_values % 3 != 0
                tree_mask_values = torch.arange(
                    len(seq_lens_cpu) * draft_token_num**2, device="cuda"
                )
                tree_masks = tree_mask_values % 4 != 0
                output = torch.empty(
                    draft_token_num
                    * (sum(seq_lens_cpu) + len(seq_lens_cpu) * draft_token_num),
                    dtype=torch.bool,
                    device="cuda",
                )

                generate_tree_mask_func(
                    req_masks, tree_masks, kv_indptr, output, draft_token_num
                )

                expected = []
                req_offset = 0
                tree_masks_3d = tree_masks.view(-1, draft_token_num, draft_token_num)
                for req_idx, seq_len in enumerate(seq_lens_cpu):
                    req_size = draft_token_num * seq_len
                    req_mask = req_masks[req_offset : req_offset + req_size].view(
                        draft_token_num, seq_len
                    )
                    expected.append(
                        torch.cat((req_mask, tree_masks_3d[req_idx]), dim=1).flatten()
                    )
                    req_offset += req_size
                self.assertTrue(torch.equal(output, torch.cat(expected)))

    def test_pack_ragged_mask_and_cuda_graph_padding(self):
        draft_token_num = 3
        # Only request 0 has a real tree. Request 1 represents CUDA-graph
        # padding and must receive an all-visible tree suffix.
        compact = torch.tensor(
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
            ],
            dtype=torch.bool,
            device="cuda",
        ).flatten()
        seq_lens = torch.tensor([2, 4], dtype=torch.int32, device="cuda")
        kv_lens = seq_lens + draft_token_num
        kv_indptr = torch.zeros(3, dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)
        output = torch.empty((5 + 7) * draft_token_num, dtype=torch.bool, device="cuda")

        pack_ngram_full_mask(compact, kv_indptr, output, draft_token_num)

        expected = []
        compact_cpu = compact.view(draft_token_num, draft_token_num).cpu()
        for query_idx in range(draft_token_num):
            expected.extend([True, True])
            expected.extend(compact_cpu[query_idx].tolist())
        expected.extend([True] * (7 * draft_token_num))
        self.assertEqual(output.cpu().tolist(), expected)


if __name__ == "__main__":
    unittest.main()
