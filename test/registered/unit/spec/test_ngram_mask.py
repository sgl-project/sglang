import unittest

import torch

from sglang.kernels.ops.speculative.tree_query_kv_layout import (
    build_tree_query_kv_layout,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestTreeQueryKVLayout(CustomTestCase):
    def test_build_layout_with_cuda_graph_padding(self):
        num_draft_tokens = 3
        tree_mask = torch.tensor(
            [
                [[1, 0, 0], [1, 1, 0], [1, 0, 1]],
                [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
            ],
            dtype=torch.bool,
            device="cuda",
        ).flatten()
        req_to_token = torch.zeros((3, 16), dtype=torch.int32, device="cuda")
        req_to_token[1, 2:5] = torch.tensor([102, 103, 104], device="cuda")
        req_to_token[2, 4:7] = torch.tensor([204, 205, 206], device="cuda")
        req_pool_indices = torch.tensor([1, 2, 0], device="cuda")
        seq_lens = torch.tensor([2, 4, 1], device="cuda")
        kv_slots = torch.full(
            (9, num_draft_tokens), -1, dtype=torch.int32, device="cuda"
        )
        kv_lens = torch.empty(9, dtype=torch.int32, device="cuda")

        build_tree_query_kv_layout(
            tree_mask,
            req_to_token,
            req_pool_indices,
            seq_lens,
            kv_slots,
            kv_lens,
            num_draft_tokens,
        )

        self.assertEqual(kv_lens.cpu().tolist(), [1, 2, 2, 1, 2, 3, 1, 1, 1])
        self.assertEqual(
            kv_slots[:, :1].cpu().flatten().tolist(),
            [102, 102, 102, 204, 204, 204, 0, 0, 0],
        )
        self.assertEqual(kv_slots[1, :2].cpu().tolist(), [102, 103])
        self.assertEqual(kv_slots[2, :2].cpu().tolist(), [102, 104])
        self.assertEqual(kv_slots[4, :2].cpu().tolist(), [204, 205])
        self.assertEqual(kv_slots[5, :3].cpu().tolist(), [204, 205, 206])


if __name__ == "__main__":
    unittest.main()
