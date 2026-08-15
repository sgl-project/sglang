from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.gemm.batch_matmul_transpose_npu import (
    batch_matmul_transpose_npu,
)
from sglang.kernels.ops.speculative.cache_locs_npu import (
    read_cache_locations_npu,
)
from sglang.kernels.ops.speculative.npu_reference import (
    build_full_tree_mask_reference,
    build_retrieval_links_reference,
    read_cache_locations_reference,
)
from sglang.kernels.ops.speculative.spec_tree_npu import build_full_tree_npu
from sglang.srt.hardware_backend.npu.utils import is_ascend_a5
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=20, suite="nightly-1-npu-a3", nightly=True)


@unittest.skipUnless(is_ascend_a5(), "requires an Ascend A5 device")
class TestNpuA5SpecKernels(CustomTestCase):
    def test_cache_locations_beyond_six_blocks(self):
        token_pool = torch.arange(3 * 512, device="npu", dtype=torch.long).view(3, 512)
        req_pool_indices = torch.tensor([1, 0], device="npu")
        start_offset = torch.tensor([7, 3], device="npu")
        end_offset = torch.tensor([264, 214], device="npu")
        expected = read_cache_locations_reference(
            req_pool_indices=req_pool_indices,
            token_pool=token_pool,
            start_offset=start_offset,
            end_offset=end_offset,
        )
        actual = torch.empty_like(expected)

        read_cache_locations_npu(
            req_pool_indices=req_pool_indices,
            token_pool=token_pool,
            start_offset=start_offset,
            end_offset=end_offset,
            out_cache_loc=actual,
        )

        self.assertTrue(torch.equal(actual.cpu(), expected.cpu()))

    def test_tree_links_match_reference_for_topk_two(self):
        parent_list = torch.tensor([[0, 0]], device="npu")
        selected_index = torch.tensor([[0, 1, 2]], device="npu")
        seq_lens = torch.tensor([5], device="npu", dtype=torch.long)
        draft_token_num = 4
        tree_mask = torch.ones(
            (int(seq_lens.sum()) * draft_token_num + draft_token_num**2,),
            device="npu",
            dtype=torch.bool,
        )
        positions = torch.empty((draft_token_num,), device="npu", dtype=torch.long)
        retrieve_buf = torch.full(
            (3, 1, draft_token_num), -1, device="npu", dtype=torch.long
        )

        build_full_tree_npu(
            parent_list=parent_list,
            selected_index=selected_index,
            verified_seq_len=seq_lens,
            tree_mask=tree_mask,
            positions=positions,
            retrieve_index=retrieve_buf[0],
            retrieve_next_token=retrieve_buf[1],
            retrieve_next_sibling=retrieve_buf[2],
            topk=2,
            draft_token_num=draft_token_num,
        )
        expected = build_retrieval_links_reference(
            parent_list=parent_list,
            selected_index=selected_index,
            topk=2,
            draft_token_num=draft_token_num,
        )

        for actual, reference in zip(retrieve_buf, expected):
            self.assertTrue(torch.equal(actual.cpu(), reference.cpu()))

        expected_mask, expected_positions = build_full_tree_mask_reference(
            parent_list=parent_list,
            selected_index=selected_index,
            verified_seq_len=seq_lens,
            topk=2,
            draft_token_num=draft_token_num,
        )
        self.assertTrue(torch.equal(tree_mask.cpu(), expected_mask.cpu()))
        self.assertTrue(torch.equal(positions.cpu(), expected_positions.cpu()))

    def test_batch_matmul_transpose_matches_torch(self):
        lhs = torch.randn((2, 3, 64), device="npu", dtype=torch.bfloat16)
        rhs = torch.randn((3, 64, 128), device="npu", dtype=torch.bfloat16)
        actual = torch.empty((2, 3, 128), device="npu", dtype=torch.bfloat16)
        batch_matmul_transpose_npu(
            tensor_a=lhs,
            tensor_b=rhs,
            tensor_c=actual,
        )
        expected = torch.einsum("bmk,mkn->bmn", lhs, rhs)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.02, atol=0.02)

    def test_batch_one_k64_uses_masked_generic_kernel(self):
        lhs = torch.randn((1, 3, 64), device="npu", dtype=torch.bfloat16)
        rhs = torch.randn((3, 64, 128), device="npu", dtype=torch.bfloat16)
        actual = torch.empty((1, 3, 128), device="npu", dtype=torch.bfloat16)
        batch_matmul_transpose_npu(
            tensor_a=lhs,
            tensor_b=rhs,
            tensor_c=actual,
        )
        expected = torch.einsum("bmk,mkn->bmn", lhs, rhs)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.02, atol=0.02)


if __name__ == "__main__":
    unittest.main()
