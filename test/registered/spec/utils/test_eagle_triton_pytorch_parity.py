import unittest

import torch

from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    sgl_build_tree_kernel_efficient_pytorch,
    sgl_build_tree_kernel_triton,
    verify_tree_greedy_pytorch,
    verify_tree_greedy_triton,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=8, stage="stage-b", runner_config="1-gpu-xpu")


class TestEagleTritonPytorchParity(CustomTestCase):
    def setUp(self):
        device_str = get_device()
        if device_str == "cpu":
            self.skipTest("This test requires a GPU device for Triton kernels.")
        self.device = torch.device(device_str)

    def _build_kernel_inputs(self):
        topk = 4
        depth = 4
        draft_token_num = 8

        verified_seq_len = torch.tensor([5, 7], device=self.device, dtype=torch.long)
        # Use a root-only topology (all parent_tb_idx == 0) to ensure
        # deterministic parity for direct low-level kernel comparisons.
        selected_index = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            device=self.device,
            dtype=torch.long,
        )
        parent_list = torch.tensor(
            [
                [-1],
                [-1],
            ],
            device=self.device,
            dtype=torch.long,
        )

        return {
            "topk": topk,
            "depth": depth,
            "draft_token_num": draft_token_num,
            "verified_seq_len": verified_seq_len,
            "selected_index": selected_index,
            "parent_list": parent_list,
        }

    def _build_kernel_inputs_nontrivial(self):
        topk = 4
        depth = 1
        draft_token_num = 8

        verified_seq_len = torch.tensor([6, 9], device=self.device, dtype=torch.long)
        # Nontrivial topology: fanout under one internal node.
        # This exercises non-root parent resolution while keeping tree-mask
        # traversal simple and deterministic at depth=1.
        selected_index = torch.tensor(
            [
                [0, 4, 4, 4, 4, 4, 4, 4],
                [0, 4, 4, 4, 4, 4, 4, 4],
            ],
            device=self.device,
            dtype=torch.long,
        )
        parent_list = torch.tensor(
            [
                [-1, 0],
                [-1, 0],
            ],
            device=self.device,
            dtype=torch.long,
        )

        return {
            "topk": topk,
            "depth": depth,
            "draft_token_num": draft_token_num,
            "verified_seq_len": verified_seq_len,
            "selected_index": selected_index,
            "parent_list": parent_list,
        }

    def _build_kernel_inputs_high_depth(self):
        topk = 4
        depth = 8
        draft_token_num = 8

        verified_seq_len = torch.tensor([8, 11], device=self.device, dtype=torch.long)
        # High-depth chain topology to stress parent traversal and tree-mask path
        # construction without sibling-order ambiguity.
        selected_index = torch.tensor(
            [
                [0, 4, 8, 12, 16, 20, 24, 28],
                [0, 4, 8, 12, 16, 20, 24, 28],
            ],
            device=self.device,
            dtype=torch.long,
        )
        parent_list = torch.tensor(
            [
                [-1, 0, 4, 8, 12, 16, 20, 24],
                [-1, 0, 4, 8, 12, 16, 20, 24],
            ],
            device=self.device,
            dtype=torch.long,
        )

        return {
            "topk": topk,
            "depth": depth,
            "draft_token_num": draft_token_num,
            "verified_seq_len": verified_seq_len,
            "selected_index": selected_index,
            "parent_list": parent_list,
        }

    def _build_input_cases(self):
        return [
            ("baseline_root_only", self._build_kernel_inputs()),
            ("nontrivial_fanout", self._build_kernel_inputs_nontrivial()),
            ("high_depth_chain", self._build_kernel_inputs_high_depth()),
        ]

    def _alloc_build_outputs(self, batch_size, draft_token_num, seq_lens_sum):
        tree_mask = torch.full(
            (
                seq_lens_sum * draft_token_num
                + batch_size * draft_token_num * draft_token_num,
            ),
            True,
            device=self.device,
            dtype=torch.bool,
        )
        positions = torch.empty(
            (batch_size * draft_token_num,),
            device=self.device,
            dtype=torch.long,
        )
        retrieve_buf = torch.full(
            (3, batch_size, draft_token_num),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
        return (
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
        )

    def _make_verify_target(
        self,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        num_spec_tokens,
    ):
        # Build a deterministic target that is independent of sibling linked-list order.
        target_predict = torch.full_like(candidates, -1)
        batch_size, num_draft_tokens = candidates.shape

        for bx in range(batch_size):
            cur_index = 0
            last_accepted_retrieve_idx = retrieve_index[bx, 0].item()

            for _ in range(1, num_spec_tokens):
                child_indices = self._children_for_parent(
                    retrieve_next_token,
                    retrieve_next_sibling,
                    bx,
                    cur_index,
                    num_draft_tokens,
                )
                if len(child_indices) == 0:
                    break

                # Pick smallest token id so both kernels follow the same path even
                # when sibling list insertion order differs.
                next_index = min(
                    child_indices,
                    key=lambda idx: candidates[bx, idx].item(),
                )

                target_row = last_accepted_retrieve_idx // num_draft_tokens
                target_col = last_accepted_retrieve_idx % num_draft_tokens
                target_predict[target_row, target_col] = candidates[bx, next_index]

                cur_index = next_index
                last_accepted_retrieve_idx = retrieve_index[bx, cur_index].item()

            target_row = last_accepted_retrieve_idx // num_draft_tokens
            target_col = last_accepted_retrieve_idx % num_draft_tokens
            if target_predict[target_row, target_col].item() == -1:
                target_predict[target_row, target_col] = candidates[bx, cur_index]

        return target_predict

    @staticmethod
    def _children_for_parent(
        retrieve_next_token,
        retrieve_next_sibling,
        batch_idx,
        parent_idx,
        num_draft_tokens,
    ):
        children = []
        child = retrieve_next_token[batch_idx, parent_idx].item()
        visited = set()
        steps = 0

        while child != -1 and steps < num_draft_tokens:
            if child in visited:
                break
            visited.add(child)
            children.append(child)
            child = retrieve_next_sibling[batch_idx, child].item()
            steps += 1

        return children

    @staticmethod
    def _canonical_children_map(
        retrieve_next_token,
        retrieve_next_sibling,
        batch_size,
        num_draft_tokens,
    ):
        canonical = []
        for bx in range(batch_size):
            per_batch = []
            for parent in range(num_draft_tokens):
                children = TestEagleTritonPytorchParity._children_for_parent(
                    retrieve_next_token,
                    retrieve_next_sibling,
                    bx,
                    parent,
                    num_draft_tokens,
                )
                per_batch.append(tuple(sorted(children)))
            canonical.append(tuple(per_batch))
        return tuple(canonical)

    def _assert_tensor_equal(self, lhs, rhs, name):
        if torch.equal(lhs, rhs):
            return

        mismatch = (lhs != rhs).nonzero(as_tuple=False)
        first_idx = tuple(mismatch[0].tolist()) if mismatch.numel() > 0 else "unknown"
        lhs_v = lhs[first_idx].item() if mismatch.numel() > 0 else "n/a"
        rhs_v = rhs[first_idx].item() if mismatch.numel() > 0 else "n/a"
        self.fail(f"{name} mismatch at {first_idx}: pytorch={lhs_v}, triton={rhs_v}")

    def _assert_tree_links_equivalent(
        self,
        retrieve_next_token_pt,
        retrieve_next_sibling_pt,
        retrieve_next_token_tr,
        retrieve_next_sibling_tr,
        batch_size,
        num_draft_tokens,
    ):
        canonical_pt = self._canonical_children_map(
            retrieve_next_token_pt,
            retrieve_next_sibling_pt,
            batch_size,
            num_draft_tokens,
        )
        canonical_tr = self._canonical_children_map(
            retrieve_next_token_tr,
            retrieve_next_sibling_tr,
            batch_size,
            num_draft_tokens,
        )
        self.assertEqual(canonical_pt, canonical_tr)

    def test_build_tree_triton_matches_pytorch(self):
        for case_name, inputs in self._build_input_cases():
            with self.subTest(case=case_name):
                batch_size = inputs["verified_seq_len"].shape[0]
                draft_token_num = inputs["draft_token_num"]
                seq_lens_sum = int(inputs["verified_seq_len"].sum().item())

                (
                    tree_mask_pt,
                    positions_pt,
                    retrieve_index_pt,
                    retrieve_next_token_pt,
                    retrieve_next_sibling_pt,
                ) = self._alloc_build_outputs(batch_size, draft_token_num, seq_lens_sum)

                (
                    tree_mask_tr,
                    positions_tr,
                    retrieve_index_tr,
                    retrieve_next_token_tr,
                    retrieve_next_sibling_tr,
                ) = self._alloc_build_outputs(batch_size, draft_token_num, seq_lens_sum)

                sgl_build_tree_kernel_efficient_pytorch(
                    inputs["parent_list"],
                    inputs["selected_index"],
                    inputs["verified_seq_len"],
                    tree_mask_pt,
                    positions_pt,
                    retrieve_index_pt,
                    retrieve_next_token_pt,
                    retrieve_next_sibling_pt,
                    topk=inputs["topk"],
                    depth=inputs["depth"],
                    draft_token_num=draft_token_num,
                    tree_mask_mode=TreeMaskMode.FULL_MASK,
                )

                sgl_build_tree_kernel_triton(
                    inputs["parent_list"],
                    inputs["selected_index"],
                    inputs["verified_seq_len"],
                    tree_mask_tr,
                    positions_tr,
                    retrieve_index_tr,
                    retrieve_next_token_tr,
                    retrieve_next_sibling_tr,
                    topk=inputs["topk"],
                    depth=inputs["depth"],
                    draft_token_num=draft_token_num,
                    tree_mask_mode=TreeMaskMode.FULL_MASK,
                )

                self._assert_tensor_equal(tree_mask_pt, tree_mask_tr, "tree_mask")
                self._assert_tensor_equal(positions_pt, positions_tr, "positions")
                self._assert_tensor_equal(
                    retrieve_index_pt,
                    retrieve_index_tr,
                    "retrieve_index",
                )
                self._assert_tree_links_equivalent(
                    retrieve_next_token_pt,
                    retrieve_next_sibling_pt,
                    retrieve_next_token_tr,
                    retrieve_next_sibling_tr,
                    batch_size,
                    draft_token_num,
                )

    def test_verify_tree_triton_matches_pytorch(self):
        for case_name, inputs in self._build_input_cases():
            with self.subTest(case=case_name):
                batch_size = inputs["verified_seq_len"].shape[0]
                draft_token_num = inputs["draft_token_num"]
                seq_lens_sum = int(inputs["verified_seq_len"].sum().item())

                (
                    _tree_mask,
                    _positions,
                    retrieve_index,
                    retrieve_next_token,
                    retrieve_next_sibling,
                ) = self._alloc_build_outputs(batch_size, draft_token_num, seq_lens_sum)

                sgl_build_tree_kernel_efficient_pytorch(
                    inputs["parent_list"],
                    inputs["selected_index"],
                    inputs["verified_seq_len"],
                    _tree_mask,
                    _positions,
                    retrieve_index,
                    retrieve_next_token,
                    retrieve_next_sibling,
                    topk=inputs["topk"],
                    depth=inputs["depth"],
                    draft_token_num=draft_token_num,
                    tree_mask_mode=TreeMaskMode.FULL_MASK,
                )

                candidates = torch.tensor(
                    [
                        [100, 101, 102, 103, 104, 105, 106, 107],
                        [200, 201, 202, 203, 204, 205, 206, 207],
                    ],
                    device=self.device,
                    dtype=torch.long,
                )
                num_speculative_tokens = draft_token_num
                target_predict = self._make_verify_target(
                    candidates,
                    retrieve_index,
                    retrieve_next_token,
                    retrieve_next_sibling,
                    num_speculative_tokens,
                )

                predicts_pt = torch.full(
                    (batch_size * draft_token_num,),
                    -1,
                    device=self.device,
                    dtype=torch.long,
                )
                accept_index_pt = torch.full(
                    (batch_size, num_speculative_tokens),
                    -1,
                    device=self.device,
                    dtype=torch.long,
                )
                accept_token_num_pt = torch.zeros(
                    (batch_size,),
                    device=self.device,
                    dtype=torch.long,
                )

                predicts_tr = predicts_pt.clone()
                accept_index_tr = accept_index_pt.clone()
                accept_token_num_tr = accept_token_num_pt.clone()

                verify_tree_greedy_pytorch(
                    predicts_pt,
                    accept_index_pt,
                    accept_token_num_pt,
                    candidates,
                    retrieve_index,
                    retrieve_next_token,
                    retrieve_next_sibling,
                    target_predict,
                )

                verify_tree_greedy_triton(
                    predicts_tr,
                    accept_index_tr,
                    accept_token_num_tr,
                    candidates,
                    retrieve_index,
                    retrieve_next_token,
                    retrieve_next_sibling,
                    target_predict,
                )

                self._assert_tensor_equal(predicts_pt, predicts_tr, "predicts")
                self._assert_tensor_equal(
                    accept_index_pt,
                    accept_index_tr,
                    "accept_index",
                )
                self._assert_tensor_equal(
                    accept_token_num_pt,
                    accept_token_num_tr,
                    "accept_token_num",
                )


if __name__ == "__main__":
    unittest.main()
