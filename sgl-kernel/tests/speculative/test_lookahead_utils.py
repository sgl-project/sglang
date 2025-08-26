import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import (
    reconstruct_indices_from_tree_mask,
    lookahead_verify_tree_greedy,
)


def test_reconstruct_indices_from_tree_mask():
    bs = 1
    num_branch_token = 4
    seq_lens = torch.tensor([12], device="cuda", dtype=torch.int32)

    retrive_index = torch.full(
        (bs, num_branch_token), -1, device="cuda", dtype=torch.int32
    )
    retrive_next_token = torch.full(
        (bs, num_branch_token), -1, device="cuda", dtype=torch.int32
    )
    retrive_next_sibling = torch.full(
        (bs, num_branch_token), -1, device="cuda", dtype=torch.int32
    )
    positions = torch.empty((bs * num_branch_token), device="cuda", dtype=torch.int32)

    tree_mask = torch.tensor([
        1, 0, 0, 0,
        1, 1, 0, 0,
        1, 0, 1, 0,
        1, 0, 1, 1,
    ], device="cuda", dtype=torch.int32).to(torch.bool)

    reconstruct_indices_from_tree_mask(
        tree_mask,
        seq_lens,
        positions,            # mutable
        retrive_index,        # mutable
        retrive_next_token,   # mutable
        retrive_next_sibling, # mutable
        bs,
        num_branch_token,
    )
    # print(f"debug: \n\n{tree_mask=}, {retrive_index=}, {retrive_next_token=}, {retrive_next_sibling=}, {positions=}\n\n")
    assert retrive_index.tolist() == [
        [0, 1, 2, 3],
    ], f"{retrive_index=}"
    assert retrive_next_token.tolist() == [
        [1, -1, 3, -1],
    ], f"{retrive_next_token=}"
    assert retrive_next_sibling.tolist() == [
        [-1, 2, -1, -1],
    ], f"{retrive_next_sibling=}"
    assert positions.tolist() == [
        12, 13, 13, 14,
    ], f"{positions=}"


def test_lookahead_verify_tree_greedy():
    bs = 1
    num_branch_token = 4
    accept_token_num = torch.full((bs,), 0, device="cuda", dtype=torch.int32)
    accept_token_ids = torch.full((bs, num_branch_token), -1, device="cuda", dtype=torch.int32)
    last_verified_ids = torch.full((bs,), -1, device="cuda", dtype=torch.int32)
    flatten_index = torch.full((bs * num_branch_token,), -1, device="cuda", dtype=torch.int32)
    total_accept_num = torch.full((1,), 0, device="cuda", dtype=torch.int32)

    candidates = torch.tensor([14, 56007, 100, 108198], device="cuda", dtype=torch.int32)
    target_predict = torch.tensor([56007, 100, 108198, 32108], device="cuda", dtype=torch.int32)

    retrive_index = torch.tensor([[0, 1, 2, 3]], device="cuda", dtype=torch.int32)
    retrive_next_token = torch.tensor([[1, 2, 3, -1]], device="cuda", dtype=torch.int32)
    retrive_next_sibling = torch.tensor([[-1, -1, -1, -1]], device="cuda", dtype=torch.int32)


    eos_token_id = 100
    lookahead_verify_tree_greedy(
        accept_token_num, # mutable
        accept_token_ids, # mutable
        last_verified_ids, # mutable
        flatten_index, # mutable
        total_accept_num, # mutable
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        eos_token_id,
    )

    assert accept_token_num.tolist() == [2]
    assert accept_token_ids.tolist() == [
        [56007, 100, -1, -1],
    ]
    assert last_verified_ids.tolist() == [
        100,
    ]
    assert flatten_index.tolist() == [
        0, 1, 2, 3,
    ]
    assert total_accept_num.tolist() == [
        2,
    ]

if __name__ == "__main__":
    test_reconstruct_indices_from_tree_mask()
    test_lookahead_verify_tree_greedy()
    pytest.main([__file__])
