import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import reconstruct_indices_from_tree_mask


def test_reconstruct_indices_from_tree_mask():
    bs = 1
    num_branch_token = 4
    seq_lens = torch.tensor([12], device="cuda", dtype=torch.int64)

    retrive_index = torch.full(
        (bs, num_branch_token), -1, device="cuda", dtype=torch.int64
    )
    retrive_next_token = torch.full(
        (bs, num_branch_token), -1, device="cuda", dtype=torch.int64
    )
    retrive_next_sibling = torch.full(
        (bs, num_branch_token), -1, device="cuda", dtype=torch.int64
    )
    positions = torch.empty((bs * num_branch_token), device="cuda", dtype=torch.int64)

    tree_mask = torch.tensor(
        [
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
        ],
        device="cuda",
        dtype=torch.int32,
    ).to(torch.bool)

    reconstruct_indices_from_tree_mask(
        tree_mask,
        seq_lens,
        positions,  # mutable
        retrive_index,  # mutable
        retrive_next_token,  # mutable
        retrive_next_sibling,  # mutable
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
        12,
        13,
        13,
        14,
    ], f"{positions=}"


if __name__ == "__main__":
    test_reconstruct_indices_from_tree_mask()
    pytest.main([__file__])
