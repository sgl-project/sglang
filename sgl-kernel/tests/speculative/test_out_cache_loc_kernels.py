import pytest
import torch
from sgl_kernel import process_out_cache_loc_with_masks_and_indices


def test_process_out_cache_loc_basic():
    device = "cuda"
    total_size = 10

    out_cache_loc = torch.arange(
        total_size, dtype=torch.int32, device=device
    ) * 10
    evict_mask = torch.tensor(
        [True, False, True, False, False, True, False, True, False, True],
        dtype=torch.bool,
        device=device,
    )
    accept_index = torch.tensor(
        [1, 3, 4, 6, 8], dtype=torch.int32, device=device
    )

    expected_evicted = torch.tensor(
        [0, 20, 50, 70, 90], dtype=torch.int32, device=device
    )
    expected_accepted = torch.tensor(
        [10, 30, 40, 60, 80], dtype=torch.int32, device=device
    )

    max_evicted = torch.sum(evict_mask).item()
    evicted_cache_loc = torch.empty(
        max_evicted, dtype=torch.int32, device=device
    )
    accepted_cache_loc = torch.empty(
        accept_index.shape[0], dtype=torch.int32, device=device
    )
    num_evicted = torch.empty(1, dtype=torch.int32, device=device)

    process_out_cache_loc_with_masks_and_indices(
        out_cache_loc,
        evict_mask,
        accept_index,
        evicted_cache_loc,
        accepted_cache_loc,
        num_evicted,
    )

    actual_num_evicted = num_evicted.item()
    expected_msg = (
        f"Expected num_evicted={max_evicted}, got {actual_num_evicted}"
    )
    assert actual_num_evicted == max_evicted, expected_msg
    
    actual_evicted = evicted_cache_loc[:actual_num_evicted]
    expected_msg = (
        f"Expected evicted={expected_evicted}, got {actual_evicted}"
    )
    assert torch.equal(actual_evicted, expected_evicted), expected_msg
    
    expected_msg = (
        f"Expected accepted={expected_accepted}, got {accepted_cache_loc}"
    )
    assert torch.equal(accepted_cache_loc, expected_accepted), expected_msg


def test_process_out_cache_loc_int64():
    device = "cuda"
    total_size = 6

    out_cache_loc = torch.arange(
        total_size, dtype=torch.int64, device=device
    ) * 100
    evict_mask = torch.tensor(
        [False, True, False, True, False, True],
        dtype=torch.bool,
        device=device,
    )
    accept_index = torch.tensor(
        [0, 2, 4], dtype=torch.int32, device=device
    )

    expected_evicted = torch.tensor(
        [100, 300, 500], dtype=torch.int64, device=device
    )
    expected_accepted = torch.tensor(
        [0, 200, 400], dtype=torch.int64, device=device
    )

    max_evicted = torch.sum(evict_mask).item()
    evicted_cache_loc = torch.empty(
        max_evicted, dtype=torch.int64, device=device
    )
    accepted_cache_loc = torch.empty(
        accept_index.shape[0], dtype=torch.int64, device=device
    )
    num_evicted = torch.empty(1, dtype=torch.int32, device=device)

    process_out_cache_loc_with_masks_and_indices(
        out_cache_loc,
        evict_mask,
        accept_index,
        evicted_cache_loc,
        accepted_cache_loc,
        num_evicted,
    )

    actual_num_evicted = num_evicted.item()
    expected_msg = (
        f"Expected num_evicted={max_evicted}, got {actual_num_evicted}"
    )
    assert actual_num_evicted == max_evicted, expected_msg
    
    actual_evicted = evicted_cache_loc[:actual_num_evicted]
    expected_msg = (
        f"Expected evicted={expected_evicted}, got {actual_evicted}"
    )
    assert torch.equal(actual_evicted, expected_evicted), expected_msg
    
    expected_msg = (
        f"Expected accepted={expected_accepted}, got {accepted_cache_loc}"
    )
    assert torch.equal(accepted_cache_loc, expected_accepted), expected_msg


def test_empty_masks():
    device = "cuda"
    total_size = 5

    out_cache_loc = torch.arange(
        total_size, dtype=torch.int32, device=device
    )
    evict_mask = torch.zeros(
        total_size, dtype=torch.bool, device=device
    )
    accept_index = torch.tensor(
        [0, 2, 4], dtype=torch.int32, device=device
    )

    evicted_cache_loc = torch.empty(
        total_size, dtype=torch.int32, device=device
    )
    accepted_cache_loc = torch.empty(
        accept_index.shape[0], dtype=torch.int32, device=device
    )
    num_evicted = torch.empty(1, dtype=torch.int32, device=device)

    process_out_cache_loc_with_masks_and_indices(
        out_cache_loc,
        evict_mask,
        accept_index,
        evicted_cache_loc,
        accepted_cache_loc,
        num_evicted,
    )

    actual_num_evicted = num_evicted.item()
    expected_msg = f"Expected num_evicted=0, got {actual_num_evicted}"
    assert actual_num_evicted == 0, expected_msg
    
    expected_accepted = torch.tensor(
        [0, 2, 4], dtype=torch.int32, device=device
    )
    expected_msg = (
        f"Expected accepted={expected_accepted}, got {accepted_cache_loc}"
    )
    assert torch.equal(accepted_cache_loc, expected_accepted), expected_msg


if __name__ == "__main__":
    pytest.main([__file__]) 
