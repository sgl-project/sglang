import pytest
import torch
from sgl_kernel import process_accept_index_evict_mask_fused


def test_process_accept_index_evict_mask_fused():
    batch_size = 1
    spec_steps_plus_one = 3
    total_draft_tokens = 3

    accept_index = torch.tensor([[0, 1, -1]], dtype=torch.int32, device="cuda")
    predict = torch.tensor([12366, 11, 3967], dtype=torch.int32, device="cuda")

    accept_length = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    max_output_size = batch_size * spec_steps_plus_one
    verified_id = torch.empty(max_output_size, dtype=torch.int32, device="cuda")
    evict_mask = torch.empty(total_draft_tokens, dtype=torch.bool, device="cuda")
    filtered_accept_index = torch.empty(
        max_output_size, dtype=torch.int32, device="cuda"
    )
    output_size = torch.empty(1, dtype=torch.int32, device="cuda")

    process_accept_index_evict_mask_fused(
        accept_index,
        predict,
        accept_length,
        verified_id,
        evict_mask,
        filtered_accept_index,
        output_size,
    )

    assert (
        accept_length[0] == 1
    ), f"Expected accept_length[0] = 1, got {accept_length[0]}"
    assert output_size[0] == 2, f"Expected output_size = 2, got {output_size[0]}"

    expected_verified_id = torch.tensor([12366, 11], dtype=torch.int32, device="cuda")
    actual_verified_id = verified_id[: output_size.item()]
    assert torch.equal(
        actual_verified_id, expected_verified_id
    ), f"Expected verified_id = {expected_verified_id}, got {actual_verified_id}"

    expected_evict_mask = torch.tensor(
        [False, False, True], dtype=torch.bool, device="cuda"
    )
    assert torch.equal(
        evict_mask, expected_evict_mask
    ), f"Expected evict_mask = {expected_evict_mask}, got {evict_mask}"

    expected_filtered_accept_index = torch.tensor(
        [0, 1], dtype=torch.int32, device="cuda"
    )
    actual_filtered_accept_index = filtered_accept_index[: output_size.item()]
    assert torch.equal(
        actual_filtered_accept_index, expected_filtered_accept_index
    ), f"Expected filtered_accept_index = {expected_filtered_accept_index}, got {actual_filtered_accept_index}"

    print("Test passed!")


def test_process_accept_index_evict_mask_fused_multiple_batches():
    batch_size = 2
    spec_steps_plus_one = 3
    total_draft_tokens = 6

    accept_index = torch.tensor(
        [[0, 1, -1], [3, 4, 5]], dtype=torch.int32, device="cuda"
    )
    predict = torch.tensor(
        [100, 101, 102, 200, 201, 202], dtype=torch.int32, device="cuda"
    )

    accept_length = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    max_output_size = batch_size * spec_steps_plus_one
    verified_id = torch.empty(max_output_size, dtype=torch.int32, device="cuda")
    evict_mask = torch.empty(total_draft_tokens, dtype=torch.bool, device="cuda")
    filtered_accept_index = torch.empty(
        max_output_size, dtype=torch.int32, device="cuda"
    )
    output_size = torch.empty(1, dtype=torch.int32, device="cuda")

    process_accept_index_evict_mask_fused(
        accept_index,
        predict,
        accept_length,
        verified_id,
        evict_mask,
        filtered_accept_index,
        output_size,
    )

    expected_msg = f"Expected accept_length[0]=1, got {accept_length[0].item()}"
    assert accept_length[0].item() == 1, expected_msg
    expected_msg = f"Expected accept_length[1]=2, got {accept_length[1].item()}"
    assert accept_length[1].item() == 2, expected_msg
    expected_msg = f"Expected output_size=5, got {output_size[0].item()}"
    assert output_size[0].item() == 5, expected_msg

    expected_verified_id = [100, 101, 200, 201, 202]
    actual_verified_id = verified_id[: output_size[0].item()].tolist()
    expected_msg = (
        f"Expected verified_id={expected_verified_id}, " f"got {actual_verified_id}"
    )
    assert actual_verified_id == expected_verified_id, expected_msg

    expected_evict_mask = [False, False, True, False, False, False]
    actual_evict_mask = evict_mask.tolist()
    expected_msg = (
        f"Expected evict_mask={expected_evict_mask}, " f"got {actual_evict_mask}"
    )
    assert actual_evict_mask == expected_evict_mask, expected_msg


def test_process_accept_index_evict_mask_fused_all_rejected():
    batch_size = 1
    spec_steps_plus_one = 3
    total_draft_tokens = 3

    accept_index = torch.tensor([[-1, -1, -1]], dtype=torch.int32, device="cuda")
    predict = torch.tensor([100, 101, 102], dtype=torch.int32, device="cuda")

    accept_length = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    max_output_size = batch_size * spec_steps_plus_one
    verified_id = torch.empty(max_output_size, dtype=torch.int32, device="cuda")
    evict_mask = torch.empty(total_draft_tokens, dtype=torch.bool, device="cuda")
    filtered_accept_index = torch.empty(
        max_output_size, dtype=torch.int32, device="cuda"
    )
    output_size = torch.empty(1, dtype=torch.int32, device="cuda")

    process_accept_index_evict_mask_fused(
        accept_index,
        predict,
        accept_length,
        verified_id,
        evict_mask,
        filtered_accept_index,
        output_size,
    )

    expected_msg = f"Expected accept_length=-1, got {accept_length[0].item()}"
    assert accept_length[0].item() == -1, expected_msg
    expected_msg = f"Expected output_size=0, got {output_size[0].item()}"
    assert output_size[0].item() == 0, expected_msg

    expected_evict_mask = [True, True, True]
    actual_evict_mask = evict_mask.tolist()
    expected_msg = (
        f"Expected evict_mask={expected_evict_mask}, " f"got {actual_evict_mask}"
    )
    assert actual_evict_mask == expected_evict_mask, expected_msg


if __name__ == "__main__":
    pytest.main([__file__])
