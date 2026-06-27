import sys

import pytest
import torch

from sglang.srt.lora.backend.base_backend import _compute_moe_lora_info
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _expected_adapter_enabled(
    lora_ranks: torch.Tensor,
    weight_indices: torch.Tensor,
) -> torch.Tensor:
    expected = torch.zeros_like(lora_ranks)
    valid_weight_indices = weight_indices[weight_indices >= 0]
    if valid_weight_indices.numel() != 0:
        expected.scatter_(
            0,
            valid_weight_indices.long(),
            (lora_ranks[valid_weight_indices.long()] > 0).to(torch.int32),
        )
    return expected


def _expected_token_lora_mapping(
    lora_ranks: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_lens: torch.Tensor,
) -> torch.Tensor:
    valid_weight_indices = weight_indices >= 0
    safe_weight_indices = torch.where(valid_weight_indices, weight_indices, 0)
    active_ranks = torch.where(
        valid_weight_indices,
        lora_ranks[safe_weight_indices.long()],
        0,
    )
    active_weight_indices = torch.where(
        valid_weight_indices & (active_ranks > 0),
        weight_indices,
        -1,
    )
    return torch.repeat_interleave(active_weight_indices, seg_lens)


@pytest.mark.parametrize("use_preallocated_buffers", [False, True])
def test_compute_moe_lora_info_expands_segments(use_preallocated_buffers: bool):
    device = "cuda"
    seg_lens = torch.tensor([5, 1, 7, 3, 9, 2], dtype=torch.int32, device=device)
    seg_indptr = torch.zeros((seg_lens.numel() + 1,), dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

    weight_indices = torch.tensor([2, 0, 5, 2, 3, 7], dtype=torch.int32, device=device)
    lora_ranks = torch.tensor(
        [0, 12, 16, 32, 24, 8, 0, 4], dtype=torch.int32, device=device
    )
    num_tokens = int(seg_indptr[-1].item())

    if use_preallocated_buffers:
        adapter_enabled = torch.full_like(lora_ranks, 123)
        token_lora_mapping = torch.full(
            (num_tokens + 11,), 456, dtype=torch.int32, device=device
        )
    else:
        adapter_enabled = None
        token_lora_mapping = None

    actual_enabled, actual_mapping = _compute_moe_lora_info(
        num_tokens,
        seg_indptr,
        lora_ranks,
        weight_indices,
        adapter_enabled,
        token_lora_mapping,
        max_len=int(seg_lens.max().item()),
    )
    torch.cuda.synchronize()

    expected_mapping = _expected_token_lora_mapping(
        lora_ranks, weight_indices, seg_lens
    )
    expected_enabled = _expected_adapter_enabled(lora_ranks, weight_indices)

    torch.testing.assert_close(actual_mapping, expected_mapping)
    torch.testing.assert_close(actual_enabled, expected_enabled)

    if use_preallocated_buffers:
        assert actual_mapping.data_ptr() == token_lora_mapping.data_ptr()
        torch.testing.assert_close(
            token_lora_mapping[num_tokens:],
            torch.full_like(token_lora_mapping[num_tokens:], -1),
        )


@pytest.mark.parametrize("use_preallocated_buffers", [False, True])
def test_compute_moe_lora_info_maps_rank_zero_segments_to_no_lora(
    use_preallocated_buffers: bool,
):
    device = "cuda"
    seg_lens = torch.tensor([2, 1, 3], dtype=torch.int32, device=device)
    seg_indptr = torch.zeros((seg_lens.numel() + 1,), dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

    weight_indices = torch.tensor([0, 1, 0], dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([0, 16], dtype=torch.int32, device=device)
    num_tokens = int(seg_indptr[-1].item())

    if use_preallocated_buffers:
        adapter_enabled = torch.full_like(lora_ranks, 123)
        token_lora_mapping = torch.full(
            (num_tokens + 5,), 456, dtype=torch.int32, device=device
        )
    else:
        adapter_enabled = None
        token_lora_mapping = None

    actual_enabled, actual_mapping = _compute_moe_lora_info(
        num_tokens,
        seg_indptr,
        lora_ranks,
        weight_indices,
        adapter_enabled,
        token_lora_mapping,
        max_len=int(seg_lens.max().item()),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_mapping,
        torch.tensor([-1, -1, 1, -1, -1, -1], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        actual_enabled, torch.tensor([0, 1], dtype=torch.int32, device=device)
    )
    if use_preallocated_buffers:
        torch.testing.assert_close(
            token_lora_mapping[num_tokens:],
            torch.full_like(token_lora_mapping[num_tokens:], -1),
        )


@pytest.mark.parametrize("use_preallocated_buffers", [False, True])
def test_compute_moe_lora_info_preserves_negative_no_lora_segments(
    use_preallocated_buffers: bool,
):
    device = "cuda"
    seg_lens = torch.tensor([1, 2, 1], dtype=torch.int32, device=device)
    seg_indptr = torch.zeros((seg_lens.numel() + 1,), dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

    weight_indices = torch.tensor([-1, 1, 0], dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([16, 8], dtype=torch.int32, device=device)
    num_tokens = int(seg_indptr[-1].item())

    if use_preallocated_buffers:
        adapter_enabled = torch.full_like(lora_ranks, 123)
        token_lora_mapping = torch.full(
            (num_tokens + 3,), 456, dtype=torch.int32, device=device
        )
    else:
        adapter_enabled = None
        token_lora_mapping = None

    actual_enabled, actual_mapping = _compute_moe_lora_info(
        num_tokens,
        seg_indptr,
        lora_ranks,
        weight_indices,
        adapter_enabled,
        token_lora_mapping,
        max_len=int(seg_lens.max().item()),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_mapping,
        torch.tensor([-1, 1, 1, 0], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        actual_enabled, torch.tensor([1, 1], dtype=torch.int32, device=device)
    )
    if use_preallocated_buffers:
        torch.testing.assert_close(
            token_lora_mapping[num_tokens:],
            torch.full_like(token_lora_mapping[num_tokens:], -1),
        )


def test_compute_moe_lora_info_rejects_undercovered_launch():
    device = "cuda"
    seg_indptr = torch.tensor([0, 300], dtype=torch.int32, device=device)
    weight_indices = torch.tensor([0], dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([16], dtype=torch.int32, device=device)

    with pytest.raises(AssertionError, match="under-covers tokens"):
        _compute_moe_lora_info(
            300,
            seg_indptr,
            lora_ranks,
            weight_indices,
            None,
            None,
            max_len=1,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
