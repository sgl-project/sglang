import sys

import pytest
import torch

from sglang.srt.lora.backend.base_backend import _compute_moe_lora_info
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")


def _expected_adapter_enabled(
    lora_ranks: torch.Tensor,
    weight_indices: torch.Tensor,
) -> torch.Tensor:
    expected = torch.zeros_like(lora_ranks)
    expected.scatter_(
        0,
        weight_indices.long(),
        (lora_ranks[weight_indices.long()] > 0).to(torch.int32),
    )
    return expected


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

    raw_mapping = torch.repeat_interleave(weight_indices, seg_lens)
    expected_mapping = torch.where(
        lora_ranks[raw_mapping.long()] > 0,
        raw_mapping,
        -1,
    )
    expected_enabled = _expected_adapter_enabled(lora_ranks, weight_indices)

    torch.testing.assert_close(actual_mapping, expected_mapping)
    torch.testing.assert_close(actual_enabled, expected_enabled)

    if use_preallocated_buffers:
        assert actual_mapping.data_ptr() == token_lora_mapping.data_ptr()
        assert token_lora_mapping[num_tokens:].tolist() == [-1] * 11


@pytest.mark.parametrize(
    ("weight_indices", "lora_ranks", "expected_mapping", "expected_enabled"),
    [
        ([0, 1, 2], [0, 8, 0], [-1, -1, 1, 1, 1, -1], [0, 1, 0]),
        ([0, 2, 0], [0, 8, 0], [-1, -1, -1, -1, -1, -1], [0, 0, 0]),
    ],
    ids=("mixed", "all_inactive"),
)
def test_compute_moe_lora_info_cpu_fallback_canonicalizes_inactive_slots(
    weight_indices: list[int],
    lora_ranks: list[int],
    expected_mapping: list[int],
    expected_enabled: list[int],
):
    seg_lens = torch.tensor([2, 3, 1], dtype=torch.int32)
    seg_indptr = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    weight_indices_tensor = torch.tensor(weight_indices, dtype=torch.int32)
    lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32)
    adapter_enabled = torch.full((3,), 123, dtype=torch.int32)
    mapping_buffer = torch.full((9,), 456, dtype=torch.int32)

    actual_enabled, actual_mapping = _compute_moe_lora_info(
        6,
        seg_indptr,
        lora_ranks_tensor,
        weight_indices_tensor,
        adapter_enabled,
        mapping_buffer,
        max_len=int(seg_lens.max().item()),
    )

    assert actual_mapping.data_ptr() == mapping_buffer.data_ptr()
    assert actual_mapping.tolist() == expected_mapping
    assert actual_enabled.tolist() == expected_enabled
    assert mapping_buffer[6:].tolist() == [-1, -1, -1]


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
