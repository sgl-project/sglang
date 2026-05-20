import pytest
import torch

from sglang.srt.layers.logits_processor import _get_chunk_logprob_indices
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    (
        "input_logprob_indices",
        "start_idx",
        "end_idx",
        "expected_mask",
        "expected_chunk",
    ),
    [
        ([0, 1, 2, 3, 5, 6, 7], 0, 3, [0, 1, 2], [0, 1, 2]),
        ([0, 1, 2, 3, 5, 6, 7], 3, 6, [3, 4], [0, 2]),
        ([0, 1, 2, 3, 5, 6, 7], 6, 9, [5, 6], [0, 1]),
        ([0, 2, 5], 3, 5, [], []),
        ([], 0, 4, [], []),
    ],
)
def test_get_chunk_logprob_indices_matches_nonzero_selection(
    input_logprob_indices, start_idx, end_idx, expected_mask, expected_chunk
):
    input_logprob_indices = torch.tensor(input_logprob_indices, dtype=torch.int64)

    mask_indices, global_indices, chunk_indices = _get_chunk_logprob_indices(
        input_logprob_indices, start_idx, end_idx
    )

    assert mask_indices.tolist() == expected_mask
    assert global_indices.tolist() == [
        input_logprob_indices[index].item() for index in expected_mask
    ]
    assert chunk_indices.tolist() == expected_chunk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_get_chunk_logprob_indices_preserves_cuda_device():
    input_logprob_indices = torch.tensor(
        [0, 1, 2, 3, 5, 6, 7], dtype=torch.int64, device="cuda"
    )

    mask_indices, global_indices, chunk_indices = _get_chunk_logprob_indices(
        input_logprob_indices, 3, 6
    )

    assert mask_indices.device.type == "cuda"
    assert global_indices.device.type == "cuda"
    assert chunk_indices.device.type == "cuda"
    assert mask_indices.cpu().tolist() == [3, 4]
    assert global_indices.cpu().tolist() == [3, 5]
    assert chunk_indices.cpu().tolist() == [0, 2]
