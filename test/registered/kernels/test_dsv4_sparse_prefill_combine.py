import pytest
import torch

from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    combine_topk_swa_indices,
    combined_topk_width,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, suite="nightly-1-gpu", nightly=True)


def test_combine_topk_swa_indices_uses_sparse_prefill_positions():
    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA")

    device = "cuda"
    topk = 4
    window_size = 4
    topk_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.int32,
        device=device,
    )
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([12], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([9], dtype=torch.int32, device=device)
    compressed_base = torch.tensor([100], dtype=torch.int32, device=device)
    swa_base = torch.tensor([200], dtype=torch.int32, device=device)
    positions = torch.tensor([7, 9, 11], dtype=torch.int32, device=device)

    combined_indices, combined_lens = combine_topk_swa_indices(
        topk_indices=topk_indices,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        compressed_base=compressed_base,
        swa_base=swa_base,
        window_size=window_size,
        compress_ratio=4,
        topk=topk,
        positions=positions,
    )
    torch.cuda.synchronize()

    expected = torch.full(
        (3, combined_topk_width(topk, window_size)),
        -1,
        dtype=torch.int32,
        device=device,
    )
    expected[0, :6] = torch.tensor([100, 101, 201, 202, 203, 204], device=device)
    expected[1, :6] = torch.tensor([110, 111, 203, 204, 205, 206], device=device)
    expected[2, :7] = torch.tensor([120, 121, 122, 205, 206, 207, 208], device=device)
    expected_lens = torch.tensor([6, 6, 7], dtype=torch.int32, device=device)

    assert torch.equal(combined_lens, expected_lens)
    assert torch.equal(combined_indices, expected)
