import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.dspark_swa_page_indices import (
    build_dspark_swa_page_indices,
    build_dspark_swa_page_indices_triton,
    compute_dspark_window_gather,
    compute_dspark_window_gather_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

BLOCK_SIZE = 5
SWA_WINDOW = 128
PAGE_ALIGN = 64


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
def test_window_gather_triton_matches_torch(bs):
    device = torch.device("cuda")
    num_q = bs * BLOCK_SIZE
    seq_lens_casual = torch.randint(1, 300, (num_q,), dtype=torch.int32, device=device)
    req_pool_indices_repeated = torch.randint(
        0, 256, (num_q,), dtype=torch.int64, device=device
    )
    ref = compute_dspark_window_gather(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
        block_size=BLOCK_SIZE,
        swa_window=SWA_WINDOW,
    )
    got = compute_dspark_window_gather_triton(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
        block_size=BLOCK_SIZE,
        swa_window=SWA_WINDOW,
    )
    assert got.num_q == ref.num_q and got.bs == ref.bs
    assert torch.equal(got.context_lens, ref.context_lens)
    assert torch.equal(
        got.req_pool_indices_per_request, ref.req_pool_indices_per_request
    )
    assert torch.equal(got.offsets, ref.offsets)
    assert torch.equal(got.invalid, ref.invalid)


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
def test_page_indices_triton_matches_torch(bs):
    device = torch.device("cuda")
    num_q = bs * BLOCK_SIZE
    max_reqs = 300
    max_ctx = 400
    n_full = 50000
    seq_lens_casual = torch.randint(1, 300, (num_q,), dtype=torch.int32, device=device)
    req_pool_indices_repeated = torch.randint(
        0, max_reqs, (num_q,), dtype=torch.int64, device=device
    )
    req_to_token = torch.randint(
        0, n_full, (max_reqs, max_ctx), dtype=torch.int32, device=device
    )
    full_to_swa_mapping = torch.randint(
        0, 20000, (n_full,), dtype=torch.int32, device=device
    )
    out_loc = torch.randint(0, n_full, (num_q,), dtype=torch.int64, device=device)

    gather = compute_dspark_window_gather(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
        block_size=BLOCK_SIZE,
        swa_window=SWA_WINDOW,
    )
    idx_ref, len_ref = build_dspark_swa_page_indices(
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa_mapping,
        req_pool_indices_per_request=gather.req_pool_indices_per_request,
        offsets=gather.offsets,
        invalid=gather.invalid,
        out_loc=out_loc,
        context_lens=gather.context_lens,
        block_size=BLOCK_SIZE,
        swa_window=SWA_WINDOW,
        page_index_aligned_size=PAGE_ALIGN,
    )
    idx_got, len_got = build_dspark_swa_page_indices_triton(
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa_mapping,
        req_pool_indices_per_request=gather.req_pool_indices_per_request,
        offsets=gather.offsets,
        out_loc=out_loc,
        context_lens=gather.context_lens,
        block_size=BLOCK_SIZE,
        swa_window=SWA_WINDOW,
        page_index_aligned_size=PAGE_ALIGN,
    )
    assert idx_got.shape == idx_ref.shape
    assert torch.equal(idx_got, idx_ref)
    assert torch.equal(len_got, len_ref)
