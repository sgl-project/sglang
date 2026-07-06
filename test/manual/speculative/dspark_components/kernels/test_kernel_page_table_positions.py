import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.page_table_positions import (
    build_page_table_positions,
    build_page_table_positions_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

NUM_POOL_REQS = 128
POOL_LEN = 4096
SWA_WINDOW = 128


@pytest.mark.parametrize("num_q", [1, 3, 8, 56, 300])
@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("max_seq_len", [4096, 4000])
def test_triton_matches_torch_prep(num_q, page_size, max_seq_len):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(num_q * 13 + page_size)
    req_to_token = torch.randint(
        0, 1 << 20, (NUM_POOL_REQS, POOL_LEN), device=device, generator=g
    ).to(torch.int32)
    req_pool_indices_repeated = torch.randint(
        0, NUM_POOL_REQS, (num_q,), device=device, generator=g
    ).to(torch.int32)
    seq_lens_casual = torch.randint(
        1, POOL_LEN, (num_q,), device=device, generator=g, dtype=torch.int64
    )

    ref = build_page_table_positions(
        req_to_token=req_to_token,
        req_pool_indices_repeated=req_pool_indices_repeated,
        seq_lens_casual=seq_lens_casual,
        max_seq_len=max_seq_len,
        page_size=page_size,
        swa_window=SWA_WINDOW,
    )
    got = build_page_table_positions_triton(
        req_to_token=req_to_token,
        req_pool_indices_repeated=req_pool_indices_repeated,
        seq_lens_casual=seq_lens_casual,
        max_seq_len=max_seq_len,
        page_size=page_size,
        swa_window=SWA_WINDOW,
    )

    assert torch.equal(got.seq_lens_casual, ref.seq_lens_casual)
    assert torch.equal(got.positions_casual, ref.positions_casual)
    assert torch.equal(got.page_table, ref.page_table)
    assert torch.equal(got.swa_topk_lengths, ref.swa_topk_lengths)
    assert got.page_table.dtype == torch.int32
    assert got.page_table.shape == ref.page_table.shape
