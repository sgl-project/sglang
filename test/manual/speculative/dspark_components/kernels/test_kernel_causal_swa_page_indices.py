import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.causal_swa_page_indices import (
    build_causal_swa_page_indices,
    build_causal_swa_page_indices_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

SWA_WINDOW = 128
ALIGNED = 96
NUM_POOL_REQS = 64
POOL_LEN = 600


@pytest.mark.parametrize("num_q", [1, 3, 8, 40])
@pytest.mark.parametrize("lens_mode", ["short", "cross", "long"])
def test_triton_matches_torch_on_attended_region(num_q, lens_mode):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(num_q * 7)
    req_to_token = torch.randint(
        0, 40000, (NUM_POOL_REQS, POOL_LEN), device=device, generator=g
    ).to(torch.int32)
    full_to_swa = torch.randint(
        0, 1 << 20, (40000,), device=device, generator=g, dtype=torch.int64
    )
    req_pool = torch.randint(0, NUM_POOL_REQS, (num_q,), device=device, generator=g).to(
        torch.int32
    )
    if lens_mode == "short":
        lens = torch.randint(1, SWA_WINDOW, (num_q,), device=device, generator=g)
    elif lens_mode == "cross":
        lens = torch.randint(
            SWA_WINDOW - 4, SWA_WINDOW + 4, (num_q,), device=device, generator=g
        )
    else:
        lens = torch.randint(
            SWA_WINDOW + 1, POOL_LEN, (num_q,), device=device, generator=g
        )
    seq_lens_casual = lens.to(torch.int32)

    ref = build_causal_swa_page_indices(
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa,
        req_pool_indices_repeated=req_pool,
        seq_lens_casual=seq_lens_casual,
        swa_window=SWA_WINDOW,
        page_index_aligned_size=ALIGNED,
    )
    got = build_causal_swa_page_indices_triton(
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa,
        req_pool_indices_repeated=req_pool,
        seq_lens_casual=seq_lens_casual,
        swa_window=SWA_WINDOW,
        page_index_aligned_size=ALIGNED,
    )

    assert got.shape == ref.shape
    assert got.dtype == ref.dtype == torch.int32
    padded_width = ref.shape[1]
    col = torch.arange(padded_width, device=device).view(1, -1)
    attended = col < torch.clamp(seq_lens_casual, max=SWA_WINDOW).view(-1, 1)
    assert torch.equal(got[attended], ref[attended])
    assert bool((got[~attended] == -1).all())
