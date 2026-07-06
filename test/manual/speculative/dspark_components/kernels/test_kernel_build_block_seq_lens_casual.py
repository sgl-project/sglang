import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.build_block_seq_lens_casual import (
    build_block_seq_lens_casual,
    build_block_seq_lens_casual_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 128])
@pytest.mark.parametrize("block_size", [1, 5, 7])
def test_triton_matches_torch_over_batch_and_block(bs, block_size):
    device = torch.device("cuda")
    seq_lens = torch.randint(1, 100000, (bs,), dtype=torch.int64, device=device)
    ref = build_block_seq_lens_casual(
        seq_lens=seq_lens, block_size=block_size, device=device
    )
    got = build_block_seq_lens_casual_triton(
        seq_lens=seq_lens, block_size=block_size, device=device
    )
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)
