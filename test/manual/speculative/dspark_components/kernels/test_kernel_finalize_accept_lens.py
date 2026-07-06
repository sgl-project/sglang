import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.finalize_accept_lens import (
    finalize_accept_lens,
    finalize_accept_lens_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("prefix_dtype", [torch.int32, torch.int64])
def test_triton_matches_torch_finalize(bs, prefix_dtype):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(bs)
    correct_len = torch.randint(0, 7, (bs,), device=device, generator=g).to(torch.int32)
    cap_trim_lens = torch.randint(0, 4, (bs,), device=device, generator=g).to(
        torch.int64
    )
    prefix_lens = torch.randint(1, 4000, (bs,), device=device, generator=g).to(
        prefix_dtype
    )

    ref = finalize_accept_lens(
        correct_len=correct_len,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=prefix_lens,
    )
    got = finalize_accept_lens_triton(
        correct_len=correct_len,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=prefix_lens,
    )

    assert torch.equal(got.commit_lens, ref.commit_lens)
    assert torch.equal(got.new_seq_lens, ref.new_seq_lens)
    assert torch.equal(got.cap_trim_lens, ref.cap_trim_lens)
    assert got.commit_lens.dtype == torch.int32
    assert got.new_seq_lens.dtype == prefix_dtype
    assert got.cap_trim_lens.dtype == torch.int32
