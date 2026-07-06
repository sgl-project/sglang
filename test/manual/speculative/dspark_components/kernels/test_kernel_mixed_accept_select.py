import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.mixed_accept_select import (
    select_mixed_accept,
    select_mixed_accept_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
def test_triton_matches_torch_selection(bs):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(bs)
    greedy_mask = torch.rand(bs, device=device, generator=g) < 0.5
    greedy_len = torch.randint(0, 7, (bs,), device=device, generator=g).to(torch.int64)
    sampling_len = torch.randint(0, 7, (bs,), device=device, generator=g).to(
        torch.int32
    )
    greedy_bonus = torch.randint(0, 100000, (bs,), device=device, generator=g)
    sampling_bonus = torch.randint(0, 100000, (bs,), device=device, generator=g)
    greedy_trim = torch.randint(0, 4, (bs,), device=device, generator=g).to(torch.int64)
    sampling_trim = torch.randint(0, 4, (bs,), device=device, generator=g).to(
        torch.int32
    )

    ref = select_mixed_accept(
        greedy_mask=greedy_mask,
        greedy_len=greedy_len,
        greedy_bonus=greedy_bonus,
        greedy_trim=greedy_trim,
        sampling_len=sampling_len,
        sampling_bonus=sampling_bonus,
        sampling_trim=sampling_trim,
    )
    got = select_mixed_accept_triton(
        greedy_mask=greedy_mask,
        greedy_len=greedy_len,
        greedy_bonus=greedy_bonus,
        greedy_trim=greedy_trim,
        sampling_len=sampling_len,
        sampling_bonus=sampling_bonus,
        sampling_trim=sampling_trim,
    )

    assert torch.equal(got.correct_len, ref.correct_len)
    assert torch.equal(got.bonus, ref.bonus)
    assert torch.equal(got.cap_trim_lens, ref.cap_trim_lens)
    assert got.correct_len.dtype == ref.correct_len.dtype
    assert got.cap_trim_lens.dtype == ref.cap_trim_lens.dtype
