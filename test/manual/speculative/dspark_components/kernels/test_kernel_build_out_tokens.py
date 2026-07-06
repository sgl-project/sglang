import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.build_out_tokens import (
    build_out_tokens,
    build_out_tokens_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("cl_dtype", [torch.int32, torch.int64])
def test_triton_matches_torch_incl_bonus_at_every_position(bs, cl_dtype):
    device = torch.device("cuda")
    gamma = 5
    verify_num_draft_tokens = gamma + 1
    draft_tokens = torch.randint(
        0, 129280, (bs, gamma), dtype=torch.int64, device=device
    )
    bonus = torch.randint(0, 129280, (bs,), dtype=torch.int64, device=device)
    correct_len = (torch.arange(bs, device=device) % (gamma + 1)).to(cl_dtype)
    ref = build_out_tokens(
        draft_tokens=draft_tokens,
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=verify_num_draft_tokens,
        gamma=gamma,
    )
    got = build_out_tokens_triton(
        draft_tokens=draft_tokens,
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=verify_num_draft_tokens,
        gamma=gamma,
    )
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)
