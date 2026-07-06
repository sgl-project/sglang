import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.accept_sampling import (
    gather_two_level_bonus_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

T = 6


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
def test_gather_two_level_bonus_matches_torch(bs):
    device = torch.device("cuda")
    n_pred = bs * T
    accept_index = torch.randint(0, n_pred, (bs, T), dtype=torch.int64, device=device)
    predicts = torch.randint(0, 129280, (n_pred,), dtype=torch.int64, device=device)
    correct_len = torch.randint(0, T, (bs,), dtype=torch.int32, device=device)
    row_ids = torch.arange(bs, device=device)
    accept_pos = accept_index[row_ids, correct_len.long()].long()
    ref = predicts[accept_pos].to(torch.int64)
    got = gather_two_level_bonus_triton(
        accept_index=accept_index, predicts=predicts, correct_len=correct_len
    )
    assert torch.equal(got, ref)
