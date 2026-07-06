import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.cap_correct_len import (
    cap_correct_len,
    cap_correct_len_triton,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("cl_dtype", [torch.int32, torch.int64])
def test_triton_matches_torch_cap_and_trim(bs, cl_dtype):
    device = torch.device("cuda")
    num_draft = 6
    verify_lens = torch.randint(
        1, num_draft + 1, (bs,), dtype=torch.int32, device=device
    )
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=verify_lens, graph_num_tokens=bs * num_draft
    )
    correct_len = (torch.arange(bs, device=device) % (num_draft + 1)).to(cl_dtype)
    capped_ref, trim_ref = cap_correct_len(correct_len=correct_len, layout=layout)
    capped, trim = cap_correct_len_triton(correct_len=correct_len, layout=layout)
    assert capped.dtype == capped_ref.dtype
    assert trim.dtype == trim_ref.dtype
    assert torch.equal(capped, capped_ref)
    assert torch.equal(trim, trim_ref)
