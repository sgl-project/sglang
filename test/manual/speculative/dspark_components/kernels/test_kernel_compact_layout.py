import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.compact_layout import (
    compact_row_index,
    compact_row_index_triton,
    compact_verify_ids,
    compact_verify_ids_triton,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

GAMMA = 5
T = GAMMA + 1


def _make_verify_lens(bs, device):
    return torch.randint(1, T + 1, (bs,), dtype=torch.int32, device=device)


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("pad", ["exact", "bucket"])
def test_row_index_triton_matches_torch(bs, pad):
    device = torch.device("cuda")
    verify_lens = _make_verify_lens(bs, device)
    total = int(verify_lens.sum().item())
    padded_total = total if pad == "exact" else bs * T
    req_r, within_r, valid_r = compact_row_index(
        verify_lens=verify_lens, padded_total=padded_total, device=device
    )
    req_t, within_t, valid_t = compact_row_index_triton(
        verify_lens=verify_lens, padded_total=padded_total, device=device
    )
    assert torch.equal(req_t, req_r)
    assert torch.equal(within_t, within_r)
    assert torch.equal(valid_t, valid_r)


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("pad", ["exact", "bucket"])
def test_verify_ids_triton_matches_torch(bs, pad):
    device = torch.device("cuda")
    verify_lens = _make_verify_lens(bs, device)
    total = int(verify_lens.sum().item())
    padded_total = total if pad == "exact" else bs * T
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=verify_lens, graph_num_tokens=padded_total
    )
    # Production DSpark passes an anchor+draft-token block here, while
    # draft_tokens only contains the gamma draft-token slots.
    draft_block_ids = torch.randint(
        0, 129280, (bs, T), dtype=torch.int64, device=device
    )
    draft_tokens = torch.randint(
        0, 129280, (bs, GAMMA), dtype=torch.int64, device=device
    )
    ref = compact_verify_ids(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )
    got = compact_verify_ids_triton(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)
