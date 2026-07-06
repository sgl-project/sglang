import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.commit_inject_layout import (
    build_commit_inject_layout,
    build_commit_inject_layout_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

STRIDE = 7
NUM_POOL_REQS = 300
POOL_LEN = 400
NUM_FULL_SLOTS = 50000


def _make_inputs(bs, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    req_to_token = torch.randint(
        0,
        NUM_FULL_SLOTS,
        (NUM_POOL_REQS, POOL_LEN),
        device=device,
        generator=g,
        dtype=torch.int64,
    )
    full_to_swa = torch.randint(
        0, 1 << 20, (NUM_FULL_SLOTS,), device=device, generator=g, dtype=torch.int64
    )
    req_pool_indices = torch.randperm(
        NUM_POOL_REQS, device=device, generator=g, dtype=torch.int64
    )[:bs]
    prefix_lens = torch.randint(
        1, POOL_LEN - STRIDE, (bs,), device=device, generator=g, dtype=torch.int64
    )
    block_pos_offsets = torch.arange(STRIDE, device=device, dtype=torch.int64)
    commit_lens = torch.randint(
        0, STRIDE + 1, (bs,), device=device, generator=g, dtype=torch.int32
    )
    return (
        req_pool_indices,
        req_to_token,
        prefix_lens,
        block_pos_offsets,
        full_to_swa,
        commit_lens,
    )


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
def test_triton_matches_torch_layout(bs):
    device = torch.device("cuda")
    (
        req_pool_indices,
        req_to_token,
        prefix_lens,
        block_pos_offsets,
        full_to_swa,
        commit_lens,
    ) = _make_inputs(bs, device, seed=1000 + bs)

    ref = build_commit_inject_layout(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        prefix_lens=prefix_lens,
        block_pos_offsets=block_pos_offsets,
        full_to_swa_mapping=full_to_swa,
        commit_lens=commit_lens,
        stride=STRIDE,
    )
    got = build_commit_inject_layout_triton(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        prefix_lens=prefix_lens,
        block_pos_offsets=block_pos_offsets,
        full_to_swa_mapping=full_to_swa,
        commit_lens=commit_lens,
        stride=STRIDE,
    )

    assert torch.equal(got.swa_loc, ref.swa_loc)
    assert torch.equal(got.positions, ref.positions)
    assert got.swa_loc.dtype == torch.int32
    assert got.positions.dtype == torch.int64


def test_commit_len_edges_mask_expected_columns():
    device = torch.device("cuda")
    (
        req_pool_indices,
        req_to_token,
        prefix_lens,
        block_pos_offsets,
        full_to_swa,
        _,
    ) = _make_inputs(2, device, seed=42)
    commit_lens = torch.tensor([0, STRIDE], device=device, dtype=torch.int32)

    got = build_commit_inject_layout_triton(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        prefix_lens=prefix_lens,
        block_pos_offsets=block_pos_offsets,
        full_to_swa_mapping=full_to_swa,
        commit_lens=commit_lens,
        stride=STRIDE,
    )

    swa_2d = got.swa_loc.view(2, STRIDE)
    assert bool((swa_2d[0] == -1).all())
    assert bool((swa_2d[1] >= 0).all())
