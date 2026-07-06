import types

import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.build_ragged_verify_window import (
    build_ragged_verify_window,
    build_ragged_verify_window_triton,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

GAMMA = 5
T = GAMMA + 1
CTX = 64


def _fixtures(bs, graph_num_tokens, device):
    verify_lens = torch.randint(1, T + 1, (bs,), dtype=torch.int32, device=device)
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=verify_lens, graph_num_tokens=graph_num_tokens
    )
    seq_lens = torch.randint(1, 20, (bs,), dtype=torch.int64, device=device)
    num_reqs = bs + 3
    req_pool_indices = torch.randperm(num_reqs, device=device)[:bs].to(torch.int64)
    req_to_token = torch.randint(
        0, 1_000_000, (num_reqs, CTX), dtype=torch.int32, device=device
    )
    batch = types.SimpleNamespace(seq_lens=seq_lens, req_pool_indices=req_pool_indices)
    model_runner = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token)
    )
    draft_block_ids = torch.randint(
        0, 129280, (bs, GAMMA), dtype=torch.int64, device=device
    )
    draft_tokens = torch.randint(
        0, 129280, (bs, GAMMA), dtype=torch.int64, device=device
    )
    return layout, batch, model_runner, draft_block_ids, draft_tokens


@pytest.mark.parametrize("bs", [1, 2, 3, 8])
@pytest.mark.parametrize("pad", ["tight", "bucket"])
def test_triton_matches_torch_window(bs, pad):
    device = torch.device("cuda")
    graph_num_tokens = bs * T if pad == "tight" else (bs + 3) * T
    layout, batch, model_runner, dbi, dt = _fixtures(bs, graph_num_tokens, device)
    ref = build_ragged_verify_window(
        batch=batch,
        layout=layout,
        draft_block_ids=dbi,
        draft_tokens=dt,
        bs=bs,
        device=device,
        verify_num_draft_tokens=T,
        model_runner=model_runner,
    )
    got = build_ragged_verify_window_triton(
        batch=batch,
        layout=layout,
        draft_block_ids=dbi,
        draft_tokens=dt,
        bs=bs,
        device=device,
        verify_num_draft_tokens=T,
        model_runner=model_runner,
    )
    assert torch.equal(got.positions, ref.positions)
    assert torch.equal(got.verify_cache_loc, ref.verify_cache_loc)
    assert torch.equal(got.verify_ids, ref.verify_ids)
