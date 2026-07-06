import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.expand_prefill_casually import (
    expand_prefill_casually,
    expand_prefill_casually_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


def _make_inputs(bs, uniform_extend, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    seq_lens = torch.randint(
        8, 500, (bs,), device=device, generator=g, dtype=torch.int64
    )
    if uniform_extend is not None:
        extend = torch.full((bs,), uniform_extend, device=device, dtype=torch.int64)
    else:
        extend = torch.randint(
            1, 8, (bs,), device=device, generator=g, dtype=torch.int64
        )
    start_loc = torch.cumsum(extend, dim=0) - extend
    req_pool_indices = torch.randperm(
        512, device=device, generator=g, dtype=torch.int64
    )[:bs]
    return seq_lens, extend, start_loc, req_pool_indices


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("pad", [0, 5])
def test_triton_matches_torch_vectorized_branch(bs, pad):
    device = torch.device("cuda")
    seq_lens, extend, start_loc, req_pool_indices = _make_inputs(
        bs, None, device, seed=100 + bs
    )
    num_tokens = int(extend.sum())
    padded = num_tokens + pad

    ref = expand_prefill_casually(
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_seq_lens=extend,
        extend_start_loc=start_loc,
        seq_lens_cpu=None,
        extend_seq_lens_cpu=None,
        num_tokens=num_tokens,
        padded_num_tokens=padded,
    )
    got = expand_prefill_casually_triton(
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_seq_lens=extend,
        num_tokens=num_tokens,
        padded_num_tokens=padded,
    )

    assert torch.equal(got.seq_lens_casual, ref.seq_lens_casual)
    assert torch.equal(got.req_pool_indices_repeated, ref.req_pool_indices_repeated)


@pytest.mark.parametrize("bs", [1, 3, 8])
def test_triton_matches_torch_loop_branch_uniform(bs):
    device = torch.device("cuda")
    block_size = 6
    seq_lens, extend, _, req_pool_indices = _make_inputs(
        bs, block_size, device, seed=200 + bs
    )
    num_tokens = bs * block_size

    ref = expand_prefill_casually(
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_seq_lens=extend,
        extend_start_loc=None,
        seq_lens_cpu=[int(x) for x in seq_lens.tolist()],
        extend_seq_lens_cpu=[block_size] * bs,
        num_tokens=num_tokens,
        padded_num_tokens=None,
    )
    got = expand_prefill_casually_triton(
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_seq_lens=extend,
        num_tokens=num_tokens,
        padded_num_tokens=None,
    )

    assert torch.equal(got.seq_lens_casual, ref.seq_lens_casual)
    assert torch.equal(got.req_pool_indices_repeated, ref.req_pool_indices_repeated)
