import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.padded_to_bucket import (
    pad_verify_lens_to_bucket,
    pad_verify_lens_to_bucket_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@pytest.mark.parametrize(
    "bs,padded_bs,graph_num_tokens",
    [
        (1, 1, 6),
        (2, 2, 12),
        (2, 4, 24),
        (3, 16, 96),
        (8, 128, 768),
        (1, 64, 384),
        (3, 3, 16),
        (3, 6, 16),
        (2, 8, 16),
        (5, 32, 64),
    ],
)
def test_triton_matches_torch_eager_and_padded_buckets(bs, padded_bs, graph_num_tokens):
    device = torch.device("cuda")
    num_draft = 6
    verify_lens = torch.randint(
        1, num_draft + 1, (bs,), dtype=torch.int32, device=device
    )
    total = int(verify_lens.sum())
    if total > graph_num_tokens:
        verify_lens = torch.ones(bs, dtype=torch.int32, device=device)
    ref = pad_verify_lens_to_bucket(
        verify_lens=verify_lens,
        graph_num_tokens=graph_num_tokens,
        bs=bs,
        padded_bs=padded_bs,
    )
    got = pad_verify_lens_to_bucket_triton(
        verify_lens=verify_lens,
        graph_num_tokens=graph_num_tokens,
        bs=bs,
        padded_bs=padded_bs,
    )
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)
    assert int(got.to(torch.int64).sum()) == graph_num_tokens
    if padded_bs > bs:
        assert torch.equal(got[:bs], verify_lens.to(torch.int32))
