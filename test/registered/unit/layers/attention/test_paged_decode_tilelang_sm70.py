import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0),
    reason="The grouped TileLang decoder requires an NVIDIA V100",
)


@pytest.mark.parametrize("cache_dtype", [torch.float16, torch.float8_e5m2])
def test_grouped_decode_matches_paged_reference(cache_dtype):
    from sglang.srt.layers.attention.tilelang_fa_v100 import (
        grouped_decode_forward,
    )

    torch.manual_seed(7)
    tokens = 257
    page_size = 16
    pages = math.ceil(tokens / page_size)
    heads = 6
    dim = 256
    k_scale = 0.75
    v_scale = 1.25

    q = torch.randn(1, heads, dim, device="cuda", dtype=torch.float16)
    logical_k = torch.randn(
        pages,
        page_size,
        1,
        dim,
        device="cuda",
        dtype=torch.float16,
    ).to(cache_dtype)
    logical_v = torch.randn_like(logical_k.to(torch.float16)).to(cache_dtype)
    permutation = torch.randperm(pages, device="cuda")
    physical_k = torch.empty_like(logical_k)
    physical_v = torch.empty_like(logical_v)
    physical_k[permutation] = logical_k
    physical_v[permutation] = logical_v
    page_table = permutation.to(torch.int32).view(1, -1)
    seq_lens = torch.tensor([tokens], device="cuda", dtype=torch.int32)

    actual = grouped_decode_forward(
        q,
        physical_k,
        physical_v,
        page_table,
        seq_lens,
        softmax_scale=dim**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    dense_k = logical_k.flatten(0, 1)[:tokens, 0].float() * k_scale
    dense_v = logical_v.flatten(0, 1)[:tokens, 0].float() * v_scale
    scores = torch.einsum("bhd,nd->bhn", q.float(), dense_k) * dim**-0.5
    expected = torch.softmax(scores, dim=-1) @ dense_v

    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-3)
