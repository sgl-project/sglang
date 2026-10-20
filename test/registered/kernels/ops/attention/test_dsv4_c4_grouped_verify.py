# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DeepSeek-V4 C4 indexer scores a static verify's draft rows per request."""

import sys

import pytest
import torch

from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="nightly", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not is_sm100_supported(), reason="Grouped verify rows run on SM100 DeepGEMM."
)

BLOCK_KV, HEAD_DIM, HEADS = 64, 128, 64


def _verify_rows(fp4, requests, next_n, pages_per_seq=16):
    """C4 indexer inputs of a static verify: request-major rows whose compressed
    lengths repeat or step by one, as positions // 4 do."""
    torch.manual_seed(0)
    rows, device = requests * next_n, "cuda"
    num_pages = requests * pages_per_seq
    capacity = pages_per_seq * BLOCK_KV
    first = torch.randint(capacity // 2, 4 * (capacity - next_n), (requests,))
    positions = first[:, None] + torch.arange(next_n)[None, :]
    lens = ((positions + 1) // 4).reshape(-1).to(torch.int32).to(device)
    table = torch.randperm(num_pages, device=device, dtype=torch.int32).view(
        requests, pages_per_seq
    )
    q = torch.randn(rows, HEADS, HEAD_DIM, device=device)
    k = torch.randn(num_pages * BLOCK_KV, HEAD_DIM, device=device)
    weights = torch.randn(rows, HEADS, device=device) / HEADS
    if fp4:
        from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache import (
            store_fp4_index_k_cache,
        )
        from sglang.kernels.ops.quantization.fp4_indexer_quant import (
            quantize_fp4_indexer_tensor,
        )

        cache = torch.empty(num_pages, BLOCK_KV * 68, dtype=torch.uint8, device=device)
        store_fp4_index_k_cache(
            k, cache, torch.arange(k.shape[0], device=device), page_size=BLOCK_KV
        )
        q_fp4, q_sf = quantize_fp4_indexer_tensor(q)
        q = (q_fp4.view(rows, 1, HEADS, 64), q_sf.view(rows, 1, HEADS))
        cache = cache.view(num_pages, BLOCK_KV, 1, 68)
    else:
        scale = k.abs().amax(-1, keepdim=True) / 448.0
        k_fp8 = (k / scale).to(torch.float8_e4m3fn)
        cache = torch.cat(
            (
                k_fp8.view(torch.uint8).view(num_pages, -1),
                scale.view(num_pages, -1).contiguous().view(torch.uint8),
            ),
            dim=1,
        ).view(num_pages, BLOCK_KV, 1, HEAD_DIM + 4)
        q = q.to(torch.float8_e4m3fn).unsqueeze(1)
    return q, cache, weights, lens, table.repeat_interleave(next_n, dim=0)


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("next_n", [2, 3, 4])
def test_grouped_verify_rows_match_per_row(fp4, next_n):
    """Grouped rows score every row exactly as DeepGEMM does row by row."""
    import deep_gemm

    from sglang.srt.layers.attention.dsv4.indexer import group_verify_rows
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

    q, cache, weights, lens, table = _verify_rows(fp4, 32, next_n)
    fn = deep_gemm.fp8_fp4_paged_mqa_logits if fp4 else deep_gemm.fp8_paged_mqa_logits

    def score(rows_per_request):
        metadata = PagedIndexerMetadata(
            256, 64, table, lens, False, rows_per_request=rows_per_request
        )
        grouped_q, grouped_lens, grouped_table = group_verify_rows(
            q, lens.unsqueeze(-1), table, rows_per_request
        )
        return metadata.max_compressed_seq_len, fn(
            grouped_q,
            cache,
            weights,
            grouped_lens,
            grouped_table,
            metadata.deep_gemm_metadata,
            metadata.max_compressed_seq_len,
            False,
        )

    max_len, per_row = score(1)
    _, grouped = score(next_n)
    assert grouped.shape == per_row.shape
    valid = torch.arange(max_len, device="cuda")[None, :] < lens[:, None]
    assert torch.isfinite(per_row[valid]).all()
    torch.testing.assert_close(grouped[valid], per_row[valid], atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
