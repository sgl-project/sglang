"""Unit test for the ROCm (gfx950) FP4 indexer score kernel.

ROCm has no DeepGEMM ``fp8_fp4_paged_mqa_logits``, so the DeepSeek-V4 fp4 indexer
scores its paged-MQA logits with the fused Triton ``fp4_paged_mqa_logits_triton``
(native mxfp4 MFMA via ``tl.dot_scaled``, fp8 q x fp4 KV, no KV dequant). This
test pins that kernel to the pure-torch reference ``fp4_paged_mqa_logits_torch``:
the fp4 KV pool is produced by the (separately tested) store kernel, then both
paths score the same fp8 q against it and must agree.

The kernel feeds fp4 straight into the CDNA4 matrix cores, so it only runs where
native mxfp4 MFMA exists (gfx950); the test is skipped elsewhere.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    fp4_paged_mqa_logits_triton,
    store_fp4_index_k_cache,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    fp4_paged_mqa_logits_torch,
)
from sglang.srt.utils import get_device, is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

HEAD_DIM = 128
FP4_DIM = HEAD_DIM // 2
SCALE_BYTES = 4
PAGE_SIZE = 64

requires_mxfp4_mfma = pytest.mark.skipif(
    not is_gfx95_supported(),
    reason="fp4 indexer score kernel needs native mxfp4 MFMA (ROCm gfx950)",
)


def _build_paged_fp4_kv(batch_size: int, seq_len: int, device: torch.device):
    """Quantize + page-store one MQA KV vector per position into the raw fp4
    indexer pool, returning the pool and the ``page_table`` that addresses it.

    Pool layout per page (fixed by store_fp4_index_k_cache): PAGE_SIZE*64 fp4
    bytes followed by PAGE_SIZE*4 UE8M0 scale bytes. Sequence ``b`` owns the
    contiguous page block ``[b*pages_per_seq, (b+1)*pages_per_seq)``.
    """
    pages_per_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = batch_size * pages_per_seq
    page_table = torch.arange(num_pages, device=device, dtype=torch.int32).reshape(
        batch_size, pages_per_seq
    )
    pos = torch.arange(seq_len, device=device)
    # flat cache slot for token (b, i): page_table[b, i // PAGE_SIZE] * PAGE_SIZE + i % PAGE_SIZE
    loc = (
        page_table[:, pos // PAGE_SIZE] * PAGE_SIZE + (pos % PAGE_SIZE)
    ).reshape(-1).to(torch.int64)
    kv = torch.randn(
        batch_size * seq_len, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros(
        num_pages,
        PAGE_SIZE * (FP4_DIM + SCALE_BYTES),
        device=device,
        dtype=torch.uint8,
    )
    store_fp4_index_k_cache(kv, cache, loc, page_size=PAGE_SIZE)
    return cache, page_table


@requires_mxfp4_mfma
@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len",
    [
        (1, 64, 100),  # single seq, spans 2 pages, tail masked
        (2, 32, 64),  # exact one page per seq
        (3, 128, 200),  # DSV4 128-head width, multi-page
        (1, 16, 1),  # single kv position (degenerate)
    ],
)
# The indexer hands the kernel the pool as a raw [pages, page_size*68] buffer or
# a reshaped [pages, 64, 1, 68] view (indexer.py); the kernel must derive the
# per-page stride identically for both, so exercise both layouts.
@pytest.mark.parametrize("cache_4d", [False, True], ids=["flat", "paged4d"])
def test_fp4_paged_mqa_logits_matches_reference(
    batch_size: int, num_heads: int, seq_len: int, cache_4d: bool
) -> None:
    torch.manual_seed(batch_size * 1000 + num_heads * 10 + seq_len)
    device = get_device()

    cache, page_table = _build_paged_fp4_kv(batch_size, seq_len, device)
    if cache_4d:
        cache = cache.view(cache.shape[0], PAGE_SIZE, 1, FP4_DIM + SCALE_BYTES)
    seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    # q kept fp8 on the ROCm path (only KV is fp4); shape [B, 1, H, HEAD].
    q_fp8 = torch.randn(
        batch_size, 1, num_heads, HEAD_DIM, device=device, dtype=torch.bfloat16
    ).to(FP8_DTYPE)
    q = (q_fp8,)
    weight = torch.randn(batch_size, num_heads, device=device, dtype=torch.float32)
    max_seq_len = page_table.shape[1] * PAGE_SIZE

    triton_logits = fp4_paged_mqa_logits_triton(
        q, cache, weight, seq_lens, page_table, max_seq_len=max_seq_len
    )
    ref_logits = fp4_paged_mqa_logits_torch(
        q, cache, weight, seq_lens, page_table, None, max_seq_len
    )

    assert triton_logits.shape == (batch_size, max_seq_len)
    # The reference scores the same exact fp4/fp8 operands in fp32, matching the
    # kernel's fp32 MFMA accumulation; only reduction order over HEAD_DIM differs.
    torch.testing.assert_close(triton_logits, ref_logits, rtol=1e-3, atol=1e-3)
    # Positions beyond seq_len must stay zero (early-exit / mask in both paths).
    if seq_len < max_seq_len:
        assert torch.count_nonzero(triton_logits[:, seq_len:]) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
