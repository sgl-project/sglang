"""Block top-k over per-row block scores for the MiniMax-M3 sparse decode indexer.

Drop-in replacement for the 2-stage split-K Triton topk
(``_topk_index_partial_kernel`` + ``_topk_index_merge_kernel``): given the
decode score tensor ``[num_heads, batch, max_seqblock]`` it produces
``topk_idx`` ``[num_heads, batch, topk]`` (0-indexed block ids, sorted
ascending, ``-1`` padded at the tail). Ascending order is required by the MSA
fmha_sm100 consumer; the Triton ``_gqa_share_sparse_decode_kernel`` is
order-insensitive.

``minimax_decode_topk_page_table`` additionally fuses the page-table transform
for the dense paged backend (trtllm_mha / fa3) and returns the page table plus
the per-query effective KV length.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(seq_dtype: torch.dtype) -> Module:
    args = make_cpp_args(seq_dtype, True)  # SeqLenT, kUsePDL
    return load_jit(
        "minimax_decode_topk",
        *args,
        cuda_files=["minimax/minimax_decode_topk.cuh"],
        cuda_wrappers=[
            ("minimax_decode_topk", f"minimax_decode_topk<{args}>"),
            (
                "minimax_decode_topk_page_table",
                f"minimax_decode_topk_page_table<{args}>",
            ),
        ],
    )


def minimax_decode_topk(
    score: torch.Tensor,  # [num_heads, batch, max_seqblock] fp32
    seq_lens: torch.Tensor,  # [batch] int32/int64
    block_size: int,
    topk: int,
    out: torch.Tensor | None = None,  # [num_heads, batch, topk] int32
) -> torch.Tensor:
    assert score.is_cuda and score.dtype == torch.float32 and score.dim() == 3
    assert seq_lens.is_cuda and seq_lens.dim() == 1
    assert seq_lens.dtype in (torch.int32, torch.int64)
    num_heads, batch, max_seqblock = score.shape
    assert seq_lens.shape[0] == batch

    if not score.is_contiguous():
        score = score.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()

    if out is None:
        out = torch.empty(
            (num_heads, batch, topk), dtype=torch.int32, device=score.device
        )
    else:
        assert out.shape == (num_heads, batch, topk)
        assert out.dtype == torch.int32 and out.is_cuda
        assert out.is_contiguous()

    module = _jit_module(seq_lens.dtype)
    module.minimax_decode_topk(score, seq_lens, out, int(block_size), int(topk))
    return out


def minimax_decode_topk_page_table(
    score: torch.Tensor,  # [num_kv_heads, batch, max_seqblock] fp32
    seq_lens: torch.Tensor,  # [batch] int32/int64
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len] int32
    slot_ids: torch.Tensor,  # [batch] int64 (req_pool_indices)
    block_size: int,
    topk: int,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused top-k + page-table transform: select the top-k blocks and emit the
    per-(batch, kv-head) paged page table consumed by the dense backend
    (trtllm_mha / fa3), instead of block ids, plus the per-pseudo-request effective
    KV length (cache_seqlens, from the actual selection). Both are allocated here
    and returned.

    For DP attention (num_kv_heads > 1) each kv head selects its own blocks, so
    (batch, head) pseudo-requests are flattened batch-major into the outputs
    ``[batch*num_kv_heads, topk*block_size/page_size]`` / ``[batch*num_kv_heads]``
    (matching ``q.view(bs, nkv, gqa, d).reshape(bs*nkv, gqa, d)``). The page index
    is head-encoded (head-minor) as ``base_page*num_kv_heads + head`` -- the index
    into an HND cache ``[num_pages, nkv, ps, D]`` reshaped to
    ``[num_pages*nkv, 1, ps, D]``; num_kv_heads==1 reproduces the single-kv-head
    TP>=4 behavior (page index == base_page)."""
    assert score.is_cuda and score.dtype == torch.float32 and score.dim() == 3
    num_heads, batch, max_seqblock = score.shape
    assert block_size % page_size == 0
    assert req_to_token.dtype == torch.int32 and slot_ids.dtype == torch.int64
    if not score.is_contiguous():
        score = score.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not slot_ids.is_contiguous():
        slot_ids = slot_ids.contiguous()

    max_sparse_pages = topk * (block_size // page_size)
    page_table = torch.empty(
        (batch * num_heads, max_sparse_pages), dtype=torch.int32, device=score.device
    )
    real_seq_lens = torch.empty(
        (batch * num_heads,), dtype=torch.int32, device=score.device
    )

    module = _jit_module(seq_lens.dtype)
    module.minimax_decode_topk_page_table(
        score,
        seq_lens,
        req_to_token,
        slot_ids,
        page_table,
        real_seq_lens,
        int(block_size),
        int(topk),
        int(page_size),
    )
    return page_table, real_seq_lens
