"""The sparse table of the two-level indexer from a row's selected blocks: the
sorted, page-transformed block table and DeepGEMM's schedule for it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .utils import make_name

if TYPE_CHECKING:
    pass

CANDIDATE_BLOCK_SIZE = 8  # positions per block; DeepGEMM accepts 8 or 16


@cache_once
def _jit_candidate_block_table_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("candidate_block_table"),
        *args,
        cuda_files=["deepseek_v4/candidate_block_table.cuh"],
        cuda_wrappers=[("transform", f"CandidateBlockTableKernel<{args}>::transform")],
    )


def sort_candidate_blocks(
    blocks: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    *,
    out_pages: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The block table of the two-level indexer from a row's selected blocks,
    in place: ``blocks`` ``[rows, k]`` int32 block ids in any order, ``-1``
    padded, become the same ids ascending with ``INT32_MAX`` past ``min(k,
    ceil(seq_lens[b] / 8))``; the matching pool slots / 8 (``page_table[b, id //
    bpp] * bpp + id % bpp``, ``bpp = page_size // 8``, same padding) go to
    ``out_pages``. A row with at most ``k`` blocks gets the identity table
    regardless of its input. Returns ``out_pages``.
    """
    if out_pages is None:
        out_pages = torch.empty_like(blocks)
    _jit_candidate_block_table_module().transform(
        blocks, seq_lens, page_table, out_pages, page_size
    )
    return out_pages


def build_sparse_indexer_schedule(
    blocks: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    q_dtype: torch.dtype,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    """DeepGEMM's schedule for the published blocks: ``seq_lens`` ``[rows]``
    int32, ``page_table`` ``[rows, pages]`` int32 at the index pool's page size.
    ``request_ids`` ``[rows]`` int32 lets DeepGEMM pair two rows of a request on
    one KV pass; each row keeps its own block list and output layout, and paired
    rows must share their page-table row."""
    import deep_gemm

    return deep_gemm.get_paged_sparse_mqa_logits_metadata(
        seq_lens.contiguous(),
        page_table,
        request_ids,
        page_size,
        blocks,
        q_dtype,
        CANDIDATE_BLOCK_SIZE,
    )
