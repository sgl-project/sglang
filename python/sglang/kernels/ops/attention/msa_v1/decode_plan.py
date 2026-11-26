"""One-launch refresh of the per-step MSA-v1 decode-plan buffers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_PAGE_SIZE = tl.constexpr(128)
_PAGE_SHIFT = tl.constexpr(_PAGE_SIZE.value.bit_length() - 1)
# MSA-v1 decode indexers take at most 16 queries per launch, so a longer query
# runs in 16-query chunks (msa_plan._MAX_DECODE_QUERY_LENGTH).
_CHUNK_QUERIES = tl.constexpr(16)
_TOPK = tl.constexpr(16)
# The MSA-v1 decode indexers' page limit (_MAXIMUM_PAGES).
_MAXIMUM_PAGES = 8192
# Lanes per loop step over requests and over query rows; arbitrary, not tuned.
_BLOCK_B = 256
_BLOCK_R = 1024


@triton.jit
def _write_page_table_rows(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_table_ptr,
    pid,
    req_to_token_stride,
    MAX_PAGES: tl.constexpr,
    PAGE_BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = pid // PAGE_BLOCKS
    pages = (pid % PAGE_BLOCKS) * BLOCK_P + tl.arange(0, BLOCK_P)
    page_mask = pages < MAX_PAGES
    req = tl.load(req_pool_indices_ptr + row).to(tl.int64)
    # Every column, past seq_len too, as build_msa_v1_page_table fills them.
    slots = tl.load(
        req_to_token_ptr + req * req_to_token_stride + pages.to(tl.int64) * _PAGE_SIZE,
        mask=page_mask,
    )
    # The arithmetic shift floors negative slots too; Triton's // truncates.
    page_ids = slots >> _PAGE_SHIFT
    tl.store(
        page_table_ptr + row.to(tl.int64) * MAX_PAGES + pages,
        page_ids,
        mask=page_mask,
    )


@triton.jit
def _write_plan_lengths(
    seq_lens_ptr,
    final_seq_lens_ptr,
    chunk_seq_lens_ptr,
    query_seq_lens_ptr,
    flashinfer_seq_lens_ptr,
    batch_size,
    QUERY_LENGTH: tl.constexpr,
    SEQ_LEN_OFFSET: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HAS_FLASHINFER: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    req_lanes = tl.arange(0, BLOCK_B)
    for start in range(0, batch_size, BLOCK_B):
        reqs = start + req_lanes
        req_mask = reqs < batch_size
        final_seq_lens = (
            tl.load(seq_lens_ptr + reqs, mask=req_mask, other=0).to(tl.int32)
            + SEQ_LEN_OFFSET
        )
        tl.store(final_seq_lens_ptr + reqs, final_seq_lens, mask=req_mask)
        for chunk in tl.static_range(NUM_CHUNKS):
            # Each chunk sees the KV length through its own last query.
            if chunk == NUM_CHUNKS - 1:
                chunk_seq_lens = final_seq_lens
            else:
                chunk_seq_lens = final_seq_lens - (
                    QUERY_LENGTH - _CHUNK_QUERIES * (chunk + 1)
                )
            tl.store(
                chunk_seq_lens_ptr + chunk * batch_size + reqs,
                chunk_seq_lens,
                mask=req_mask,
            )
    num_query_rows = batch_size * QUERY_LENGTH
    query_lanes = tl.arange(0, BLOCK_R)
    for start in range(0, num_query_rows, BLOCK_R):
        query_rows = start + query_lanes
        query_mask = query_rows < num_query_rows
        final_seq_lens = (
            tl.load(
                seq_lens_ptr + query_rows // QUERY_LENGTH, mask=query_mask, other=0
            ).to(tl.int32)
            + SEQ_LEN_OFFSET
        )
        query_seq_lens = tl.maximum(
            final_seq_lens - (QUERY_LENGTH - 1 - query_rows % QUERY_LENGTH), 0
        )
        tl.store(query_seq_lens_ptr + query_rows, query_seq_lens, mask=query_mask)
        if HAS_FLASHINFER:
            # Mirrors FlashInferQ8K8Decode.update_lengths.
            skipped_pages = tl.maximum(
                ((query_seq_lens - 1) >> _PAGE_SHIFT) - (_TOPK - 1), 0
            )
            sparse_lens = tl.maximum(query_seq_lens - skipped_pages * _PAGE_SIZE, 1)
            for head in tl.static_range(NUM_KV_HEADS):
                tl.store(
                    flashinfer_seq_lens_ptr + head * num_query_rows + query_rows,
                    sparse_lens,
                    mask=query_mask,
                )


@triton.jit(do_not_specialize=["batch_size", "req_to_token_stride"])
def _msa_v1_decode_plan_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    page_table_ptr,
    final_seq_lens_ptr,
    chunk_seq_lens_ptr,
    query_seq_lens_ptr,
    flashinfer_seq_lens_ptr,
    batch_size,
    req_to_token_stride,
    MAX_PAGES: tl.constexpr,
    QUERY_LENGTH: tl.constexpr,
    SEQ_LEN_OFFSET: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HAS_FLASHINFER: tl.constexpr,
    PAGE_BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # One program per (request, page block) for the page table; the last one
    # writes every length buffer.
    pid = tl.program_id(0)
    if pid < batch_size * PAGE_BLOCKS:
        _write_page_table_rows(
            req_to_token_ptr,
            req_pool_indices_ptr,
            page_table_ptr,
            pid,
            req_to_token_stride,
            MAX_PAGES,
            PAGE_BLOCKS,
            BLOCK_P,
        )
    else:
        _write_plan_lengths(
            seq_lens_ptr,
            final_seq_lens_ptr,
            chunk_seq_lens_ptr,
            query_seq_lens_ptr,
            flashinfer_seq_lens_ptr,
            batch_size,
            QUERY_LENGTH,
            SEQ_LEN_OFFSET,
            NUM_CHUNKS,
            NUM_KV_HEADS,
            HAS_FLASHINFER,
            BLOCK_B,
            BLOCK_R,
        )


def _check_int32(
    *, name: str, tensor: torch.Tensor, shape: tuple[int, ...], device: torch.device
) -> None:
    if (
        tensor.dtype != torch.int32
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.device != device
    ):
        raise ValueError(
            f"{name} must be a contiguous int32 {shape} tensor on {device}, got "
            f"{tensor.dtype} {tuple(tensor.shape)} on {tensor.device}"
        )


def _check_decode_plan_buffers(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    query_length: int,
    page_table: torch.Tensor,
    final_seq_lens: torch.Tensor,
    chunk_seq_lens: torch.Tensor,
    query_seq_lens: torch.Tensor,
    flashinfer_seq_lens: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if page_table.ndim != 2 or query_length <= 0:
        raise ValueError("page_table must be [batch, max_pages], query_length > 0")
    batch_size, max_pages = page_table.shape
    num_chunks = triton.cdiv(query_length, _CHUNK_QUERIES.value)
    if not 0 < max_pages <= _MAXIMUM_PAGES:
        raise ValueError(
            f"MSA-v1 supports 1 to {_MAXIMUM_PAGES} pages, got {max_pages}"
        )
    if (
        req_to_token.dtype != torch.int32
        or req_to_token.ndim != 2
        or req_to_token.stride(1) != 1
    ):
        raise ValueError("req_to_token must be int32 [rows, tokens], unit stride")
    if (max_pages - 1) * _PAGE_SIZE.value >= req_to_token.shape[1]:
        raise ValueError(
            "MSA-v1 page table exceeds req_to_token capacity: "
            f"{max_pages} pages, capacity={req_to_token.shape[1]}."
        )
    num_query_rows = batch_size * query_length
    num_kv_heads = 1 if flashinfer_seq_lens is None else flashinfer_seq_lens.shape[0]
    device = req_to_token.device
    for name, tensor, shape in (
        ("page_table", page_table, (batch_size, max_pages)),
        ("final_seq_lens", final_seq_lens, (batch_size,)),
        ("chunk_seq_lens", chunk_seq_lens, (num_chunks, batch_size)),
        ("query_seq_lens", query_seq_lens, (num_query_rows,)),
        ("flashinfer_seq_lens", flashinfer_seq_lens, (num_kv_heads, num_query_rows)),
    ):
        if tensor is not None:
            _check_int32(name=name, tensor=tensor, shape=shape, device=device)
    # Read with unit stride; the pool indices are not bounds-checked.
    for name, tensor in (
        ("req_pool_indices", req_pool_indices),
        ("seq_lens", seq_lens),
    ):
        if (
            tensor.shape != (batch_size,)
            or tensor.dtype not in (torch.int32, torch.int64)
            or not tensor.is_contiguous()
            or tensor.device != device
        ):
            raise ValueError(
                f"{name} must be a contiguous int32 or int64 ({batch_size},) "
                f"tensor on {device}, got {tensor.dtype} "
                f"{tuple(tensor.shape)} on {tensor.device}"
            )
    return batch_size, max_pages, num_chunks, num_kv_heads


def update_msa_v1_decode_plan(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    query_length: int,
    seq_len_offset: int,
    page_table: torch.Tensor,
    final_seq_lens: torch.Tensor,
    chunk_seq_lens: torch.Tensor,
    query_seq_lens: torch.Tensor,
    flashinfer_seq_lens: torch.Tensor | None = None,
) -> None:
    """Rewrites the per-step buffers of an MSA-v1 decode plan in one launch.

    The page table gets ``req_to_token[req_pool_indices, 128 * p] >> 7`` for every page;
    the length buffers derive from ``seq_lens + seq_len_offset``. ``chunk_seq_lens``
    has one row per 16-query indexer chunk: the KV length through its last query.
    """
    batch_size, max_pages, num_chunks, num_kv_heads = _check_decode_plan_buffers(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        query_length=query_length,
        page_table=page_table,
        final_seq_lens=final_seq_lens,
        chunk_seq_lens=chunk_seq_lens,
        query_seq_lens=query_seq_lens,
        flashinfer_seq_lens=flashinfer_seq_lens,
    )
    # Pages per program; arbitrary, not tuned.
    block_p = min(1024, max(128, triton.next_power_of_2(max_pages)))
    page_blocks = triton.cdiv(max_pages, block_p)
    _msa_v1_decode_plan_kernel[(batch_size * page_blocks + 1,)](
        req_to_token_ptr=req_to_token,
        req_pool_indices_ptr=req_pool_indices,
        seq_lens_ptr=seq_lens,
        page_table_ptr=page_table,
        final_seq_lens_ptr=final_seq_lens,
        chunk_seq_lens_ptr=chunk_seq_lens,
        query_seq_lens_ptr=query_seq_lens,
        flashinfer_seq_lens_ptr=flashinfer_seq_lens,
        batch_size=batch_size,
        req_to_token_stride=req_to_token.stride(0),
        MAX_PAGES=max_pages,
        QUERY_LENGTH=query_length,
        SEQ_LEN_OFFSET=seq_len_offset,
        NUM_CHUNKS=num_chunks,
        NUM_KV_HEADS=num_kv_heads,
        HAS_FLASHINFER=flashinfer_seq_lens is not None,
        PAGE_BLOCKS=page_blocks,
        BLOCK_P=block_p,
        BLOCK_B=_BLOCK_B,
        BLOCK_R=_BLOCK_R,
    )
