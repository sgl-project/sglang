"""Device-side page-table builder for the trtllm_mha attention backend.

trtllm_mha builds its block (page) table from the global ``req_to_token`` pool.
Doing it with a host-max PyTorch gather forces a ``seq_lens.max().item()`` D2H
sync (the CPU must know the page-table width before launching). This kernel
instead derives the per-request page count from the device-side ``seq_lens``
tensor, so the build is sync-free: the grid/buffer use the static
``max_num_pages`` upper bound, while each program self-guards on the real length.

The kernel is MHA-owned (no dependency on the MLA kv-index kernels) and also
emits the SWA-translated block table in the same pass via the full->SWA lookup
table, so SWA hybrid models stay sync-free too.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# Tokens covered per CTA along the page-block (grid axis-1) dimension.
_MHA_KV_INDEX_BLOCK_TOKENS = 4096
# Triton kernels can only read module globals that are tl.constexpr instances.
_MHA_KV_INDEX_BLOCK_TOKENS_TL = tl.constexpr(_MHA_KV_INDEX_BLOCK_TOKENS)


def get_num_mha_kv_index_blocks(num_pages: int, page_size: int) -> int:
    """Grid axis-1 size: number of page-block CTAs spanning the widest sequence.

    ``num_pages`` is the per-row width of the page-table buffer (the static
    ``max_num_pages`` upper bound). One CTA handles ``_MHA_KV_INDEX_BLOCK_TOKENS
    // page_size`` pages.
    """
    pages_per_block = _MHA_KV_INDEX_BLOCK_TOKENS // page_size
    return (num_pages + pages_per_block - 1) // pages_per_block


@triton.jit
def create_trtllm_mha_kv_indices_triton(
    req_to_token_ptr,  # [max_reqs, max_context_len], int32
    req_pool_indices_ptr,  # [bs]
    seq_lens_ptr,  # [bs], per-request KV length in tokens
    full_to_swa_ptr,  # full->SWA token-slot lookup table, or dummy when not SWA
    page_table_ptr,  # [bs, num_pages] int32 block ids (output)
    swa_page_table_ptr,  # [bs, num_pages] int32 SWA block ids (output), or dummy
    req_to_token_stride: tl.constexpr,
    page_table_stride: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HAS_SWA: tl.constexpr,
):
    """Fill ``page_table_ptr`` (and ``swa_page_table_ptr`` when ``HAS_SWA``).

    Program ``(pid_req, pid_blk)`` writes the block ids of request ``pid_req`` for
    the page-block ``pid_blk``. It reads the KV token slot at each page boundary
    from ``req_to_token`` and converts it to a block id (``slot // PAGE_SIZE``).
    Programs past the request's page count are guarded out, so the work (and the
    DRAM traffic) is bounded by the device-side ``seq_lens`` — no host max needed.
    """
    PAGES_PER_BLOCK: tl.constexpr = _MHA_KV_INDEX_BLOCK_TOKENS_TL // PAGE_SIZE
    pid_req = tl.program_id(0)
    pid_blk = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_req)
    num_pages = tl.cdiv(seq_len, PAGE_SIZE)
    num_page_blocks = tl.cdiv(seq_len, _MHA_KV_INDEX_BLOCK_TOKENS_TL)
    if pid_blk >= num_page_blocks:
        return

    req_pool_index = tl.load(req_pool_indices_ptr + pid_req)
    page_idx = tl.arange(0, PAGES_PER_BLOCK) + pid_blk * PAGES_PER_BLOCK
    token_pos = page_idx.to(tl.int64) * PAGE_SIZE
    mask = page_idx < num_pages

    slot = tl.load(
        req_to_token_ptr
        + req_pool_index.to(tl.int64) * req_to_token_stride
        + token_pos,
        mask=mask,
    )
    out_off = pid_req * page_table_stride + page_idx
    tl.store(page_table_ptr + out_off, (slot // PAGE_SIZE).to(tl.int32), mask=mask)
    if HAS_SWA:
        swa_slot = tl.load(full_to_swa_ptr + slot.to(tl.int64), mask=mask)
        tl.store(
            swa_page_table_ptr + out_off,
            (swa_slot // PAGE_SIZE).to(tl.int32),
            mask=mask,
        )


def build_trtllm_mha_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    swa_page_table: Optional[torch.Tensor] = None,
    full_to_swa: Optional[torch.Tensor] = None,
) -> None:
    """Fill ``page_table`` (and ``swa_page_table`` when SWA) on-device, no D2H sync.

    Computes the launch grid from the static page-table width and dispatches
    ``create_trtllm_mha_kv_indices_triton``. ``page_table`` (and, for SWA models,
    ``swa_page_table``) are written in place; the caller owns the buffers so the
    cuda-graph path can reuse its pre-allocated tensors. SWA is enabled iff
    ``full_to_swa`` is provided, which then also requires ``swa_page_table``.
    """
    has_swa = full_to_swa is not None
    assert has_swa == (
        swa_page_table is not None
    ), "full_to_swa and swa_page_table must be provided together"
    bs, num_pages = page_table.shape
    create_trtllm_mha_kv_indices_triton[
        (bs, get_num_mha_kv_index_blocks(num_pages, page_size))
    ](
        req_to_token,
        req_pool_indices,
        cache_seqlens,
        full_to_swa,
        page_table,
        swa_page_table,
        req_to_token.stride(0),
        page_table.stride(0),
        PAGE_SIZE=page_size,
        HAS_SWA=has_swa,
    )
