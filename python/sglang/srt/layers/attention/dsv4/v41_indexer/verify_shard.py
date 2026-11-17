"""Shard the SM90 target verify indexer across ranks with replicated queries."""

import logging

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.deep_select import topk_page_transform
from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    amax_topk_blocks,
    candidate_block_mask,
)
from sglang.srt.runtime_context import get_parallel, get_platform

from .dense_blocks import BlockIds
from .scoring import PagedDecodeScores, decode_scores, select_decode
from .types import RowShard

logger = logging.getLogger(__name__)
_logged = False


@triton.jit
def _remap_verify_positions(
    Raw,
    Requests,
    Table,
    Output,
    ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    idx = tl.load(Raw + row * TOPK + col, (row < ROWS) & (col < TOPK), -1)
    valid = (row < ROWS) & (col < TOPK) & (idx >= 0)
    if PAGE_SIZE:
        page = tl.load(Table + row * TABLE_STRIDE + idx // PAGE_SIZE, valid, 0)
        slot = page * PAGE_SIZE + idx % PAGE_SIZE
    else:
        req = tl.load(Requests + row, row < ROWS, 0).to(tl.int64)
        slot = tl.load(Table + req * TABLE_STRIDE + idx * RATIO, valid, 0) // RATIO
    tl.store(Output + row * OUTPUT_STRIDE + col, tl.where(valid, slot, -1), col < TOPK)


def try_sharded_verify(backend, batch_size, inputs, is_source, is_consumer, published):
    global _logged
    parallel = get_parallel()
    group = parallel.attn_cp_group
    world = parallel.attn_cp_size
    rows = inputs.req_rows.shape[0]
    width = backend.speculative_num_draft_tokens
    if not (
        get_platform().is_sm90
        and backend.is_dsv41
        and not backend.is_draft_runner
        and world in (4, 8)
        and world == parallel.tp_size
        and parallel.attn_dp_size == 1
        and group.ranks == parallel.tp_group.ranks
        and width is not None
        and width > 1
        and batch_size > 0
        and batch_size % world == 0
        and rows == batch_size * width
        and inputs.out_page_indices.shape[0] >= rows
    ):
        return False

    # Keep each request and its candidate mask on the same rank.
    shard = RowShard(group=group, rank=parallel.attn_cp_rank, world=world, row_align=width)
    count = shard.rows_per_rank(rows)
    deep_select = (
        not (is_source or is_consumer)
        and backend.full_topk_indexer.use_deep_select_decode
    )
    mask = published.decode_mask if is_consumer and published is not None else None
    data = decode_scores(
        inputs=inputs,
        token_to_kv_pool=backend.token_to_kv_pool,
        req_to_token=backend.req_to_token,
        candidate_mask=mask,
        compute_topk_plan=not deep_select,
        score_rows=shard.local_rows(rows),
    )
    if data is None:
        if is_source:
            backend._publish_candidate_metadata(None)
        return True
    assert isinstance(data, PagedDecodeScores)
    if is_consumer:
        assert mask is not None and mask.shape[0] == count
    if is_source:
        selector = backend.decode_candidates
        assert selector.block_size == 8
        blocks = amax_topk_blocks(
            data.scores, data.lens, (data.lens + 7) // 8, selector.topk_blocks
        )
        backend._publish_candidate_metadata(
            BlockIds(
                blocks=blocks,
                decode_mask=candidate_block_mask(blocks, data.lmax, selector.block_size),
            )
        )

    output = inputs.out_page_indices
    topk = inputs.indexer.index_topk
    assert topk == output.shape[1]
    page_size = 0
    table = data.req_to_token
    if deep_select:
        page_size = inputs.paged_metadata.compressed_page_size
        table = inputs.paged_metadata.page_table
        identity = torch.arange(
            triton.cdiv(data.scores.shape[1], page_size),
            device=data.scores.device,
            dtype=torch.int32,
        ).expand(count, -1)
        raw = topk_page_transform(
            data.scores, topk, page_table=identity, page_size=page_size,
            end=data.lens, sorted_index=False,
        )
    else:
        raw = output.new_empty((count, topk))
        select_decode(inputs, data, topk, raw_indices=raw)

    # Gather logical positions because physical KV slots are rank-local.
    merged = shard.gather(raw, rows)
    _remap_verify_positions[(output.shape[0],)](
        merged, inputs.req_rows, table, output, rows, topk, table.stride(0),
        output.stride(0), data.ratio, page_size, triton.next_power_of_2(topk),
        num_warps=4,
    )
    if not _logged:
        logger.info("DSV4.1 Hopper verify indexer sharding active: %d rows per rank", count)
        _logged = True
    return True
