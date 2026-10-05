"""Opt-in SM90 static-verification selection for the V4.1 decode backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.environ import envs

from .types import CandidateMetadata, DecodeInputs, Selection

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


@dataclass
class CandidateBlocks(CandidateMetadata):
    """Sorted block IDs with device lengths refreshed on each graph replay.

    Ordinary decode consumers can use ``blocks`` directly. Length-aware
    consumers also use the device lengths; never cache a derived full mask.
    """

    blocks: torch.Tensor  # [rows, topk_blocks] int32, -1 padding
    lengths: torch.Tensor  # [rows] int32, selected capacity in positions
    is_prefix: torch.Tensor  # [rows] int32, blocks are consecutive from zero
    width: int  # source capacity in compressed positions
    block_size: int


def _candidate_mask(published, width: int, block_size: int) -> torch.Tensor:
    """Adapt a standard block-ID publish to prefix scoring, without caching."""
    if isinstance(published, CandidateBlocks):
        from sglang.kernels.ops.attention.dsv4.sm90_length_aware_indexer import (
            materialize_candidate_mask,
        )

        assert published.width >= width and published.block_size == block_size
        return materialize_candidate_mask(published)
    blocks = published.blocks
    num_blocks = (width + block_size - 1) // block_size
    keep = torch.zeros(
        (*blocks.shape[:-1], num_blocks + 1), dtype=torch.bool, device=blocks.device
    )
    ids = blocks.to(torch.int64)
    ids = ids.masked_fill((ids < 0) | (ids >= num_blocks), num_blocks)
    keep.scatter_(-1, ids, True)
    return keep[..., :num_blocks].repeat_interleave(block_size, dim=-1)[..., :width]


def try_sm90_decode(
    *,
    inputs: DecodeInputs,
    out: Selection,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    is_source: bool = False,
    is_consumer: bool = False,
    published: Optional[CandidateMetadata] = None,
) -> Tuple[bool, Optional[CandidateMetadata]]:
    """Return whether selection was handled, plus a source's published blocks.

    Candidate roles come from the caller: graph variants may bypass filtering
    even for a layer whose indexer normally publishes or consumes candidates.
    """
    ratio = inputs.compress_ratio
    if not (
        envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.get()
        and inputs.is_verify
        and inputs.group_size > 1
        and ratio in (1, 2)
    ):
        return False, None
    indexer = inputs.indexer
    req, pos = inputs.req_rows, inputs.positions
    bs = req.shape[0]
    assert pos.shape[0] == bs, (
        f"decode expects one token per request, {pos.shape=} {bs=}"
    )
    # Metadata reserves graph replay capacity; visibility remains on device.
    lmax = min(
        inputs.paged_metadata.max_compressed_seq_len, req_to_token.shape[1] // ratio
    )
    if bs == 0 or lmax == 0:
        out.reset()
        return True, None
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    table = token_to_kv_pool.get_index_k_with_scale_buffer(inputs.layer_id)
    if not (
        q.is_cuda
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(q.device)[0] == 9
        and q.dtype == weights.dtype == torch.bfloat16
        and q.shape[1:] == (32, 128)
        and q.is_contiguous()
        and weights.is_contiguous()
        and req_to_token.dtype == torch.int32
        and req_to_token.stride(1) == 1
        and table.dtype == torch.uint8
        and table.dim() == 2
        and table.stride(1) == 1
    ):
        return False, None

    from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
        select_candidate_block_ids,
        topk_among_blocks,
    )

    # Imported only after dispatch to avoid a module-initialization cycle.
    from .dense_blocks import BlockIds

    out.reset()
    if is_consumer:
        assert published is not None and published.blocks.shape[0] == bs
    use_length_aware = (
        envs.SGLANG_OPT_DSV41_SM90_LENGTH_AWARE_INDEXER.get()
        and 0 < indexer.index_topk <= 2048
        and (
            not is_source
            or (
                0 < indexer.candidate_topk_blocks <= 2048
                and indexer.candidate_block_size in (1, 2, 4, 8, 16, 32, 64, 128)
            )
        )
    )
    if use_length_aware:
        from sglang.kernels.ops.attention.dsv4.sm90_length_aware_indexer import (
            candidate_blocks,
            prefix_logits,
            prepare_candidate_lengths,
            publish_topk,
            select_prefix_topk,
        )

        request = req.to(torch.int64).contiguous()
        compact = envs.SGLANG_OPT_DSV41_SM90_COMPACT_CANDIDATES.get() and (
            is_source or is_consumer
        )
        candidates = (
            published
            if compact and is_consumer and isinstance(published, CandidateBlocks)
            else None
        )
        if compact:
            visible, score_lens = prepare_candidate_lengths(
                pos, ratio, lmax, candidates
            )
        else:
            visible = ((pos + 1) // ratio).clamp(0, lmax).to(torch.int32).contiguous()
            score_lens = visible
        consume = None
        if is_consumer and candidates is None:
            consume = _candidate_mask(published, lmax, indexer.candidate_block_size)
            assert consume.shape[0] == bs and consume.shape[1] >= lmax
            assert consume.stride(1) == 1
        score_width = lmax
        if candidates is not None:
            assert candidates.width >= lmax
            assert candidates.block_size == indexer.candidate_block_size
            score_width = min(lmax, candidates.blocks.shape[1] * candidates.block_size)
        scores = prefix_logits(
            q,
            weights,
            req_to_token,
            request,
            score_lens,
            table,
            table.shape[1] // 68,
            ratio,
            score_width,
            consume,
            candidates=candidates,
            visible=visible,
        )
        result = None
        if is_source:
            result = candidate_blocks(
                scores,
                visible,
                lmax,
                indexer.candidate_topk_blocks,
                indexer.candidate_block_size,
            )
            if not compact:
                # Main's decode protocol carries block IDs even without the
                # compact scoring flag. Drop only the optional device lengths.
                result = BlockIds(blocks=result.blocks)
        idx = select_prefix_topk(
            scores, score_lens, min(indexer.index_topk, score_width)
        )
        publish_topk(
            idx,
            scores,
            score_lens,
            request,
            req_to_token,
            out.page_indices,
            out.raw_indices,
            ratio,
            is_consumer,
            candidates=candidates,
        )
        return True, result

    from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import (
        fp4_index_logits_mapped_sm90,
    )

    lens = (pos + 1) // ratio
    scores = fp4_index_logits_mapped_sm90(
        q,
        weights,
        req_to_token,
        req.to(torch.int64).contiguous(),
        lens.to(torch.int64).contiguous(),
        table,
        table.shape[1] // 68,
        ratio,
        lmax,
    )
    result = None
    if is_source:
        result = BlockIds(
            blocks=select_candidate_block_ids(
                scores,
                lens[:, None],
                indexer.candidate_topk_blocks,
                indexer.candidate_block_size,
            )
        )
    k = min(indexer.index_topk, lmax)
    if is_consumer:
        idx = topk_among_blocks(
            scores, lens, published.blocks, k, block_size=indexer.candidate_block_size
        )
        idx = idx.masked_fill(idx < 0, lmax)
    else:
        idx = scores.topk(k, dim=-1, sorted=False).indices
    if 0 < k <= 1024:
        from sglang.kernels.ops.attention.dsv4.sm90_fp4_topk import sort_map_topk

        sort_map_topk(
            idx, lens, req, req_to_token, out.page_indices, out.raw_indices, ratio
        )
    else:
        idx = idx.sort(dim=-1).values
        reach = idx < lens[:, None]
        slots = (
            req_to_token[req[:, None], idx.clamp_max(lmax - 1) * ratio].to(torch.int64)
            // ratio
        )
        out.page_indices[:bs, :k] = torch.where(reach, slots, -1).to(torch.int32)
        if out.raw_indices is not None:
            out.raw_indices[:bs, :k] = torch.where(reach, idx, -1).to(torch.int32)
    return True, result
