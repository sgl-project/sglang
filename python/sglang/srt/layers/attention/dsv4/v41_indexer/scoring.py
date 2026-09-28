"""Index scores shared by the full top-k and both candidate schemes: the DeepGEMM
operands and flattened-K scores, the torch per-request scores, and the writes of
a selection back into pool slots."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Generator, Iterator, List, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    finish_paged_indexer_topk,
    fp4_index_logits_candidates,
    fp4_index_logits_decode,
    fp4_index_logits_paged,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import index_q_rope_pack_weights
from sglang.kernels.ops.attention.dsv4.index_logits import flat_index_logits_tiles
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
)
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import async_h2d

from .types import DecodeInputs, PrefillInputs

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

# TODO: use a per-forward mqa_logits_budget_bytes() budget that also
# leaves room for candidate block ids and block-selection scratch.
_DEEP_GEMM_SCORE_BUDGET_BYTES = 2 << 30


class DeepGEMMDecodeData(msgspec.Struct, frozen=True):
    q_fp4: torch.Tensor  # [rows, 1, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, 1, heads] int32, packed ue8m0
    weights: torch.Tensor  # [rows, heads] bf16/fp32 head weights
    k_cache: torch.Tensor  # [pages, page_size, 1, 68] uint8, the layer's index-K pool


class DeepGEMMPrefillData(msgspec.Struct, frozen=True):
    """The rows of one request are consecutive; a selection is a flattened-K
    column, request start plus compressed position."""

    k_slots: torch.Tensor  # [columns] index-K pool slot of each flattened-K column
    request_starts: torch.Tensor  # [rows] int32, first column of the row's request
    lens_per_request: List[int]  # compressed positions at each request's newest token
    rows_per_request: List[int]  # query rows of each request
    compress_lens: torch.Tensor  # [rows] int32, compressed positions the row sees
    q_fp4: torch.Tensor  # [rows, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, heads] int32, packed ue8m0
    weights: torch.Tensor  # [rows, heads] fp32 head weights

    @property
    def num_rows(self) -> int:
        return self.q_fp4.shape[0]

    def empty_selection(self, topk: int) -> torch.Tensor:
        return self.request_starts.new_full(
            (self.num_rows, topk), -1, dtype=torch.int32
        )

    def write_page_indices(self, inputs: PrefillInputs) -> None:
        """The pool slots of the request-local ``inputs.out_raw_indices`` rows."""
        if inputs.out_page_indices is None:
            return
        raw = inputs.out_raw_indices[: self.num_rows]
        columns = (raw + self.request_starts[:, None]).clamp_min(0)
        inputs.out_page_indices[: self.num_rows, : raw.shape[1]] = torch.where(
            raw >= 0, self.k_slots[columns], -1
        )

    def write_selection(self, selected: torch.Tensor, inputs: PrefillInputs) -> None:
        """Write flattened-K columns ``selected`` into the selection buffers."""
        num_tokens, topk = selected.shape
        chosen = selected >= 0
        if inputs.out_page_indices is not None:
            inputs.out_page_indices[:num_tokens, :topk] = torch.where(
                chosen, self.k_slots[selected.clamp_min(0)], -1
            ).to(torch.int32)
        inputs.out_raw_indices[:num_tokens, :topk] = torch.where(
            chosen, selected - self.request_starts[:, None], -1
        )


def quantize_index_q(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        quantize_fp4_indexer_tensor,
    )

    rows, heads = q.shape[0], q.shape[1]
    q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
    return q_fp4.view(rows, heads, 64), q_sf.view(rows, heads)


def get_index_k_cache(
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    layer_id: int,
    page_size: int,
) -> torch.Tensor:
    """The layer's index-K pool as ``[pages, page_size, 1, 68]`` uint8."""
    k_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
    assert k_cache.dim() == 2
    # fp4: 64 payload + 4 scale per slot
    return k_cache.view(k_cache.shape[0], page_size, 1, 68)


def _index_q_and_weights(
    *,
    indexer: DeepseekV41Indexer,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp4 index queries ``[rows, heads, 64]`` int8 with ``[rows, heads]`` packed
    ue8m0 scales, and fp32 head weights ``[rows, heads]``."""
    rows, heads = x.shape[0], indexer.n_local_heads
    q, _ = indexer.wq_b(q_lora)
    q_fp4, q_sf, weights = index_q_rope_pack_weights(
        q.view(rows, heads, indexer.index_head_dim),
        torch.view_as_real(freqs_cis).flatten(-2),
        positions,
        indexer.head_weights_raw(x),
        indexer.head_weight_scale,
    )
    return q_fp4.view(rows, heads, 64), q_sf.view(rows, heads), weights


def get_deep_gemm_decode_data(
    inputs: DecodeInputs, token_to_kv_pool: DeepSeekV4TokenToKVPool
) -> DeepGEMMDecodeData:
    indexer = inputs.indexer
    # The kernel sums head scores locally, so the indexer heads must be replicated.
    assert indexer.n_local_heads == indexer.n_heads
    q_fp4, q_sf, weights = _index_q_and_weights(
        indexer=indexer,
        x=inputs.x,
        q_lora=inputs.q_lora,
        freqs_cis=inputs.freqs_cis,
        positions=inputs.positions,
    )
    bs = q_fp4.shape[0]
    q_fp4 = q_fp4.view(bs, 1, indexer.n_local_heads, 64)
    q_sf = q_sf.view(bs, 1, indexer.n_local_heads)

    # Index pool page (64 slots); metadata.page_table is expanded to match.
    k_cache = get_index_k_cache(
        token_to_kv_pool=token_to_kv_pool,
        layer_id=inputs.layer_id,
        page_size=inputs.paged_metadata.compressed_page_size,
    )
    return DeepGEMMDecodeData(q_fp4, q_sf, weights, k_cache)


def get_deep_gemm_prefill_data(
    inputs: PrefillInputs, req_to_token: torch.Tensor
) -> Optional[DeepGEMMPrefillData]:
    """None when the chunk has no row or no visible compressed position."""
    ratio = inputs.compress_ratio
    pos = inputs.positions
    assert (
        inputs.seq_lens_cpu is not None
        and inputs.rows_per_request is not None
        and inputs.rows_per_request_device is not None
    ), "the DeepGEMM prefill indexer needs the CPU length vectors"
    device = pos.device
    # Visible compressed positions per request; (pos + 1) // ratio bounds each row.
    lens_per_request = [s // ratio for s in inputs.seq_lens_cpu]
    columns = sum(lens_per_request)
    num_tokens = pos.shape[0]
    if columns == 0 or num_tokens == 0:
        return None
    starts = list(itertools.accumulate(lens_per_request, initial=0))[:-1]
    lens = async_h2d(lens_per_request, dtype=torch.int64, device=device)
    starts_dev = async_h2d(starts, dtype=torch.int64, device=device)
    request = torch.repeat_interleave(
        torch.arange(len(lens_per_request), device=device), lens, output_size=columns
    )
    position = torch.arange(columns, device=device) - starts_dev[request]
    pool_rows = inputs.req_pool_indices.to(torch.int64)[request]
    q_fp4, q_sf, weights = _index_q_and_weights(
        indexer=inputs.indexer,
        x=inputs.x,
        q_lora=inputs.q_lora,
        freqs_cis=inputs.freqs_cis,
        positions=pos,
    )
    return DeepGEMMPrefillData(
        k_slots=req_to_token[pool_rows, position * ratio].to(torch.int64) // ratio,
        request_starts=torch.repeat_interleave(
            starts_dev.to(torch.int32),
            inputs.rows_per_request_device.to(torch.int64),
            output_size=num_tokens,
        ),
        lens_per_request=lens_per_request,
        rows_per_request=inputs.rows_per_request,
        compress_lens=((pos + 1) // ratio).to(torch.int32),
        q_fp4=q_fp4,
        q_sf=q_sf,
        weights=weights,
    )


def score_tiles(
    data: DeepGEMMPrefillData,
    kv: Tuple[torch.Tensor, torch.Tensor],
    *,
    width_align: int,
) -> Generator[Tuple[slice, torch.Tensor], None, None]:
    yield from flat_index_logits_tiles(
        q=(data.q_fp4, data.q_sf),
        kv=kv,
        weights=data.weights,
        starts=data.request_starts,
        lengths=data.compress_lens,
        context_lengths=data.lens_per_request,
        budget_bytes=_DEEP_GEMM_SCORE_BUDGET_BYTES,
        width_align=width_align,
    )


def dense_prefill_topk(
    data: DeepGEMMPrefillData,
    kv: Tuple[torch.Tensor, torch.Tensor],
    *,
    out: torch.Tensor,
) -> None:
    """Request-local compressed positions into ``out``, ``-1`` padded, unordered."""
    from sglang.kernels.ops.attention.dsv4 import topk_transform_ragged_v2

    # Score columns are request-relative, so a zero offset emits request-local positions.
    zero_offsets = torch.zeros_like(data.request_starts)
    for tile, logits in score_tiles(data, kv, width_align=4):
        topk_transform_ragged_v2(
            logits,
            data.compress_lens[tile],
            out_offsets=zero_offsets[tile],
            out_indices=out[tile],
        )
        # Free this tile's logits before the generator scores the next one.
        del logits


# ---------- torch ----------


# Arbitrary cap on one bf16 [rows, heads, lc] score chunk; transients run ~3x this.
_TORCH_SCORE_BUDGET_BYTES = 1 << 30


class RequestScores(msgspec.Struct, frozen=True):
    lc: int  # compressed positions visible at its newest row
    k: int
    columns: torch.Tensor  # arange(lc)
    tok: torch.Tensor  # [rows_b] its query rows in the chunk
    lens: torch.Tensor  # [rows_b] compressed positions each row sees
    slots: torch.Tensor  # [lc] index-K pool slot of each position


class ChunkScores(msgspec.Struct, frozen=True):
    rows: slice  # into the request's rows
    tok: torch.Tensor
    lens: torch.Tensor
    scores: torch.Tensor  # [rows, lc], or compact candidate scores
    candidate_blocks: Optional[torch.Tensor] = None


class DecodeScores(msgspec.Struct, frozen=True):
    bs: int
    lmax: int
    lens: torch.Tensor  # [bs]
    slots: torch.Tensor  # [bs, lmax]
    scores: torch.Tensor  # [bs, lmax]


def prefill_requests(
    *,
    inputs: PrefillInputs,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    candidate_blocks: Optional[torch.Tensor] = None,
    block_size: int = 8,
) -> Iterator[tuple[RequestScores, Iterator[ChunkScores]]]:
    """Requests with nothing visible yet are skipped."""
    pool = token_to_kv_pool
    ratio = inputs.compress_ratio
    indexer = inputs.indexer
    req, pos = inputs.req_rows, inputs.positions
    assert req is not None, "the torch indexer needs the batch's rows"
    inputs.reset_outputs()
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    # A compressed position is visible once the query has passed its last token.
    compress_lens = (pos + 1) // ratio
    topk = indexer.index_topk
    # Unsupported shapes keep the independent Torch scoring/selection path.
    if candidate_blocks is not None and not (
        q.is_cuda
        and torch.version.cuda is not None
        and q.shape[1:] == (32, 128)
        and q.dtype == weights.dtype == torch.bfloat16
        and block_size == 8
    ):
        candidate_blocks = None
    for r in torch.unique_consecutive(req).tolist():
        tok = (req == r).nonzero().squeeze(1)
        lens = compress_lens[tok]
        lc = int(lens.max().item())
        if lc == 0:
            continue
        j = torch.arange(lc, device=pos.device)
        slots = req_to_token[r, j * ratio].to(torch.int64) // ratio
        # Dequantize only this request's visible K rows; the table is pool-sized.
        index_k = pool.get_low_ratio_index_k_dequant(inputs.layer_id, slots)
        request = RequestScores(
            lc=lc, k=min(topk, lc), columns=j, tok=tok, lens=lens, slots=slots
        )
        yield (
            request,
            _score_chunks(
                indexer=indexer,
                q=q,
                weights=weights,
                index_k=index_k,
                request=request,
                candidate_blocks=candidate_blocks,
            ),
        )


def _score_chunks(
    *,
    indexer: DeepseekV41Indexer,
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    request: RequestScores,
    candidate_blocks: Optional[torch.Tensor] = None,
) -> Iterator[ChunkScores]:
    lc, j = request.lc, request.columns
    # Chunk rows so the [rows, heads, lc] bf16 scores stay under the budget.
    rows_per_chunk = max(1, _TORCH_SCORE_BUDGET_BYTES // (q.shape[1] * lc * 2))
    if candidate_blocks is not None and index_k.dtype == torch.bfloat16:
        from sglang.kernels.ops.attention.dsv4.candidate_bf16_mqa import (
            candidate_bf16_mqa_logits,
        )

        # No [rows, heads, lc] or per-query K tensor on this path. Bound the
        # FP32 output and the logical-index/top-k temporaries instead.
        width = candidate_blocks.shape[1] * 8
        rows_per_chunk = max(1, _TORCH_SCORE_BUDGET_BYTES // max(1, width * 32))
    else:
        candidate_blocks = None
    for start in range(0, request.tok.numel(), rows_per_chunk):
        rows = slice(start, start + rows_per_chunk)
        tok_c, lens_c = request.tok[rows], request.lens[rows]
        blocks = None if candidate_blocks is None else candidate_blocks[tok_c]
        if blocks is None:
            s = indexer.scores(q[tok_c], index_k, weights[tok_c])
            s = s.masked_fill(j[None, :] >= lens_c[:, None], -torch.inf)
        else:
            s = candidate_bf16_mqa_logits(
                q[tok_c], index_k, weights[tok_c], blocks, lens_c
            )
        yield ChunkScores(
            rows=rows, tok=tok_c, lens=lens_c, scores=s, candidate_blocks=blocks
        )


def write_prefill(
    inputs: PrefillInputs,
    request: RequestScores,
    chunk: ChunkScores,
    idx: torch.Tensor,
) -> None:
    k, lc = request.k, request.lc
    idx = idx.sort(dim=-1).values
    reach = idx < chunk.lens[:, None]
    if inputs.out_page_indices is not None:
        inputs.out_page_indices[chunk.tok, :k] = torch.where(
            reach, request.slots[idx.clamp_max(lc - 1)], -1
        ).to(torch.int32)
    inputs.out_raw_indices[chunk.tok, :k] = torch.where(reach, idx, -1).to(torch.int32)


class PagedDecodeScores(msgspec.Struct, frozen=True):
    """Paged decode scores with graph-stable logical capacity ``lmax``.

    Only positions below each device-side ``lens`` are initialized. Top-k
    consumers must respect those lengths. With ``has_candidate_mask``, masked
    positions have -inf scores and must be discarded during selection writeback.
    Compact candidates instead initialize every score column; score_lens is
    their scan width, while lens remains the causal bound in logical K space.
    """

    bs: int
    lmax: int
    lens: torch.Tensor
    scores: torch.Tensor
    req: torch.Tensor
    req_to_token: torch.Tensor
    ratio: int
    plan: torch.Tensor
    has_candidate_mask: bool
    # Compact columns need their own scan lengths; lens stays in logical K space.
    candidate_blocks: Optional[torch.Tensor] = None
    score_lens: Optional[torch.Tensor] = None


def decode_scores(
    *,
    inputs: DecodeInputs,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    candidate_blocks: Optional[torch.Tensor] = None,
) -> Optional[DecodeScores | PagedDecodeScores]:
    """None when there is no row, or nothing visible yet."""
    pool = token_to_kv_pool
    ratio = inputs.compress_ratio
    indexer = inputs.indexer
    req, pos = inputs.req_rows, inputs.positions
    paged = get_platform().is_sm90
    bs = req.shape[0]
    assert pos.shape[0] == bs, (
        f"decode expects one token per request, {pos.shape=} {bs=}"
    )
    if bs == 0:
        inputs.reset_outputs()
        return None
    lens = (pos + 1) // ratio
    metadata = inputs.paged_metadata
    # V4 reserves the replay bound in metadata; visibility stays on device.
    # A capture-time length read would both synchronize and truncate replay.
    lmax = min(metadata.max_compressed_seq_len, req_to_token.shape[1] // ratio)
    if lmax == 0:
        inputs.reset_outputs()
        return None
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    table = pool.get_index_k_with_scale_buffer(inputs.layer_id)
    if paged:
        lens = lens.to(torch.int32)
        if candidate_blocks is not None:
            assert candidate_mask is None
            # Valid candidates may be interspersed with padding/causal holes.
            score_lens = torch.full_like(lens, candidate_blocks.shape[1] * 8)
            plan = plan_topk_v2(score_lens)
            scores = fp4_index_logits_candidates(
                q,
                weights,
                req,
                req_to_token,
                lens,
                table,
                table.shape[1] // 68,
                lmax,
                ratio,
                candidate_blocks,
            )
        else:
            score_lens = None
            # Prepare the plan before scoring, away from its top-k consumer.
            plan = plan_topk_v2(lens)
            scores = fp4_index_logits_paged(
                q,
                weights,
                req,
                req_to_token,
                lens,
                table,
                table.shape[1] // 68,
                lmax,
                ratio,
                candidate_mask,
            )
        return PagedDecodeScores(
            bs=bs,
            lmax=lmax,
            lens=lens,
            scores=scores,
            req=req,
            req_to_token=req_to_token,
            ratio=ratio,
            plan=plan,
            has_candidate_mask=candidate_mask is not None,
            candidate_blocks=candidate_blocks,
            score_lens=score_lens,
        )
    assert candidate_blocks is None, "compact paged scoring requires SM90"
    inputs.reset_outputs()
    j = torch.arange(lmax, device=pos.device)
    valid = j[None, :] < lens[:, None]
    slots = req_to_token[req[:, None], (j * ratio)[None, :]].to(torch.int64) // ratio
    slots = slots.masked_fill(~valid, 0)
    scores = fp4_index_logits_decode(
        q, weights, slots, lens, table, table.shape[1] // 68
    )
    return DecodeScores(bs=bs, lmax=lmax, lens=lens, slots=slots, scores=scores)


def select_decode(
    inputs: DecodeInputs, d: DecodeScores | PagedDecodeScores, topk: int
) -> None:
    k = min(topk, d.lmax)
    if isinstance(d, PagedDecodeScores):
        k = min(k, d.scores.shape[1])
        if not k:
            inputs.reset_outputs()
            return
        idx = torch.empty((d.bs, k), dtype=torch.int32, device=d.scores.device)
        scan_lens = d.lens if d.score_lens is None else d.score_lens
        topk_transform_paged_v2(d.scores, scan_lens, None, idx, 1, d.plan)
        finish_paged_indexer_topk(
            idx,
            d.scores,
            d.lens,
            d.req,
            d.req_to_token,
            inputs.out_page_indices,
            None,
            d.ratio,
            d.has_candidate_mask,
            candidate_blocks=d.candidate_blocks,
        )
    else:
        write_decode(inputs, d, d.scores.topk(k, dim=-1, sorted=False).indices)


def write_decode(inputs: DecodeInputs, d: DecodeScores, idx: torch.Tensor) -> None:
    k = idx.shape[1]
    idx = idx.sort(dim=-1).values
    reach = idx < d.lens[:, None]
    inputs.out_page_indices[: d.bs, :k] = torch.where(
        reach, d.slots.gather(1, idx.clamp_max(d.lmax - 1)), -1
    ).to(torch.int32)


def compact_topk(
    scores: torch.Tensor,
    blocks: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    width: int,
    block_size: int,
) -> torch.Tensor:
    """Restore compact columns to request-local positions; width pads invalid picks.

    write_prefill sorts these positions before mapping to physical slots. The -inf check drops padding even when top-k exceeds visibility.
    """
    out = torch.full(
        (scores.shape[0], topk), width, dtype=torch.int64, device=scores.device
    )
    k = min(topk, scores.shape[1])
    if not k or not scores.shape[0]:
        return out
    values, columns = scores.topk(k, dim=-1, sorted=False)
    positions = (
        blocks.long().gather(1, columns // block_size) * block_size
        + columns % block_size
    )
    valid = (
        (values > -torch.inf)
        & (positions >= 0)
        & (positions < width)
        & (positions < lengths[:, None])
    )
    out[:, :k] = positions.masked_fill(~valid, width)
    return out
