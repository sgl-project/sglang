"""Index scores shared by the full top-k and both candidate schemes: the DeepGEMM
operands and flattened-K scores, the torch per-request scores, and the writes of
a selection back into pool slots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Iterator, List, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    finish_flat_indexer_topk,
    finish_paged_indexer_topk,
    fp4_index_logits_candidates,
    fp4_index_logits_decode,
    fp4_index_logits_paged,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer_prefill import fused_index_scores
from sglang.kernels.ops.attention.dsv4.index_logits import flat_index_logits_tiles
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform

from .types import (
    DecodeInputs,
    PrefillInputs,
    Selection,
)

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

    def write_selection(self, selected: torch.Tensor, out: Selection) -> None:
        if (
            selected.is_cuda
            and get_platform().is_sm100
            and selected.dtype == torch.int32
        ):
            finish_flat_indexer_topk(
                selected,
                self.k_slots,
                self.request_starts,
                out.page_indices,
                out.raw_indices,
            )
            return
        num_tokens, topk = selected.shape
        unselected = torch.iinfo(torch.int32).max
        selected = selected.masked_fill(selected < 0, unselected).sort(dim=-1).values
        chosen = selected != unselected
        out.page_indices[:num_tokens, :topk] = torch.where(
            chosen, self.k_slots[selected.clamp_max(self.k_slots.shape[0] - 1)], -1
        ).to(torch.int32)
        if out.raw_indices is not None:
            out.raw_indices[:num_tokens, :topk] = torch.where(
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


def get_deep_gemm_decode_data(
    inputs: DecodeInputs, token_to_kv_pool: DeepSeekV4TokenToKVPool
) -> DeepGEMMDecodeData:
    indexer = inputs.indexer
    x, q_lora, pos = inputs.x, inputs.q_lora, inputs.positions
    # The kernel sums head scores locally, so the indexer heads must be replicated.
    assert indexer.n_local_heads == indexer.n_heads
    if (
        x.is_cuda
        and torch.version.cuda is not None
        and x.dtype == torch.bfloat16
        and indexer.index_head_dim == 128
    ):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
            index_q_rope_pack_weights,
        )

        q, _ = indexer.wq_b(q_lora)
        q = q.view(q.shape[0], indexer.n_local_heads, indexer.index_head_dim)
        # The fused pack also computes head_weights(x).float(), same rounding.
        q_fp4, q_sf, weights = index_q_rope_pack_weights(
            q,
            torch.view_as_real(inputs.freqs_cis).flatten(-2),
            pos,
            indexer.head_weights_raw(x),  # [bs, n_local] bf16, n32k5120
            indexer.head_weight_scale,
        )
    else:
        q = indexer.queries(q_lora, inputs.freqs_cis[pos])
        q_fp4, q_sf = quantize_index_q(q)
        weights = indexer.head_weights(x).float()  # [bs, n_local]
    bs = q.shape[0]
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
    indexer = inputs.indexer
    pos = inputs.positions
    seq_lens_cpu = inputs.seq_lens_cpu
    assert (
        seq_lens_cpu is not None
        and inputs.rows_per_request is not None
        and inputs.rows_per_request_device is not None
    ), "the DeepGEMM prefill indexer needs the CPU length vectors"
    device = pos.device
    # Visible compressed positions per request; (pos + 1) // ratio bounds each row.
    lc_per_req = [s // ratio for s in seq_lens_cpu]
    req_pool_indices = inputs.req_pool_indices.to(torch.int64)
    slot_chunks, starts, start = [], [], 0
    for r, lc in enumerate(lc_per_req):
        starts.append(start)
        if lc == 0:
            continue
        j = torch.arange(lc, device=device)
        slot_chunks.append(
            req_to_token[req_pool_indices[r], j * ratio].to(torch.int64) // ratio
        )
        start += lc
    num_tokens = pos.shape[0]
    if not slot_chunks or num_tokens == 0:
        return None
    k_slots = torch.cat(slot_chunks)
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    q_fp4, q_sf = quantize_index_q(q)
    weights = indexer.head_weights(inputs.x).float()
    compress_lens = ((pos + 1) // ratio).to(torch.int32)
    request_starts = torch.repeat_interleave(
        torch.tensor(starts, dtype=torch.int32, device=device),
        inputs.rows_per_request_device.to(torch.int64),
        output_size=num_tokens,
    )
    return DeepGEMMPrefillData(
        k_slots=k_slots,
        request_starts=request_starts,
        lens_per_request=lc_per_req,
        rows_per_request=inputs.rows_per_request,
        compress_lens=compress_lens,
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
    topk: int,
) -> torch.Tensor:
    """Flattened-K columns, ``-1`` padded, unordered."""
    from sglang.kernels.ops.attention.dsv4 import topk_transform_ragged_v2

    selected = data.empty_selection(topk)
    for tile, logits in score_tiles(data, kv, width_align=4):
        topk_transform_ragged_v2(
            logits,
            data.compress_lens[tile],
            out_offsets=data.request_starts[tile],
            out_indices=selected[tile],
        )
        # Free this tile's logits before the generator scores the next one.
        del logits
    return selected


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
    out: Selection,
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
    out.reset()
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    # A compressed position is visible once the query has passed its last token.
    compress_lens = (pos + 1) // ratio
    topk = indexer.index_topk
    # Keep the Torch oracle and unsupported shapes on the existing full scorer.
    if candidate_blocks is not None and not (
        q.is_cuda
        and get_platform().is_sm90
        and not envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get()
        and q.shape[1:] == (32, 128)
        and q.dtype == weights.dtype == torch.bfloat16
        and block_size == 8
    ):
        candidate_blocks = None
    rows_per_request = inputs.rows_per_request
    seq_lens = inputs.seq_lens_cpu
    assert rows_per_request is not None and seq_lens is not None, (
        "the torch prefill indexer needs CPU request lengths"
    )
    start = 0
    # Extend rows are contiguous within each request. The scheduler already
    # knows both boundaries and the newest position; deriving them with
    # unique/nonzero/item on CUDA would block this lane during submission.
    for i, (rows, seq_len) in enumerate(zip(rows_per_request, seq_lens)):
        end = start + rows
        tok = torch.arange(start, end, device=pos.device)
        start = end
        lc = seq_len // ratio
        if rows == 0 or lc == 0:
            continue
        lens = compress_lens[tok]
        j = torch.arange(lc, device=pos.device)
        if inputs.req_pool_indices_cpu is not None:
            r = inputs.req_pool_indices_cpu[i]
            slots = req_to_token[r, j * ratio]
        else:
            # Keep the request index one-dimensional: scalar CUDA indices
            # can invoke .item() inside Tensor.__getitem__.
            slots = req_to_token[
                inputs.req_pool_indices[i : i + 1, None], (j * ratio)[None, :]
            ].reshape(-1)
        slots = slots.to(torch.int64) // ratio
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
    # The torch prefill indexer flag asks for the torch oracle, not this kernel.
    fused = (
        q.is_cuda
        and get_platform().is_sm90
        and envs.SGLANG_OPT_USE_FUSED_DSV41_INDEXER_SCORES.get()
        and not envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get()
        and q.dtype == index_k.dtype == weights.dtype == torch.bfloat16
        and q.shape[-1] == 128
    )
    # Chunk rows so the [rows, heads, lc] bf16 scores stay under the budget.
    # The fused kernel holds only the fp32 [rows, lc] scores; at a quarter of
    # the budget its [rows, lc] copies downstream stay under the torch peak.
    row_bytes = lc * 16 if fused else q.shape[1] * lc * 2
    if candidate_blocks is not None and index_k.dtype == torch.bfloat16:
        from sglang.kernels.ops.attention.dsv4.candidate_bf16_mqa import (
            candidate_bf16_mqa_logits,
        )

        # Bound the FP32 output and selection temporaries in candidate space.
        row_bytes = max(1, candidate_blocks.shape[1] * 8 * 32)
    else:
        candidate_blocks = None
    rows_per_chunk = max(1, _TORCH_SCORE_BUDGET_BYTES // row_bytes)
    for start in range(0, request.tok.numel(), rows_per_chunk):
        rows = slice(start, start + rows_per_chunk)
        tok_c, lens_c = request.tok[rows], request.lens[rows]
        blocks = None if candidate_blocks is None else candidate_blocks[tok_c]
        if blocks is not None:
            s = candidate_bf16_mqa_logits(
                q[tok_c], index_k, weights[tok_c], blocks, lens_c
            )
        elif fused:
            s = fused_index_scores(q[tok_c], index_k, weights[tok_c], lens_c)
        else:
            s = indexer.scores(q[tok_c], index_k, weights[tok_c])
            s = s.masked_fill(j[None, :] >= lens_c[:, None], -torch.inf)
        yield ChunkScores(
            rows=rows, tok=tok_c, lens=lens_c, scores=s, candidate_blocks=blocks
        )


def write_prefill(
    out: Selection, request: RequestScores, chunk: ChunkScores, idx: torch.Tensor
) -> None:
    k, lc = request.k, request.lc
    idx = idx.sort(dim=-1).values
    reach = idx < chunk.lens[:, None]
    out.page_indices[chunk.tok, :k] = torch.where(
        reach, request.slots[idx.clamp_max(lc - 1)], -1
    ).to(torch.int32)
    if out.raw_indices is not None:
        out.raw_indices[chunk.tok, :k] = torch.where(reach, idx, -1).to(torch.int32)


class PagedDecodeScores(msgspec.Struct, frozen=True):
    """Paged decode scores with graph-stable logical capacity ``lmax``.

    Only positions below each device-side ``lens`` are initialized. Top-k
    consumers must respect those lengths. With ``has_candidate_mask``, masked
    positions have -inf scores and must be discarded during selection writeback.
    Compact candidates initialize every column; score_lens is their scan width,
    while lens remains the causal bound in logical K coordinates.
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
    candidate_blocks: Optional[torch.Tensor] = None
    score_lens: Optional[torch.Tensor] = None


def decode_scores(
    *,
    inputs: DecodeInputs,
    out: Selection,
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
        out.reset()
        return None
    lens = (pos + 1) // ratio
    metadata = inputs.paged_metadata
    # V4 reserves the replay bound in metadata; visibility stays on device.
    # A capture-time length read would both synchronize and truncate replay.
    lmax = min(metadata.max_compressed_seq_len, req_to_token.shape[1] // ratio)
    if lmax == 0:
        out.reset()
        return None
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    table = pool.get_index_k_with_scale_buffer(inputs.layer_id)
    if paged:
        lens = lens.to(torch.int32)
        if candidate_blocks is not None:
            assert candidate_mask is None
            # Padding/causal holes can precede valid blocks: scan the full width.
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
            plan = plan_topk_v2(lens)
            if (
                envs.SGLANG_OPT_USE_DSV41_SM90_GROUPED_INDEXER.get()
                and inputs.is_verify
                and 2 <= inputs.verify_group_size <= 6
                and bs % inputs.verify_group_size == 0
                and candidate_mask is None
                and q.is_cuda
                and q.shape[1] in (32, 64)
                and q.shape[2] == 128
                and q.dtype == weights.dtype == torch.bfloat16
            ):
                from sglang.kernels.ops.attention.dsv4.sm90_fp4_grouped_indexer import (
                    fp4_index_logits_grouped_sm90,
                )

                scores = fp4_index_logits_grouped_sm90(
                    q,
                    weights,
                    req_to_token,
                    req,
                    lens,
                    table,
                    table.shape[1] // 68,
                    ratio,
                    lmax,
                    inputs.verify_group_size,
                )
            else:
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
    out.reset()
    j = torch.arange(lmax, device=pos.device)
    valid = j[None, :] < lens[:, None]
    slots = req_to_token[req[:, None], (j * ratio)[None, :]].to(torch.int64) // ratio
    slots = slots.masked_fill(~valid, 0)
    scores = fp4_index_logits_decode(
        q, weights, slots, lens, table, table.shape[1] // 68
    )
    return DecodeScores(bs=bs, lmax=lmax, lens=lens, slots=slots, scores=scores)


def select_decode(
    out: Selection, d: DecodeScores | PagedDecodeScores, topk: int
) -> None:
    k = min(topk, d.lmax)
    if isinstance(d, PagedDecodeScores):
        k = min(k, d.scores.shape[1])
        if not k:
            out.reset()
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
            out.page_indices,
            out.raw_indices,
            d.ratio,
            d.has_candidate_mask,
            candidate_blocks=d.candidate_blocks,
        )
    else:
        write_decode(out, d, d.scores.topk(k, dim=-1, sorted=False).indices)


def write_decode(out: Selection, d: DecodeScores, idx: torch.Tensor) -> None:
    k = idx.shape[1]
    idx = idx.sort(dim=-1).values
    reach = idx < d.lens[:, None]
    out.page_indices[: d.bs, :k] = torch.where(
        reach, d.slots.gather(1, idx.clamp_max(d.lmax - 1)), -1
    ).to(torch.int32)
    if out.raw_indices is not None:
        out.raw_indices[: d.bs, :k] = torch.where(reach, idx, -1).to(torch.int32)


def select_compact_prefill(
    inputs: PrefillInputs,
    out: Selection,
    request: RequestScores,
    chunk: ChunkScores,
    req_to_token: torch.Tensor,
) -> None:
    rows, width = chunk.scores.shape
    k = min(request.k, width)
    if not rows or not k:
        return  # prefill_requests already reset the output to -1.
    indices = torch.empty((rows, k), dtype=torch.int32, device=chunk.scores.device)
    scan_lens = torch.full(
        (rows,), width, dtype=torch.int32, device=chunk.scores.device
    )
    topk_transform_ragged_v2(
        chunk.scores,
        scan_lens,
        out_offsets=torch.zeros_like(scan_lens),
        out_indices=indices,
    )
    finish_paged_indexer_topk(
        indices,
        chunk.scores,
        chunk.lens,
        inputs.req_rows[chunk.tok],
        req_to_token,
        out.page_indices,
        out.raw_indices,
        inputs.compress_ratio,
        True,
        candidate_blocks=chunk.candidate_blocks,
        output_rows=chunk.tok,
    )
