"""Index scores shared by the full top-k and both candidate schemes: the DeepGEMM
operands and flattened-K scores, the torch per-request scores, and the writes of
a selection back into pool slots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Iterator, List, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode
from sglang.kernels.ops.attention.dsv4.index_logits import flat_index_logits_tiles

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
    scores: torch.Tensor  # [rows, lc] bf16, -inf past each row's length


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
                indexer=indexer, q=q, weights=weights, index_k=index_k, request=request
            ),
        )


def _score_chunks(
    *,
    indexer: DeepseekV41Indexer,
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    request: RequestScores,
) -> Iterator[ChunkScores]:
    lc, j = request.lc, request.columns
    # Chunk rows so the [rows, heads, lc] bf16 scores stay under the budget.
    rows_per_chunk = max(1, _TORCH_SCORE_BUDGET_BYTES // (q.shape[1] * lc * 2))
    for start in range(0, request.tok.numel(), rows_per_chunk):
        rows = slice(start, start + rows_per_chunk)
        tok_c, lens_c = request.tok[rows], request.lens[rows]
        s = indexer.scores(q[tok_c], index_k, weights[tok_c])
        s = s.masked_fill(j[None, :] >= lens_c[:, None], -torch.inf)
        yield ChunkScores(rows=rows, tok=tok_c, lens=lens_c, scores=s)


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


def decode_scores(
    *,
    inputs: DecodeInputs,
    out: Selection,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
) -> Optional[DecodeScores]:
    """None when there is no row, or nothing visible yet."""
    pool = token_to_kv_pool
    ratio = inputs.compress_ratio
    indexer = inputs.indexer
    req, pos = inputs.req_rows, inputs.positions
    out.reset()
    bs = req.shape[0]
    assert pos.shape[0] == bs, (
        f"decode expects one token per request, {pos.shape=} {bs=}"
    )
    if bs == 0:
        return None
    lens = (pos + 1) // ratio
    metadata = inputs.paged_metadata
    # V4 reserves the replay bound in metadata; visibility stays on device.
    # A capture-time length read would both synchronize and truncate replay.
    lmax = min(metadata.max_compressed_seq_len, req_to_token.shape[1] // ratio)
    if lmax == 0:
        return None
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    j = torch.arange(lmax, device=pos.device)
    valid = j[None, :] < lens[:, None]
    slots = req_to_token[req[:, None], (j * ratio)[None, :]].to(torch.int64) // ratio
    slots = slots.masked_fill(~valid, 0)
    table = pool.get_index_k_with_scale_buffer(inputs.layer_id)
    scores = fp4_index_logits_decode(
        q, weights, slots, lens, table, table.shape[1] // 68
    )
    return DecodeScores(bs=bs, lmax=lmax, lens=lens, slots=slots, scores=scores)


def write_decode(out: Selection, d: DecodeScores, idx: torch.Tensor) -> None:
    k = idx.shape[1]
    idx = idx.sort(dim=-1).values
    reach = idx < d.lens[:, None]
    out.page_indices[: d.bs, :k] = torch.where(
        reach, d.slots.gather(1, idx.clamp_max(d.lmax - 1)), -1
    ).to(torch.int32)
    if out.raw_indices is not None:
        out.raw_indices[: d.bs, :k] = torch.where(reach, idx, -1).to(torch.int32)
