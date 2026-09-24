"""Torch scoring shared by the dense layers and the candidate scheme: bf16 scores
per request on prefill, the fp4 decode logits kernel on decode, and the writes of
a ``torch.topk`` back into the pool-slot layout attention reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode

from .inputs import (
    DecodeInputs,
    PrefillInputs,
    Selection,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


# Arbitrary cap on one bf16 [rows, heads, lc] score chunk; transients run ~3x this.
_SCORE_BUDGET_BYTES = 1 << 30


class RequestScores(msgspec.Struct, frozen=True):
    """One request of a prefill chunk, scored row chunk by row chunk."""

    index: int  # position among the chunk's requests, empty ones included
    lc: int  # compressed positions visible at its newest row; 0 scores nothing
    k: int
    columns: Optional[torch.Tensor] = None  # arange(lc)
    tok: Optional[torch.Tensor] = None  # [rows_b] its query rows in the chunk
    lens: Optional[torch.Tensor] = None  # [rows_b] compressed positions each row sees
    slots: Optional[torch.Tensor] = None  # [lc] index-K pool slot of each position


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
    """Each request of the chunk with its row chunks' scores; an empty request
    comes with no chunk."""
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
    for b, r in enumerate(torch.unique_consecutive(req).tolist()):
        tok = (req == r).nonzero().squeeze(1)
        lens = compress_lens[tok]
        lc = int(lens.max().item())
        if lc == 0:
            yield RequestScores(index=b, lc=0, k=0), iter(())
            continue
        j = torch.arange(lc, device=pos.device)
        slots = req_to_token[r, j * ratio].to(torch.int64) // ratio
        # Dequantize only this request's visible K rows; the table is pool-sized.
        index_k = pool.get_low_ratio_index_k_dequant(inputs.layer_id, slots)
        request = RequestScores(
            index=b, lc=lc, k=min(topk, lc), columns=j, tok=tok, lens=lens, slots=slots
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
    # Every step is per query row; chunk rows so the [rows, heads, lc] bf16
    # scores stay under the budget (16 GiB at once for a 16k-token prompt).
    rows_per_chunk = max(1, _SCORE_BUDGET_BYTES // (q.shape[1] * lc * 2))
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
