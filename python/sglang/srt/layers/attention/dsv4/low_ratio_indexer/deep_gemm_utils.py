"""DeepGEMM scoring shared by the dense layers and the candidate scheme.

On prefill: the chunk's index-K slots, fp4 queries and head weights, the tiled
``fp8_fp4_mqa_logits`` scores over the flattened index K, the plain top-k, and
the write of a selection back into the pool-slot layout attention reads. On
decode: the fp4 queries and the paged pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.layers.attention.mqa_logits_utils import (
    mqa_logits_row_bytes,
    mqa_logits_rows_per_chunk,
)
from sglang.srt.utils.common import ceil_align

from .inputs import (
    DecodeInputs,
    PrefillInputs,
    Selection,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

# TODO: use a per-forward mqa_logits_budget_bytes() budget that also
# leaves room for candidate masks and block-selection scratch.
_SCORE_BUDGET_BYTES = 2 << 30


class DeepGEMMDecodeData(msgspec.Struct, frozen=True):
    q_fp4: torch.Tensor  # [rows, 1, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, 1, heads] int32, packed ue8m0
    weights: torch.Tensor  # [rows, heads] bf16/fp32 head weights
    k_cache: torch.Tensor  # [pages, page_size, 1, 68] uint8, the layer's index-K pool


class DeepGEMMPrefillData(msgspec.Struct, frozen=True):
    """One index layer's operands on a prefill chunk; the rows of one request are
    consecutive, and a selection is a flattened-K column, request start plus
    compressed position."""

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

    def empty_selection(self, topk: int, init: bool = True) -> torch.Tensor:
        out = self.request_starts.new_empty((self.num_rows, topk), dtype=torch.int32)
        if init:
            out.fill_(-1)
        return out

    def write_selection(self, selected: torch.Tensor, out: Selection) -> None:
        # write_prefill_selection(out=out, data=self, selected=selected)
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
    """``q`` ``[rows, heads, 128]`` bf16 to fp4 ``[rows, heads, 64]`` int8 and its
    scales ``[rows, heads]`` int32 (packed ue8m0)."""
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


def get_flat_index_k(
    *,
    data: DeepGEMMPrefillData,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    layer_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The chunk's index K gathered in flattened-K column order: int8
    ``[columns, 64]`` and int32 ``[columns]``; a copy, for dense scoring only."""
    return token_to_kv_pool.get_low_ratio_index_k_fp4(layer_id, data.k_slots)


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
    """None when the chunk has no row or no visible compressed position, which is
    the same for every index layer of the forward."""
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
    # Visible compressed positions per request at its newest token; the
    # per-token count (pos + 1) // ratio bounds each row below.
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
    """The dense scores of ``data`` row tile by row tile under the budget:
    ``(rows, logits)`` with fp32 ``logits[i, j]`` the score of query row
    ``rows.start + i`` against ``kv[request_starts + j]``, garbage past the row's
    ``compress_lens``; the width is the batch's largest context aligned to
    ``width_align`` columns."""
    from deep_gemm import fp8_fp4_mqa_logits

    rows = data.num_rows
    width = ceil_align(max(data.lens_per_request, default=0), width_align)
    if rows == 0 or width == 0:
        return
    rows_per_chunk = _rows_per_chunk(rows, width, heads=data.q_fp4.shape[1])
    for offset in range(0, rows, rows_per_chunk):
        tile = slice(offset, min(offset + rows_per_chunk, rows))
        starts = data.request_starts[tile]
        yield (
            tile,
            fp8_fp4_mqa_logits(
                (data.q_fp4[tile], data.q_sf[tile]),
                kv,
                data.weights[tile],
                starts,
                starts + data.compress_lens[tile],
                False,
                width,
            ),
        )


def dense_prefill_topk(
    data: DeepGEMMPrefillData,
    kv: Tuple[torch.Tensor, torch.Tensor],
    *,
    topk: int,
) -> torch.Tensor:
    """The plain top-``topk`` of each row as flattened-K columns, ``-1`` padded,
    unordered."""
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


# TODO(dark): make it SM aware; align to SM count to avoid wave quantization.
def _rows_per_chunk(rows: int, width: int, *, heads: int) -> int:
    """Query rows per logits tile so one fp32 [rows, width] tile fits the
    budget; the row count stays a multiple of the kernel's row alignment."""
    row_alignment = 128 // heads
    rows_per_chunk = mqa_logits_rows_per_chunk(
        num_rows=ceil_align(rows, row_alignment),
        row_bytes=mqa_logits_row_bytes(width),
        budget_bytes=_SCORE_BUDGET_BYTES,
    )
    if rows_per_chunk is None:
        return rows
    return max(row_alignment, rows_per_chunk // row_alignment * row_alignment)
