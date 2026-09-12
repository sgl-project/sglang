"""DeepSeek V4.1 low-ratio (1 / 2) indexer on ROCm: FlyDSL fp4 paged MQA logits over the split
payload / scale index-K pools, then the AOT top-k transform -- the DeepGEMM path's contract."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4 import topk_transform_paged
from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
    LOW_RATIO_PAGE_TABLE_BUCKET,
    FP4DecodeWorkspace,
    FP4PrefillWorkspace,
    aiter_fp4_paged_mqa_logits,
    index_q_pack_weights_hip,
    logits_rows_per_chunk,
    pack_fp4_query_flydsl,
    prepare_fp4_decode_workspace,
    prepare_fp4_prefill_workspace,
    rocm_indexer_head_weights,
    sort_selection_rows,
)
from sglang.srt.layers.attention.deepseek_v4_backend import (
    _TORCH_INDEXER_SCORE_BUDGET_BYTES,
    _as_int_list,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# the k the AOT fast_topk op (topk_hip.hip) is instantiated for; any other k takes torch.topk
_AOT_FAST_TOPK_K = 2048


@functools.lru_cache(maxsize=1)
def _aot_topk_sorts_output() -> bool:
    """Whether the installed AOT top-k transform takes ``sort_output`` (orders each
    row in its epilogue). An older sgl_kernel build falls back to the sort launch."""
    import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel

    schema = torch.ops.sgl_kernel.deepseek_v4_topk_transform_512.default._schema
    supported = any(arg.name == "sort_output" for arg in schema.arguments)
    if not supported:
        logger.warning(
            "sgl_kernel's deepseek_v4_topk_transform_512 predates sort_output; the "
            "low-ratio indexer keeps a separate sort launch per layer"
        )
    return supported


def topk_transform_paged_sorted(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    raw_indices: Optional[torch.Tensor],
) -> None:
    """``topk_transform_paged`` followed by ``sort_selection_rows``: the -1 padded
    paged top-k of every row, ascending by position. One launch when the AOT
    kernel sorts in its epilogue (bitwise the same rows)."""
    if _aot_topk_sorts_output():
        torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
            scores, seq_lens, page_table, page_indices, page_size, raw_indices, True
        )
        return
    topk_transform_paged(
        scores, seq_lens, page_table, page_indices, page_size, raw_indices
    )
    sort_selection_rows(page_indices, raw_indices)


# Candidate blocks one Triton program reduces; times block_size positions of logits.
_LEVEL_ONE_BLOCKS_PER_PROGRAM = 256


@triton.jit
def _candidate_block_scores_kernel(
    logits_ptr,
    seq_lens_ptr,
    out_ptr,
    logits_stride,
    width,
    num_blocks,
    out_stride,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    FILL_TAIL: tl.constexpr,
):
    """out[row, blk] = max(logits[row, blk * BLOCK_SIZE : (blk + 1) * BLOCK_SIZE])
    over the positions < seq_lens[row]; +inf for the block holding the newest
    position, -inf for blocks past the reach (written only with FILL_TAIL). A
    program whose blocks all lie past the reach reads no logits."""
    row = tl.program_id(0)
    block0 = tl.program_id(1) * BLOCKS_PER_PROGRAM
    length = tl.load(seq_lens_ptr + row)
    blocks = block0 + tl.arange(0, BLOCKS_PER_PROGRAM)
    in_table = blocks < num_blocks
    if block0 * BLOCK_SIZE >= length:
        if FILL_TAIL:
            tl.store(out_ptr + row * out_stride + blocks, float("-inf"), mask=in_table)
        return
    cols = blocks[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    vals = tl.load(
        logits_ptr + row * logits_stride + cols,
        mask=(cols < length) & (cols < width),
        other=float("-inf"),
    )
    scores = tl.max(vals, axis=1)
    last = (length - 1) // BLOCK_SIZE
    scores = tl.where(blocks == last, float("inf"), scores)
    tl.store(out_ptr + row * out_stride + blocks, scores, mask=in_table)


@triton.jit
def _gather_candidate_blocks_kernel(
    logits_ptr,
    seq_lens_ptr,
    ids_ptr,
    out_ptr,
    logits_stride,
    width,
    ids_stride,
    out_stride,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    """out[row, j * BLOCK_SIZE + t] = logits[row, ids[row, j] * BLOCK_SIZE + t] for
    the reachable positions of the kept blocks, -inf elsewhere (an id of -1, or a
    position past the reach in the newest block)."""
    row = tl.program_id(0)
    j0 = tl.program_id(1) * BLOCKS_PER_PROGRAM
    length = tl.load(seq_lens_ptr + row)
    j = j0 + tl.arange(0, BLOCKS_PER_PROGRAM)
    ids = tl.load(ids_ptr + row * ids_stride + j)
    t = tl.arange(0, BLOCK_SIZE)[None, :]
    cols = ids[:, None] * BLOCK_SIZE + t
    vals = tl.load(
        logits_ptr + row * logits_stride + cols,
        mask=(ids[:, None] >= 0) & (cols < length) & (cols < width),
        other=float("-inf"),
    )
    tl.store(out_ptr + row * out_stride + j[:, None] * BLOCK_SIZE + t, vals)


def _num_candidate_blocks(width: int, block_size: int) -> int:
    return (width + block_size - 1) // block_size


def candidate_block_scores(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int,
    fill_tail: bool,
) -> torch.Tensor:
    """[rows, num_blocks] fp32 block maxima of `logits` over the reachable positions
    (see `_candidate_block_scores_kernel`). `seq_lens` int32 [rows], contiguous."""
    assert logits.dim() == 2 and logits.dtype == torch.float32 and logits.stride(1) == 1
    assert block_size & (block_size - 1) == 0, f"{block_size = } must be a power of 2"
    rows, width = logits.shape
    num_blocks = _num_candidate_blocks(width, block_size)
    scores = torch.empty((rows, num_blocks), dtype=torch.float32, device=logits.device)
    grid = (rows, triton.cdiv(num_blocks, _LEVEL_ONE_BLOCKS_PER_PROGRAM))
    _candidate_block_scores_kernel[grid](
        logits,
        seq_lens,
        scores,
        logits.stride(0),
        width,
        num_blocks,
        scores.stride(0),
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=_LEVEL_ONE_BLOCKS_PER_PROGRAM,
        FILL_TAIL=fill_tail,
    )
    return scores


@triton.jit
def _candidate_lengths_kernel(
    seq_lens_ptr,
    block_lens_ptr,
    compact_lens_ptr,
    rows,
    topk_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """block_lens = ceil(len / BLOCK_SIZE): the blocks with a reachable position;
    compact_lens = min(block_lens, topk_blocks) * BLOCK_SIZE: the width of the
    compact candidate row (every block is kept while there are at most
    topk_blocks of them)."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = r < rows
    length = tl.load(seq_lens_ptr + r, mask=mask, other=0)
    block_lens = (length + (BLOCK_SIZE - 1)) // BLOCK_SIZE
    tl.store(block_lens_ptr + r, block_lens, mask=mask)
    tl.store(
        compact_lens_ptr + r,
        tl.minimum(block_lens, topk_blocks) * BLOCK_SIZE,
        mask=mask,
    )


@triton.jit
def _map_compact_selection_kernel(
    compact_pos_ptr,
    ids_ptr,
    seq_lens_ptr,
    page_table_ptr,
    page_indices_ptr,
    raw_indices_ptr,
    ids_stride,
    pt_stride,
    out_stride,
    n_pages,
    BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    WRITE_RAW: tl.constexpr,
    SORT: tl.constexpr,
    PAD_KEY: tl.constexpr,
):
    """Compact position c -> real position ids[c // BLOCK_SIZE] * BLOCK_SIZE + c % BLOCK_SIZE ->
    slot via the row's page table; reachable selections packed first, -1 after. With SORT the valid
    prefix is ascending (the key rule of ``_sort_selection_rows_kernel``)."""
    row = tl.program_id(0)
    length = tl.load(seq_lens_ptr + row)
    j = tl.arange(0, BLOCK)
    in_k = j < TOPK
    c = tl.load(compact_pos_ptr + row * TOPK + j, mask=in_k, other=-1)
    valid = c >= 0
    cc = tl.where(valid, c, 0)
    blk = tl.load(ids_ptr + row * ids_stride + cc // BLOCK_SIZE)
    real = blk * BLOCK_SIZE + cc % BLOCK_SIZE
    valid = valid & (blk >= 0) & (real < length)
    page = tl.load(
        page_table_ptr + row * pt_stride + tl.minimum(real // PAGE_SIZE, n_pages - 1),
        mask=valid,
        other=0,
    )
    slot = page * PAGE_SIZE + real % PAGE_SIZE
    if SORT:
        # BLOCK == TOPK (power of two): the sort key is the position with raw indices, else the slot
        if WRITE_RAW:
            hi = real
        else:
            hi = slot
        key = tl.where(valid, hi, PAD_KEY).to(tl.int64) << 32
        key = tl.sort(
            key | (tl.where(valid, slot, -1).to(tl.int64) & 0xFFFFFFFF), dim=0
        )
        pad = (key >> 32) == PAD_KEY
        tl.store(
            page_indices_ptr + row * out_stride + j,
            tl.where(pad, -1, (key & 0xFFFFFFFF).to(tl.int32)),
        )
        if WRITE_RAW:
            tl.store(
                raw_indices_ptr + row * out_stride + j,
                tl.where(pad, -1, (key >> 32).to(tl.int32)),
            )
    else:
        v = valid.to(tl.int32)
        count = tl.sum(v, axis=0)
        wpos = tl.cumsum(v, axis=0) - 1
        # The packed entries and the padding never share an address.
        tl.store(page_indices_ptr + row * out_stride + j, -1, mask=in_k & (j >= count))
        tl.store(page_indices_ptr + row * out_stride + wpos, slot, mask=valid)
        if WRITE_RAW:
            tl.store(
                raw_indices_ptr + row * out_stride + j, -1, mask=in_k & (j >= count)
            )
            tl.store(raw_indices_ptr + row * out_stride + wpos, real, mask=valid)


class CandidateBlocks(NamedTuple):
    """What the candidate-source layer publishes for the decode rows of a step."""

    # int32 [rows, topk_blocks]: kept block ids in no particular order, -1 padded after the last
    ids: torch.Tensor
    # int32 [rows]: kept blocks * block_size, the width of the compact row.
    compact_lens: torch.Tensor
    # int32 [rows, 1] zeros; with page_size = compact_page_size a position maps to itself
    compact_page_table: torch.Tensor
    compact_page_size: int
    block_size: int


def slice_candidate_blocks(candidates: CandidateBlocks, rows: slice) -> CandidateBlocks:
    """The rows `rows` of a per-request publication."""
    return candidates._replace(
        ids=candidates.ids[rows],
        compact_lens=candidates.compact_lens[rows],
        compact_page_table=candidates.compact_page_table[rows],
    )


def cat_candidate_blocks(pieces: List[CandidateBlocks]) -> CandidateBlocks:
    """Row chunks of one request's publication, in row order."""
    if len(pieces) == 1:
        return pieces[0]
    return pieces[0]._replace(
        ids=torch.cat([p.ids for p in pieces]),
        compact_lens=torch.cat([p.compact_lens for p in pieces]),
        compact_page_table=torch.cat([p.compact_page_table for p in pieces]),
    )


def select_candidate_blocks_hip(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
) -> CandidateBlocks:
    """Level one over a [rows, capacity] logits rectangle, bounded by `seq_lens`: the `topk_blocks`
    best blocks of each row, the block with its newest position always among them; as a set per
    row the ids equal the reference's `select_candidate_blocks`. Graph-safe: no host sync."""
    seq_lens = seq_lens.to(torch.int32).contiguous()
    rows, width = logits.shape
    device = logits.device
    num_blocks = _num_candidate_blocks(width, block_size)
    use_aot = topk_blocks == _AOT_FAST_TOPK_K
    scores = candidate_block_scores(
        logits, seq_lens, block_size=block_size, fill_tail=not use_aot
    )
    block_lens = torch.empty(rows, dtype=torch.int32, device=device)
    compact_lens = torch.empty(rows, dtype=torch.int32, device=device)
    _candidate_lengths_kernel[(triton.cdiv(rows, 1024),)](
        seq_lens,
        block_lens,
        compact_lens,
        rows,
        topk_blocks,
        BLOCK_SIZE=block_size,
        BLOCK=1024,
    )
    if use_aot:
        # exact top-k over the first ceil(len / block_size) block scores of each row, -1 padded
        ids = torch.empty((rows, topk_blocks), dtype=torch.int32, device=device)
        torch.ops.sgl_kernel.fast_topk(scores, ids, block_lens, None)
    else:
        picked = scores.topk(min(topk_blocks, num_blocks), dim=-1)
        ids = picked.indices.to(torch.int32).masked_fill(
            picked.values == -torch.inf, -1
        )
        # the gather takes the width from ids and tiles it in 256-block programs
        ids = F.pad(ids, (0, topk_blocks - ids.shape[1]), value=-1)
    compact_width = topk_blocks * block_size
    return CandidateBlocks(
        ids=ids,
        compact_lens=compact_lens,
        compact_page_table=torch.zeros((rows, 1), dtype=torch.int32, device=device),
        compact_page_size=triton.next_power_of_2(compact_width),
        block_size=block_size,
    )


def gather_candidate_blocks(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    ids: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """The compact [rows, topk_blocks * block_size] fp32 row of each request's
    candidate positions in id order (see `_gather_candidate_blocks_kernel`)."""
    rows, width = logits.shape
    topk_blocks = ids.shape[1]
    assert ids.shape[0] == rows and ids.dtype == torch.int32 and ids.stride(1) == 1
    per_program = min(topk_blocks, _LEVEL_ONE_BLOCKS_PER_PROGRAM)
    assert topk_blocks % per_program == 0, topk_blocks
    compact = torch.empty(
        (rows, topk_blocks * block_size), dtype=torch.float32, device=logits.device
    )
    grid = (rows, topk_blocks // per_program)
    _gather_candidate_blocks_kernel[grid](
        logits,
        seq_lens.to(torch.int32).contiguous(),
        ids,
        compact,
        logits.stride(0),
        width,
        ids.stride(0),
        compact.stride(0),
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=per_program,
    )
    return compact


def topk_within_candidate_blocks_hip(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    candidates: CandidateBlocks,
    *,
    page_table: torch.Tensor,
    page_size: int,
    page_indices: torch.Tensor,
    raw_indices: Optional[torch.Tensor],
    sort_output: bool = False,
) -> None:
    """Level two for a consumer layer: the top-k of ``logits`` inside the published candidate
    blocks, written as the paged transform writes it (-1 padded, valid prefix first; ascending with
    ``sort_output``, k a power of two). Runs on the compact row, so the cost stops growing with context."""
    rows, width = logits.shape
    topk = page_indices.shape[1]
    block_size = candidates.block_size
    seq_lens = seq_lens.to(torch.int32).contiguous()
    compact = gather_candidate_blocks(
        logits, seq_lens, candidates.ids, block_size=block_size
    )
    assert compact.shape[1] <= candidates.compact_page_size
    compact_pos = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    topk_transform_paged(
        compact,
        candidates.compact_lens,
        candidates.compact_page_table,
        compact_pos,
        candidates.compact_page_size,
        None,
    )
    assert page_indices.stride(1) == 1 and page_indices.shape == (rows, topk)
    assert page_table.stride(1) == 1 and page_table.shape[0] == rows
    write_raw = raw_indices is not None
    if write_raw:
        assert raw_indices.shape == (rows, topk) and raw_indices.stride(1) == 1
        assert raw_indices.stride(0) == page_indices.stride(0)
    assert not sort_output or topk & (topk - 1) == 0, topk
    _map_compact_selection_kernel[(rows,)](
        compact_pos,
        candidates.ids,
        seq_lens,
        page_table,
        page_indices,
        raw_indices if write_raw else page_indices,
        candidates.ids.stride(0),
        page_table.stride(0),
        page_indices.stride(0),
        page_table.shape[1],
        BLOCK_SIZE=block_size,
        PAGE_SIZE=page_size,
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
        WRITE_RAW=write_raw,
        SORT=sort_output,
        PAD_KEY=torch.iinfo(torch.int32).max,
    )


def _gemv_head_weight_rows(indexer, x: torch.Tensor) -> bool:
    """Row counts `rocm_indexer_head_weights` / the split-K GEMV serve: a small contiguous bf16 batch."""
    return (
        0 < x.shape[0] <= indexer.weights_proj_hip_max_tokens
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
    )


def _indexer_head_weights(indexer, x: torch.Tensor) -> torch.Tensor:
    """`indexer.head_weights(x)` as the contiguous bf16 [T, H] the FlyDSL kernels take; decode row
    counts run `rocm_indexer_head_weights` (same two roundings as aiter's GEMM plus the scale)."""
    if _gemv_head_weight_rows(indexer, x):
        return rocm_indexer_head_weights(
            x, indexer.weights_proj.weight, indexer.head_weight_scale
        )
    return indexer.head_weights(x).contiguous()


def _indexer_inputs(layer, x, q_lora, pos):
    """(payload, scale) query in the FlyDSL layout and the pre-scaled head weights."""
    indexer = layer.indexer
    # The kernel sums head scores locally, so the indexer heads must be replicated.
    assert indexer.n_local_heads == indexer.n_heads
    if (
        _gemv_head_weight_rows(indexer, x)
        and indexer.n_heads % 16 == 0
        and indexer.n_heads <= 64
        and indexer.index_head_dim == 128
        and layer.freqs_cis.dtype == torch.complex64
    ):
        # decode rows: wq_b, split-K head-weight GEMV, then one launch for RoPE, fp4 pack and reduce
        from sglang.kernels.ops.moe.rocm_router_gate import rocm_router_gemv_split_k

        q, _ = indexer.wq_b(q_lora)
        partials = rocm_router_gemv_split_k(x, indexer.weights_proj.weight)
        return index_q_pack_weights_hip(
            q,
            layer.freqs_cis,
            pos,
            indexer.rope_head_dim,
            partials,
            indexer.head_weight_scale,
            num_heads=indexer.n_heads,
        )
    # [T, H, 128] fp4 grid; the RoPE launch gathers freqs_cis[pos] itself
    q = indexer.queries(q_lora, layer.freqs_cis, positions=pos)
    q_fp4, q_scale = pack_fp4_query_flydsl(q)
    weights = _indexer_head_weights(indexer, x)  # [T, H] bf16, already scaled
    return q_fp4, q_scale, weights


def build_low_ratio_decode_workspaces(
    metadata_by_ratio: Dict[int, PagedIndexerMetadata],
) -> Dict[int, FP4DecodeWorkspace]:
    """Capture-safe: everything the schedule kernel touches is pinned in the workspace."""
    return {
        ratio: prepare_fp4_decode_workspace(
            meta.page_table, meta.c4_seq_lens, bucket=LOW_RATIO_PAGE_TABLE_BUCKET
        )
        for ratio, meta in metadata_by_ratio.items()
    }


def refresh_low_ratio_prefill_workspaces(
    metadata_by_ratio: Dict[int, PagedIndexerMetadata],
    previous: Optional[Dict[int, FP4PrefillWorkspace]],
) -> Dict[int, FP4PrefillWorkspace]:
    """Must run outside CUDA-graph capture; see `prepare_fp4_prefill_workspace`."""
    previous = previous or {}
    return {
        ratio: prepare_fp4_prefill_workspace(
            meta.page_table,
            meta.c4_seq_lens,
            workspace=previous.get(ratio),
            bucket=LOW_RATIO_PAGE_TABLE_BUCKET,
        )
        for ratio, meta in metadata_by_ratio.items()
    }


def low_ratio_identity_skip_enabled(
    *, index_topk: int, candidate_topk_blocks: int, candidate_block_size: int
) -> bool:
    """Whether identity requests may bypass scoring under this model config: the score-free
    candidate mask is right only while <= index_topk positions fit the candidate top-k."""
    return -(-index_topk // max(candidate_block_size, 1)) <= candidate_topk_blocks


def is_identity_request(seq_len: int, ratio: int, index_topk: int) -> bool:
    """Every row of the request selects all its visible compressed positions."""
    return seq_len // ratio <= index_topk


def _fill_identity_request(
    indexer,
    *,
    lc: int,
    slots_j: torch.Tensor,
    lens: torch.Tensor,
    page_rows: torch.Tensor,
    raw_rows: Optional[torch.Tensor],
) -> None:
    """Rows [t, topk] of the -1 filled buffers: column j < lc holds compressed
    position j (its slot / its raw index) when j < lens_row, else -1. An identity
    request publishes no candidates: every reachable block is one."""
    j = torch.arange(lc, device=lens.device)
    reach = j[None, :] < lens[:, None]
    page_rows[:, :lc] = torch.where(reach, slots_j[None, :], -1).to(torch.int32)
    if raw_rows is not None:
        raw_rows[:, :lc] = torch.where(reach, j[None, :], -1).to(torch.int32)


def _decode_batch_max_seq_len(forward_batch) -> Optional[int]:
    """The host-side longest context of a decode batch, None when unknown."""
    seq_lens_cpu = forward_batch.seq_lens_cpu
    if seq_lens_cpu is None:
        return None
    if torch.is_tensor(seq_lens_cpu):
        if seq_lens_cpu.numel() == 0:
            return None
        return int(seq_lens_cpu.max().item())
    if len(seq_lens_cpu) == 0:
        return None
    return int(max(seq_lens_cpu))


def low_ratio_decode_rows_are_identity(backend, forward_batch, ratio: int) -> bool:
    """Decode: every row's compressed context fits index_topk, so nothing needs scoring. Inside a
    captured graph the answer is the variant being captured; eagerly it is the host-side batch
    maximum. Target-verify rows keep the scored path."""
    if not backend.low_ratio_identity_skip or forward_batch is None:
        return False
    if not forward_batch.forward_mode.is_decode():
        return False
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
        skip_low_ratio_indexer,
    )

    if get_is_capture_mode():
        return skip_low_ratio_indexer(ratio)
    max_len = _decode_batch_max_seq_len(forward_batch)
    return max_len is not None and max_len // ratio <= backend.index_topk


def low_ratio_candidate_span(hf_text_config) -> Optional[int]:
    """The span (in positions) within which every block of a request is a
    candidate, when the model has a candidate source; None otherwise."""
    # optional HF-config keys: a model without a candidate source lacks them
    if getattr(hf_text_config, "candidate_source_layer_id", -1) < 0:
        return None
    span = getattr(hf_text_config, "candidate_topk_blocks", 0) * getattr(
        hf_text_config, "candidate_block_size", 0
    )
    return span if span > 0 else None


def low_ratio_decode_rows_fit_candidate_span(backend, forward_batch) -> bool:
    """Decode: every request's context fits the candidate span, so two-level top-k equals the plain
    paged top-k. Captured: the variant being captured; eager: the host-side batch maximum."""
    span = backend.low_ratio_candidate_span
    if span is None or forward_batch is None:
        return False
    if not forward_batch.forward_mode.is_decode():
        return False
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_dsa_variant,
        get_is_capture_mode,
    )

    if get_is_capture_mode():
        return get_capture_dsa_variant() in (
            "candidate_all",
            "candidate_c2_all",
            "candidate_unfiltered",
        )
    max_len = _decode_batch_max_seq_len(forward_batch)
    return max_len is not None and max_len <= span


def low_ratio_index_topk_hip_decode(
    backend, layer, x, q_lora, pos, forward_batch: ForwardBatch
) -> None:
    """One token per request: paged fp4 logits over every visible compressed slot, level-one
    candidate blocks where the layer publishes or consumes them, then the top-k transform
    writes the -1 padded page indices."""
    pool = backend.token_to_kv_pool
    metadata = backend.forward_metadata
    core = metadata.core_metadata
    ratio = layer.compress_ratio
    indexer = layer.indexer
    indexer_metadata = metadata.low_ratio_indexer_metadata(ratio)
    assert indexer_metadata is not None, f"no decode indexer metadata for {ratio = }"
    page_indices = core.sparse_page_indices(ratio)
    raw_indices = core.sparse_raw_indices(ratio)

    if low_ratio_decode_rows_are_identity(backend, forward_batch, ratio):
        # every row takes the transform's sequential branch (len <= topk), which reads no scores
        scores = torch.empty(
            (page_indices.shape[0], 1), dtype=torch.float32, device=pos.device
        )
        topk_transform_paged(
            scores,
            indexer_metadata.c4_seq_lens,
            indexer_metadata.page_table,
            page_indices,
            indexer_metadata.c4_page_size,
            raw_indices,
        )
        return

    two_level = indexer.is_candidate_source or indexer.uses_candidates
    if two_level and low_ratio_decode_rows_fit_candidate_span(backend, forward_batch):
        # every block is a candidate: the source publishes nothing, every layer runs the plain top-k
        if indexer.is_candidate_source:
            backend.candidate_masks = None
        two_level = False

    q_fp4, q_scale, weights = _indexer_inputs(layer, x, q_lora, pos)
    logits = aiter_fp4_paged_mqa_logits(
        q_fp4=q_fp4,
        q_scale=q_scale,
        k_payload=pool.get_index_k_fp4_payload_buffer(layer.layer_id),
        k_scale=pool.get_index_k_fp4_scale_buffer(layer.layer_id),
        weights=weights,
        page_table=indexer_metadata.page_table,
        c4_seq_lens=indexer_metadata.c4_seq_lens,
        weight_scale=1.0,
        page_table_bucket=LOW_RATIO_PAGE_TABLE_BUCKET,
        is_decode=True,
        decode_workspace=metadata.fp4_low_ratio_decode_workspaces.get(ratio),
    )
    # level one is bounded on device by the compressed lengths: a captured step cannot read them back
    if two_level and indexer.uses_candidates:
        candidates = backend.candidate_masks
        assert (
            isinstance(candidates, CandidateBlocks)
            and candidates.ids.shape[0] == logits.shape[0]
            and candidates.block_size == indexer.candidate_block_size
        ), "candidate blocks missing for decode"
        topk_within_candidate_blocks_hip(
            logits,
            indexer_metadata.c4_seq_lens,
            candidates,
            page_table=indexer_metadata.page_table,
            page_size=indexer_metadata.c4_page_size,
            page_indices=core.sparse_page_indices(ratio),
            raw_indices=core.sparse_raw_indices(ratio),
            sort_output=True,
        )
        return
    if two_level and indexer.is_candidate_source:
        backend.candidate_masks = select_candidate_blocks_hip(
            logits,
            indexer_metadata.c4_seq_lens,
            topk_blocks=indexer.candidate_topk_blocks,
            block_size=indexer.candidate_block_size,
        )
    topk_transform_paged_sorted(
        logits,
        indexer_metadata.c4_seq_lens,
        indexer_metadata.page_table,
        page_indices,
        indexer_metadata.c4_page_size,
        raw_indices,
    )


def _extend_k_slots(req_to_token, *, ratio, lc_per_req, req_pool_indices, device):
    """Per request, the c1/c2 pool slots of its visible compressed positions, and
    each request's start offset in their concatenation."""
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
    return slot_chunks, starts


def store_index_k_norm_rope_split(pool, layer, latent, pos, out_loc, freqs_cis) -> None:
    """The index-K store of ``DeepseekV4AttnBackend._low_ratio_compress_fused`` in the FlyDSL
    split payload / scale layout (same bytes as ``store_fp4_index_k_cache_split``); ``out_loc``
    is -1 for an incomplete group and 0 for padding, and the kernel stores neither."""
    from sglang.kernels.ops.attention.dsv4.fp4_rope_hip import (
        index_k_norm_rope_pack_store_split,
    )

    indexer = layer.indexer
    layer_id = layer.layer_id
    index_k_norm_rope_pack_store_split(
        indexer.forward_wk(latent),
        indexer.k_norm.weight.data,
        indexer.k_norm.eps,
        freqs_cis,
        pos,
        out_loc,
        pool.get_index_k_fp4_payload_buffer(layer_id),
        pool.get_index_k_fp4_scale_buffer(layer_id),
        ratio=layer.compress_ratio,
    )


def low_ratio_index_topk_hip_extend(
    backend, layer, x, q_lora, pos, forward_batch: ForwardBatch
) -> None:
    """Ragged prefill: the FlyDSL prefill kernel scores every token's visible compressed positions,
    candidate masks are published or applied per request, one paged top-k selects every row.
    Identity requests are written without scores; an all-identity batch skips the kernel."""
    pool = backend.token_to_kv_pool
    metadata = backend.forward_metadata
    core = metadata.core_metadata
    ratio = layer.compress_ratio
    indexer = layer.indexer
    page_indices = core.sparse_page_indices(ratio)
    raw_indices = core.sparse_raw_indices(ratio)
    page_indices.fill_(-1)
    if raw_indices is not None:
        raw_indices.fill_(-1)

    seq_lens_cpu = _as_int_list(forward_batch.seq_lens_cpu)
    # under decoder SWA bounded replay the late layers score each request's tail rows only
    tail = metadata.late_layer_tail
    extend_lens_cpu = (
        tail.extend_seq_lens_cpu
        if tail is not None
        else _as_int_list(forward_batch.extend_seq_lens_cpu)
    )
    assert seq_lens_cpu is not None and extend_lens_cpu is not None
    lc_per_req = [s // ratio for s in seq_lens_cpu]
    if not any(lc_per_req):
        if indexer.is_candidate_source:
            backend.candidate_masks = []
        return

    if backend.low_ratio_identity_skip:
        is_identity = [
            is_identity_request(s, ratio, indexer.index_topk) for s in seq_lens_cpu
        ]
    else:
        is_identity = [False] * len(seq_lens_cpu)
    compress_lens = ((pos + 1) // ratio).to(torch.int32)
    consume = backend.candidate_masks if indexer.uses_candidates else None
    publish = [] if indexer.is_candidate_source else None
    if any(is_identity):
        # only the score-free fill resolves slots itself; scored rows resolve in the top-k transform
        slot_chunks, starts = _extend_k_slots(
            backend.req_to_token,
            ratio=ratio,
            lc_per_req=lc_per_req,
            req_pool_indices=forward_batch.req_pool_indices.to(torch.int64),
            device=pos.device,
        )
        k_slots = torch.cat(slot_chunks)

    def fill_identity_requests(req_lo, req_hi, tok_lo):
        """Identity requests req_lo..req_hi: rows written without scores, None published per request."""
        tok = tok_lo
        for b in range(req_lo, req_hi):
            t_len, lc = extend_lens_cpu[b], lc_per_req[b]
            rows = slice(tok, tok + t_len)
            tok += t_len
            if lc == 0:
                if publish is not None:
                    publish.append(None)
                continue
            _fill_identity_request(
                indexer,
                lc=lc,
                slots_j=k_slots[starts[b] : starts[b] + lc],
                lens=compress_lens[rows],
                page_rows=page_indices[rows],
                raw_rows=None if raw_indices is None else raw_indices[rows],
            )
            if publish is not None:
                publish.append(None)

    if all(is_identity):
        fill_identity_requests(0, len(extend_lens_cpu), 0)
        if publish is not None:
            backend.candidate_masks = publish
        return

    indexer_metadata = metadata.low_ratio_indexer_metadata(ratio)
    assert indexer_metadata is not None, f"no prefill indexer metadata for {ratio = }"
    num_tokens = pos.shape[0]
    assert indexer_metadata.page_table.shape[0] >= num_tokens
    q_fp4, q_scale, weights = _indexer_inputs(layer, x, q_lora, pos)
    k_payload = pool.get_index_k_fp4_payload_buffer(layer.layer_id)
    k_scale = pool.get_index_k_fp4_scale_buffer(layer.layer_id)
    prefill_workspace = metadata.fp4_low_ratio_prefill_workspaces.get(ratio)

    def score_rows(rows: slice) -> torch.Tensor:
        return aiter_fp4_paged_mqa_logits(
            q_fp4=q_fp4[rows],
            q_scale=q_scale[rows],
            k_payload=k_payload,
            k_scale=k_scale,
            weights=weights[rows],
            page_table=indexer_metadata.page_table[rows],
            c4_seq_lens=indexer_metadata.c4_seq_lens[rows],
            weight_scale=1.0,
            page_table_bucket=LOW_RATIO_PAGE_TABLE_BUCKET,
            is_decode=False,
            prefill_workspace=prefill_workspace if rows.start == 0 else None,
        )

    def select_rows(
        rows: slice, req_lo, req_hi, group_is_identity, consume_rows, publish
    ):
        _select_topk_extend_hip(
            indexer=indexer,
            logits=score_rows(rows),
            lc_per_req=lc_per_req[req_lo:req_hi],
            extend_lens_cpu=[rows.stop - rows.start]
            if req_hi == req_lo + 1
            else extend_lens_cpu[req_lo:req_hi],
            is_identity=group_is_identity,
            compress_lens=compress_lens[rows],
            page_table=indexer_metadata.page_table[rows],
            page_size=indexer_metadata.c4_page_size,
            page_indices=page_indices[rows],
            raw_indices=raw_indices[rows] if raw_indices is not None else None,
            consume=consume_rows,
            publish=publish,
        )

    # group requests to fit the pooled logits block; rows are independent, so grouping is exact
    rows_per_chunk = logits_rows_per_chunk(
        indexer_metadata.page_table, LOW_RATIO_PAGE_TABLE_BUCKET
    )
    groups = _request_groups(extend_lens_cpu, rows_per_chunk)
    for req_lo, req_hi, tok_lo, tok_hi in groups:
        group_is_identity = is_identity[req_lo:req_hi]
        if all(group_is_identity):
            fill_identity_requests(req_lo, req_hi, tok_lo)
            continue
        if tok_hi - tok_lo <= rows_per_chunk:
            select_rows(
                slice(tok_lo, tok_hi),
                req_lo,
                req_hi,
                group_is_identity,
                None if consume is None else consume[req_lo:req_hi],
                publish,
            )
            continue
        # a request wider than the logits block: scored in row chunks, its mask published in one piece
        assert req_hi == req_lo + 1, (req_lo, req_hi)
        pieces = []
        for lo in range(tok_lo, tok_hi, rows_per_chunk):
            rows = slice(lo, min(lo + rows_per_chunk, tok_hi))
            piece = [] if publish is not None else None
            consume_rows = None
            if consume is not None:
                consume_rows = [
                    None
                    if consume[req_lo] is None
                    else slice_candidate_blocks(
                        consume[req_lo], slice(lo - tok_lo, rows.stop - tok_lo)
                    )
                ]
            select_rows(rows, req_lo, req_hi, group_is_identity, consume_rows, piece)
            if piece:
                pieces.append(piece[0])
        if publish is not None:
            publish.append(cat_candidate_blocks(pieces))
    if publish is not None:
        backend.candidate_masks = publish


def _select_topk_extend_hip(
    *,
    indexer,
    logits: torch.Tensor,
    lc_per_req: List[int],
    extend_lens_cpu: List[int],
    is_identity: List[bool],
    compress_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    page_indices: torch.Tensor,
    raw_indices: Optional[torch.Tensor],
    consume: Optional[List[Optional[CandidateBlocks]]],
    publish: Optional[List[Optional[CandidateBlocks]]],
) -> None:
    """Row t of ``logits`` scores its request's compressed positions in columns 0..lc-1, reachable
    up to ``compress_lens[t]``. A source layer publishes one ``CandidateBlocks`` per request (None
    for identity / empty), a consumer selects inside its published blocks, others run one paged top-k."""
    assert page_indices.shape[1] == indexer.index_topk, (
        f"the paged top-k selects page_indices.shape[1] = {page_indices.shape[1]} "
        f"slots, the indexer wants {indexer.index_topk}"
    )
    if publish is not None:
        tok_start = 0
        for b, (lc, t_len) in enumerate(zip(lc_per_req, extend_lens_cpu)):
            rows = slice(tok_start, tok_start + t_len)
            tok_start += t_len
            if lc == 0 or t_len == 0 or is_identity[b]:
                # Consumers index the publication by request, so keep the slot.
                publish.append(None)
                continue
            step = max(1, _TORCH_INDEXER_SCORE_BUDGET_BYTES // (lc * 4))
            pieces = []
            for lo in range(0, t_len, step):
                chunk = slice(rows.start + lo, rows.start + min(lo + step, t_len))
                pieces.append(
                    select_candidate_blocks_hip(
                        logits[chunk, :lc],
                        compress_lens[chunk],
                        topk_blocks=indexer.candidate_topk_blocks,
                        block_size=indexer.candidate_block_size,
                    )
                )
            publish.append(cat_candidate_blocks(pieces))
    if consume is not None:
        tok_start = 0
        for b, t_len in enumerate(extend_lens_cpu):
            rows = slice(tok_start, tok_start + t_len)
            tok_start += t_len
            if t_len == 0:
                continue
            rows_page = page_indices[rows]
            rows_raw = raw_indices[rows] if raw_indices is not None else None
            if consume[b] is None:
                topk_transform_paged_sorted(
                    logits[rows],
                    compress_lens[rows].contiguous(),
                    page_table[rows],
                    rows_page,
                    page_size,
                    rows_raw,
                )
            else:
                topk_within_candidate_blocks_hip(
                    logits[rows],
                    compress_lens[rows],
                    consume[b],
                    page_table=page_table[rows],
                    page_size=page_size,
                    page_indices=rows_page,
                    raw_indices=rows_raw,
                    sort_output=True,
                )
        return
    topk_transform_paged_sorted(
        logits,
        compress_lens.contiguous(),
        page_table,
        page_indices,
        page_size,
        raw_indices,
    )


def _request_groups(
    extend_lens_cpu: List[int], rows_per_chunk: int
) -> List[Tuple[int, int, int, int]]:
    """Consecutive request groups whose token rows fit `rows_per_chunk` (a single
    request always forms a group): (req_lo, req_hi, tok_lo, tok_hi)."""
    groups = []
    req_lo, tok_lo, tok = 0, 0, 0
    for r, t_len in enumerate(extend_lens_cpu):
        if r > req_lo and tok + t_len - tok_lo > rows_per_chunk:
            groups.append((req_lo, r, tok_lo, tok))
            req_lo, tok_lo = r, tok
        tok += t_len
    groups.append((req_lo, len(extend_lens_cpu), tok_lo, tok))
    return groups
