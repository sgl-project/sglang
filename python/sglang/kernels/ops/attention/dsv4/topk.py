from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_xpu

from .candidate_table import CANDIDATE_BLOCK_SIZE
from .utils import make_name


@cache_once
def _jit_topk_v1_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk_v1"),
        *args,
        cuda_files=["deepseek_v4/topk_v1.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKKernel<{args}>::transform")],
    )


@cache_once
def _jit_topk_v2_module(deterministic: bool = False, bf16_scores: bool = False):
    from sglang.kernels.jit.utils.occupancy import (
        NoSchedulableClustersError,
        get_max_active_clusters,
    )

    # The DSA decode context-parallel merge needs index-ordered ties on its own
    # calls, whatever the process-wide mode.
    mode = 2 if deterministic else envs.SGLANG_TOPK_V2_MODE.get()
    assert mode in (0, 1, 2), f"SGLANG_TOPK_V2_MODE must be 0, 1 or 2, got {mode}"
    extra_cuda_cflags = [f"-DSGL_TOPK_MODE={mode}"]
    if bf16_scores:
        extra_cuda_cflags.append("-DSGL_TOPK_SCORE_BF16")
    arch = current_arch()
    # on CUDA, ptxas schedules kernels in parallel to speed up compilation
    if arch.cuda is not None:
        extra_cuda_cflags.append("-Xptxas=--split-compile=0")
    # Enable each cluster path only when its occupancy probe reports capacity.
    if arch.cuda is not None and arch.cuda.supports_cluster:
        for cluster_size, occupancy in ((8, 2), (16, 1)):
            try:
                max_active_clusters = get_max_active_clusters(
                    cluster_size=cluster_size,
                    occupancy=occupancy,
                )
            except NoSchedulableClustersError:
                max_active_clusters = 0
            extra_cuda_cflags.append(
                f"-DSGL_TOPK_V2_MAX_C{cluster_size}_OCC{occupancy}={max_active_clusters}"
            )
    args = make_cpp_args(arch.cuda is not None and arch.supports_pdl)
    kernel = f"TopKKernel<{args}>"
    wrappers = [
        ("topk_transform_paged", f"{kernel}::transform_paged"),
        ("topk_transform_ragged", f"{kernel}::transform_ragged"),
        ("topk_transform_packed", f"{kernel}::transform_packed"),
        ("topk_transform_candidates", f"{kernel}::transform_candidates"),
        ("topk_transform_candidates_paged", f"{kernel}::transform_candidates_paged"),
        ("topk_plan", f"{kernel}::plan"),
        ("topk_transform_kpool", f"{kernel}::transform_kpool"),
        ("topk_transform_kpool_ragged", f"{kernel}::transform_kpool_ragged"),
    ]
    return load_jit(
        make_name("topk_v2"),
        *args,
        f"mode{mode}" + ("_bf16" if bf16_scores else ""),
        extra_cuda_cflags=extra_cuda_cflags,
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=wrappers,
    )


@cache_once
def _jit_topk_bf16_small_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk_bf16_small"),
        *args,
        cuda_files=["deepseek_v4/topk_bf16_small.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKBF16Kernel<{args}>::transform")],
    )


def topk_transform_bf16_small(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
) -> None:
    """bf16 top-k for rows of at most 16384 scores (the DeepSeek-V4.1 sparse
    indexer's consumer rows), fused with a page-table transform.

    Row ``b`` selects the ``k = out_page_indices.shape[1]`` best of its first
    ``seq_lens[b]`` scores (``k`` at most 2048); a selected index ``i`` is
    written as ``page_table[b, i // page_size] * page_size + i % page_size``,
    in no particular order, and ``-1`` fills the slots past
    ``min(k, seq_lens[b])``. Selection is exact (two radix passes over the raw
    bf16 bytes locate the k-th largest value); which of the elements equal to
    it fill the last slots is arbitrary. NaN scores are not supported.
    """
    _jit_topk_bf16_small_module().topk_transform(
        scores, seq_lens, page_table, out_page_indices, page_size
    )


def topk_transform_paged(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    if is_hip_runtime():
        torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )
    elif is_xpu():
        torch.ops.sgl_kernel.topk_transform(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )
    else:
        module = _jit_topk_v1_module()
        module.topk_transform(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )


# metadata is (batch+1, 2) int32: row 0 = {cluster_threshold, num_cluster_items};
# rows 1..N = {batch_id, seq_len} of items routed to the persistent cluster pool.
_PLAN_METADATA_INTS_PER_BATCH = 2


def plan_topk_v2(
    seq_lens: torch.Tensor,
    static_threshold: int = -1,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Preprocess the per-batch routing plan for :func:`topk_transform_paged_v2`.
    NOTE: every entry of ``seq_lens`` must be NON-NEGATIVE.

    :param static_threshold: If a batch item has `seq_len` > `static_threshold`,
                             prefer the cluster implementation.
                             Negative number means internal heuristic.
    """
    module = _jit_topk_v2_module()
    bs = seq_lens.shape[0]
    metadata = (
        seq_lens.new_empty(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)
        if out is None
        else out
    )
    module.topk_plan(seq_lens, metadata, static_threshold)
    return metadata


def topk_v2_plan_is_written(seq_lens: torch.Tensor) -> bool:
    """Whether :func:`plan_topk_v2` writes a plan for these lengths. Small
    batches and devices without clusters leave the plan buffer untouched."""
    probe = torch.full(
        (seq_lens.shape[0] + 1, _PLAN_METADATA_INTS_PER_BATCH),
        -1,
        dtype=torch.int32,
        device=seq_lens.device,
    )
    _jit_topk_v2_module().topk_plan(seq_lens, probe, -1)
    return probe[0, 1].item() != -1


def topk_transform_ragged_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    out_offsets: torch.Tensor,
    out_indices: torch.Tensor,
    row_starts: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> None:
    """Ragged (prefill) fused top-k for a contiguous-KV score matrix.

    Row ``i`` selects the top-k of ``scores[i, ks : ks + seq_lens[i]]`` (``ks =
    row_starts[i]``, 0 when ``row_starts`` is omitted) and writes
    ``selected_position + out_offsets[i]`` into ``out_indices``, ``-1`` padded.
    With the production convention ``out_offsets == row_starts`` that is the
    column index itself, i.e. the token's slot in the batch's flattened KV.

    Unlike :func:`topk_transform_paged_v2` this needs no page table and no plan
    (the cluster path only pays off for very few rows, and prefill has many).

    NOTE: ``scores`` is written in place -- the <= 3 columns ahead of each
    row's window that the 16-byte-aligned read base pulls in are masked out.
    They are invalid for that row and the buffer must have no other consumer.
    ``seq_lens`` entries must be NON-NEGATIVE, as for the paged entry point.
    """
    if is_xpu():
        torch.ops.sgl_kernel.topk_transform_ragged(
            scores,
            seq_lens,
            out_indices,
            out_offsets,
            row_starts,
        )
        return
    module = _jit_topk_v2_module(deterministic, scores.dtype == torch.bfloat16)
    module.topk_transform_ragged(scores, seq_lens, row_starts, out_offsets, out_indices)


# Candidate rows longer than this split over a cluster of blocks.
_CANDIDATES_CLUSTER_FLOOR = 16384


def _candidates_cluster_size(rows: int) -> int:
    # SM90 / SM100 only, as SUPPORT_CLUSTER in topk_impl.cuh builds the cluster path.
    arch = current_arch()
    if rows <= 0 or not (arch.is_sm(9) or arch.is_sm(10)):
        return 0
    from sglang.kernels.jit.utils.occupancy import (
        NoSchedulableClustersError,
        get_max_active_clusters,
    )

    for size in (4, 2):
        try:
            if rows <= get_max_active_clusters(size, occupancy=1):
                return size
        except NoSchedulableClustersError:
            continue
    return 0


def topk_transform_candidates_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    out_candidates: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    row_starts: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    cluster_size: Optional[int] = None,
    cluster_floor: int = _CANDIDATES_CLUSTER_FLOOR,
) -> None:
    """Ragged top-k emitting candidates for a distributed top-k merge.

    Row ``i`` selects the top-k of ``scores[i, ks : ks + seq_lens[i]]`` (``ks
    = row_starts[i]``, 0 when omitted) exactly as :func:`topk_transform_ragged_v2`
    does, and writes ``out_candidates[i]`` (``[topk, 2]`` int32) = (fp32 score
    bits, ``position * dcp_size + dcp_rank``) per selected position, ``(0, -1)``
    padded: local position ``l`` of rank ``dcp_rank`` is global position
    ``l * dcp_size + dcp_rank`` under decode context parallelism.

    ``cluster_size`` None picks one from the row count (rows starting at
    column 0 only); 0 keeps one block per row.

    NOTE: ``scores`` is written in place as for the ragged entry point.
    """
    if cluster_size is None:
        rows = seq_lens.shape[0]
        cluster_size = 0 if row_starts is not None else _candidates_cluster_size(rows)
    _jit_topk_v2_module(deterministic).topk_transform_candidates(
        scores,
        seq_lens,
        row_starts,
        out_candidates,
        dcp_size,
        dcp_rank,
        cluster_size,
        cluster_floor,
    )


def topk_transform_paged_candidates_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    out_candidates: torch.Tensor,
    world: int,
    rank: int,
    row_starts: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    cluster_floor: Optional[int] = None,
    page_bits: int = 0,
    overflow_rows: Optional[torch.Tensor] = None,
    coarse_bits: int = 10,
) -> None:
    """:func:`topk_transform_candidates_v2` for rows split by page.

    Local position ``l`` of rank ``rank`` is global position
    ``((l >> b) * world + rank) << b | (l & (2**b - 1))`` with ``page_bits`` b
    (b = 0 is the token interleave). ``cluster_floor`` (no ``row_starts``) splits
    rows longer than it across a cluster at coarse bin width ``coarse_bits``
    (10 / 11 / 12); the selection stays exact. ``overflow_rows`` (int32 [1])
    counts rows redone past the tie buffer.

    NOTE: ``scores`` is written in place as for the ragged entry point.
    """
    _jit_topk_v2_module(deterministic).topk_transform_candidates_paged(
        scores,
        seq_lens,
        row_starts,
        out_candidates,
        world,
        rank,
        -1 if cluster_floor is None else int(cluster_floor),
        int(page_bits),
        overflow_rows,
        int(coarse_bits),
    )


def topk_transform_paged_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: Optional[torch.Tensor],
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
    out_raw_indices: Optional[torch.Tensor] = None,
    *,
    deterministic: bool = False,
) -> None:
    """Fused top-k + optional page-table transform (DeepSeek-V4 top-k v2 kernel).

    Output mode is chosen from ``page_tables`` and ``out_raw_indices`` and
    resolved to a device-side template parameter, so an unused page-table gather
    is compiled out rather than skipped at runtime:

    * ``page_tables=None`` -- ``out_page_indices`` receives the raw selected
      indices and no page table is read.
    * ``page_tables`` given -- ``out_page_indices`` receives the page-table
      transform of them.
    * Both outputs given -- ``out_page_indices`` receives the page-table
      transform and ``out_raw_indices`` receives the selected raw indices.

    For the packed (DSA extend prefill) layout see
    :func:`topk_transform_packed_v2`.

    NOTE: every entry of `seq_lens` must be NON-NEGATIVE, and `metadata` must
    come from :func:`plan_topk_v2` over the same `seq_lens` values.
    A length of 0 is the valid way to express "no tokens": the row takes the
    trivial path and the output is guaranteed to be all -1.

    Among scores equal to the k-th largest, which fill the last slots is
    arbitrary, unless ``deterministic`` builds this call in top-k mode 2
    (``SGLANG_TOPK_V2_MODE``): then the smallest indices do (the merge of the
    DSA decode context-parallel top-k relies on it). The same holds for
    the candidates and packed entry points.
    """
    if is_xpu():
        if out_raw_indices is not None:
            topk_transform_paged(
                scores,
                seq_lens,
                page_tables,
                out_page_indices,
                page_size,
                out_raw_indices,
            )
            return
        torch.ops.sgl_kernel.topk_transform_paged(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            page_size,
            metadata,
        )
        return
    module = _jit_topk_v2_module(deterministic)
    module.topk_transform_paged(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        metadata,
        out_raw_indices,
    )


def topk_transform_kpool_v2(
    scores: torch.Tensor,
    pool_lens: torch.Tensor,
    out: torch.Tensor,
    pool_size: int,
    metadata: torch.Tensor,
    *,
    token_seq_lens: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 1,
    out_offsets: Optional[torch.Tensor] = None,
) -> None:
    """GLM k-pool top-k over per-pool scores, fused with pool expansion, the
    ``token_seq_lens % pool_size`` tail and the page-table or offset transform.

    ``metadata`` is :func:`plan_topk_v2` over ``pool_lens``. The kernel reads it
    before its PDL wait, so plan before launching the kernel that writes ``scores``.
    """
    _jit_topk_v2_module().topk_transform_kpool(
        scores,
        pool_lens,
        token_seq_lens,
        page_table,
        out_offsets,
        out,
        page_size,
        pool_size,
        metadata,
    )


def topk_transform_kpool_ragged_v2(
    scores: torch.Tensor,
    pool_lens: torch.Tensor,
    out: torch.Tensor,
    pool_size: int,
    *,
    row_starts: Optional[torch.Tensor] = None,
    token_seq_lens: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
    page_size: int = 1,
    out_offsets: Optional[torch.Tensor] = None,
) -> None:
    """Ragged (prefill) :func:`topk_transform_kpool_v2`: row ``i`` selects among
    ``scores[i, ks : ks + pool_lens[i]]`` (``ks = row_starts[i]``, 0 when omitted);
    no plan is needed.

    NOTE: ``scores`` is written in place, as in :func:`topk_transform_ragged_v2`.
    """
    _jit_topk_v2_module().topk_transform_kpool_ragged(
        scores,
        pool_lens,
        row_starts,
        token_seq_lens,
        page_table,
        page_table_row_index,
        out_offsets,
        out,
        page_size,
        pool_size,
    )


def topk_transform_packed_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    *,
    row_starts: torch.Tensor,
    row_to_batch: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> None:
    """Packed (DSA extend prefill) fused top-k + page-table transform.

    Row ``i`` selects the top-k of ``scores[i, ks : ks + seq_lens[i]]``
    (``ks = row_starts[i]``) and writes the page-table transform of the selected
    row-local positions into ``out_page_indices``, ``-1`` padded. Prefill expands
    one request into many query-token rows, so ``row_to_batch[i]`` (optional,
    ``(rows,)`` int32) names the ``page_tables`` row of the request row ``i``
    belongs to; omitting it indexes the table by score row. ``row_to_batch`` is
    not range-checked.

    This is :func:`topk_transform_ragged_v2` with a page-table output instead of
    an additive offset. Like ragged, it dispatches the implementation per row at
    runtime, so it needs no plan and no :func:`plan_topk_v2` metadata.

    NOTE: ``scores`` is MODIFIED IN PLACE -- the <= 3 columns ahead of each row's
    window that the 16-byte-aligned read base pulls in are masked out. They are
    invalid for that row and the buffer must have no other consumer, so do not
    pass a view with overlapping rows.
    ``seq_lens`` entries must be NON-NEGATIVE, as for the paged entry point.
    """
    module = _jit_topk_v2_module(deterministic)
    module.topk_transform_packed(
        scores,
        seq_lens,
        row_starts,
        page_tables,
        out_page_indices,
        page_size,
        row_to_batch,
    )


def topk_transform_sparse(
    logits: torch.Tensor,
    valid_lens: torch.Tensor,
    blocks: torch.Tensor,
    out_indices: torch.Tensor,
) -> None:
    """Top-``k`` (``k = out_indices.shape[1]``) of each row of the bf16 sparse
    ``logits`` within its first ``valid_lens[b]`` columns, ``-1`` padded, unordered;
    column ``j`` is written as ``blocks[b, j // 8] * 8 + j % 8``: pool slots for the
    published blocks as pool slots / 8, compressed positions for logical ids."""
    topk_transform_bf16_small(
        logits, valid_lens, blocks, out_indices, CANDIDATE_BLOCK_SIZE
    )
