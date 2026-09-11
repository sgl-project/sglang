from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_xpu

from .utils import make_name


@cache_once
def _jit_topk_v1_module():
    # topk (<= 1024) is a runtime argument, not a compile-time constant, so a
    # single module serves every k. Baking it in via -DSGL_TOPK used to build one
    # module per k, and since the macro fed a `constexpr` rather than a template
    # parameter every module exported identically mangled symbols -- see the
    # comment in topk_v1.cuh for how that broke the second module's launch.
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk_v1"),
        *args,
        cuda_files=["deepseek_v4/topk_v1.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKKernel<{args}>::transform")],
    )


@cache_once
def _jit_topk_v2_module():
    # v2 is universal: topk (<= 2048) is a runtime argument, not a compile-time
    # constant, so a single module serves every k.
    return load_jit(
        make_name("topk_v2"),
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=[
            ("topk_transform_paged", "TopKKernel::transform_paged"),
            ("topk_transform_ragged", "TopKKernel::transform_ragged"),
            ("topk_plan", "TopKKernel::plan"),
            ("topk_coop_workspace_bytes", "TopKKernel::coop_workspace_bytes"),
        ],
    )


# kClusterFloorSmall in topk_v2.cuh. The handoff guard the cooperative kernel
# pairs with lives on the official kernel's cluster path, which a batch-1 row
# only takes above this length.
_COOP_TOPK_MIN_FLOOR = 32768


def _coop_topk_floor() -> int:
    # Row length above which paged top-k v2 hands off to the cooperative kernel;
    # 0 is the wire value for "off". Read per call, like SGLANG_OPT_USE_TOPK_V2
    # on this path; a captured graph freezes both at capture.
    if is_hip_runtime() or not envs.SGLANG_OPT_USE_COOP_TOPK.get():
        return 0
    floor = envs.SGLANG_OPT_COOP_TOPK_FLOOR.get()
    # Under the bound, a row the cooperative kernel claims can take the official
    # kernel's non-cluster path, which carries no handoff guard: two writers into
    # one output row. The kernel takes the floor as uint32_t, so a value past
    # 0xffffffff would arrive truncated -- and truncated to 0 means "off".
    if floor is None or not _COOP_TOPK_MIN_FLOOR <= floor <= 0xFFFFFFFF:
        raise ValueError(
            f"SGLANG_OPT_COOP_TOPK_FLOOR must be at least {_COOP_TOPK_MIN_FLOOR} "
            f"and at most {0xFFFFFFFF} when SGLANG_OPT_USE_COOP_TOPK is set, "
            f"got {floor}"
        )
    return floor


@cache_once
def _coop_topk_workspace(device_index: int) -> torch.Tensor:
    # Cross-block state for the cooperative kernel: zeroed once here, and every
    # launch leaves it ready for the next. One buffer per device serves every
    # launch, so that handoff holds only while launches never overlap -- true
    # while they are enqueued in order on the caller's stream, and false if a
    # second stream ever reaches this entry point concurrently.
    if torch.cuda.is_current_stream_capturing():
        # Graph-private memory is reused after capture, so a buffer allocated
        # here would be handed out again while a replay still reads it. Raising
        # rather than asserting: python -O strips the assert and leaves the
        # corruption silent.
        raise RuntimeError(
            "coop top-k workspace must be allocated before CUDA graph capture"
        )
    nbytes = int(_jit_topk_v2_module().topk_coop_workspace_bytes())
    return torch.zeros(
        -(-nbytes // 4), dtype=torch.int32, device=torch.device("cuda", device_index)
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


def plan_topk_v2(seq_lens: torch.Tensor, static_threshold: int = 0) -> torch.Tensor:
    """Preprocess the per-batch routing plan for :func:`topk_transform_paged_v2`.

    IMPORTANT: every entry of ``seq_lens`` must be NON-NEGATIVE. The device
    kernel reads the int32 buffer as ``uint32_t``, so a negative length (e.g.
    -4 from a DP-padded / idle-companion row) reinterprets as ~4e9, poisons
    the plan, and drives the transform kernel into an illegal memory access.
    Producers of padded rows must clamp their lengths to 0 (0 selects the
    trivial all-(-1) output path, which is safe).
    """
    module = _jit_topk_v2_module()
    bs = seq_lens.shape[0]
    metadata = seq_lens.new_empty(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)
    module.topk_plan(seq_lens, metadata, static_threshold)
    return metadata


def topk_transform_ragged_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    out_offsets: torch.Tensor,
    out_indices: torch.Tensor,
    row_starts: Optional[torch.Tensor] = None,
) -> None:
    """Ragged (prefill) fused top-k for a contiguous-KV score matrix.

    Row ``i`` selects the top-k of ``scores[i, ks : ks + seq_lens[i]]`` (``ks =
    row_starts[i]``, 0 when ``row_starts`` is omitted) and writes
    ``selected_position + out_offsets[i]`` into ``out_indices``, ``-1`` padded.
    With the production convention ``out_offsets == row_starts`` that is the
    column index itself, i.e. the token's slot in the batch's flattened KV.

    Unlike :func:`topk_transform_paged_v2` this needs no page table and no plan
    (the cluster path only pays off for very few rows, and prefill has many).

    IMPORTANT: ``scores`` is written in place -- the <= 3 columns ahead of each
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
    module = _jit_topk_v2_module()
    module.topk_transform_ragged(scores, seq_lens, row_starts, out_offsets, out_indices)


def topk_transform_paged_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: Optional[torch.Tensor],
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
) -> None:
    """Fused top-k + optional page-table transform (DeepSeek-V4 top-k v2 kernel).

    Two output modes, chosen by whether ``page_tables`` is given and resolved to
    a device-side template parameter, so an unused page-table gather is compiled
    out rather than skipped at runtime:

    * ``page_tables=None`` -- ``out_page_indices`` receives the raw selected
      indices and no page table is read.
    * ``page_tables`` given -- ``out_page_indices`` receives the page-table
      transform of them.

    With ``SGLANG_OPT_USE_COOP_TOPK`` set, rows longer than
    ``SGLANG_OPT_COOP_TOPK_FLOOR`` are instead selected by a second,
    grid-cooperative kernel enqueued on the same stream; both output modes and
    the contract below are unchanged.

    IMPORTANT: every entry of ``seq_lens`` must be NON-NEGATIVE, and
    ``metadata`` must come from :func:`plan_topk_v2` over the same ``seq_lens``
    values. The kernel reads lengths as ``uint32_t``: a negative entry
    reinterprets as a ~4e9-token sequence, sending the row down the cluster
    path over garbage scores and crashing with an illegal memory access
    (GLM 5.2 MTP DP-idle companion rows hit exactly this). A length of 0 is
    the valid way to express "no tokens": the row takes the trivial path and
    the output is all -1.
    """
    if is_xpu():
        torch.ops.sgl_kernel.topk_transform_paged(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            page_size,
            metadata,
        )
        return
    module = _jit_topk_v2_module()
    coop_floor = _coop_topk_floor()
    module.topk_transform_paged(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        metadata,
        coop_floor,
        _coop_topk_workspace(scores.device.index) if coop_floor else None,
    )
