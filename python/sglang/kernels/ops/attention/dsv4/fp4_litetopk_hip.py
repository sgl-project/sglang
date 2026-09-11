"""Fail-closed adapter for AITER's gfx950 FP4 LiteTopK prefill operator."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from inspect import signature
from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from aiter.ops.flydsl import FP4LiteTopKWorkspace


_HEADS = 64
_PACKED_HEAD_DIM = 64
_SUPPORTED_TOPKS = (512, 1024)
_PAGE_SIZE = 64
_MIN_CONTEXT = 65_536
_MAX_CONTEXT = 196_608
_MAX_WORKSPACE_ROWS = 1024
_ALLOCATION_HEADROOM_BYTES = 256 * 1024 * 1024


class FP4LiteTopKScratch(NamedTuple):
    workspace: FP4LiteTopKWorkspace
    rows: int
    topk: int
    device_index: int
    stream_id: int


def _is_status_aware_aiter_api(run_litetopk, workspace_type: type) -> bool:
    try:
        parameters = signature(run_litetopk).parameters
    except (TypeError, ValueError):
        return False
    return "status_ok" in getattr(workspace_type, "_fields", ()) and (
        "enforce_status" in parameters
    )


@lru_cache(maxsize=1)
def get_aiter_fp4_litetopk():
    """Resolve the complete optional AITER API, or return ``None``."""
    try:
        from aiter.ops.flydsl import (
            FP4_LITETOPK_SUPPORTED_TOPKS,
            FP4LiteTopKResult,
            FP4LiteTopKWorkspace,
            allocate_fp4_litetopk_workspace,
            flydsl_pa_mqa_litetopk_fp4_prefill,
            fp4_litetopk_workspace_size,
        )
    except (ImportError, AttributeError):
        return None
    if not _is_status_aware_aiter_api(
        flydsl_pa_mqa_litetopk_fp4_prefill, FP4LiteTopKWorkspace
    ):
        return None
    try:
        supported_topks = frozenset(int(topk) for topk in FP4_LITETOPK_SUPPORTED_TOPKS)
    except (TypeError, ValueError):
        return None
    return (
        flydsl_pa_mqa_litetopk_fp4_prefill,
        allocate_fp4_litetopk_workspace,
        fp4_litetopk_workspace_size,
        FP4LiteTopKWorkspace,
        FP4LiteTopKResult,
        supported_topks,
    )


def aiter_fp4_litetopk_supports_topk(topk: int) -> bool:
    api = get_aiter_fp4_litetopk()
    return topk in _SUPPORTED_TOPKS and api is not None and topk in api[-1]


def can_use_fp4_litetopk(
    *,
    enabled: bool,
    is_hip: bool,
    arch: str,
    is_extend: bool,
    batch_size: int,
    query_rows: int,
    heads: int,
    packed_head_dim: int,
    topk: int,
    page_size: int,
    max_context: int,
    attn_cp_size: int,
    use_sgl_topk: bool,
    use_prefill_graph: bool,
    capturing: bool,
    in_piecewise_graph: bool,
    in_breakable_graph: bool,
    has_tbo_parent: bool,
    has_tbo_children: bool,
    has_spec_info: bool,
    has_spec_algorithm: bool,
    enable_multi_stream: bool,
    skip_compressor: bool,
    has_hisparse: bool,
    has_prefill_workspace: bool,
    aiter_available: bool,
) -> bool:
    """Return whether a call matches the deliberately narrow v1 contract."""
    return fp4_litetopk_ineligible_reason(**locals()) is None


def fp4_litetopk_ineligible_reason(**kwargs) -> str | None:
    """Return the first failed v1 requirement, or ``None`` when eligible."""
    checks = (
        (kwargs["enabled"], "feature flag is disabled"),
        (kwargs["is_hip"], "platform is not HIP"),
        (kwargs["arch"] == "gfx950", f"architecture is {kwargs['arch']!r}"),
        (kwargs["is_extend"], "forward mode is not plain EXTEND"),
        (kwargs["batch_size"] == 1, "batch size is not one"),
        (kwargs["query_rows"] > 0, "query row count is zero"),
        (kwargs["heads"] == _HEADS, "query head count is not 64"),
        (
            kwargs["packed_head_dim"] == _PACKED_HEAD_DIM,
            "packed query head dimension is not 64",
        ),
        (
            kwargs["topk"] in _SUPPORTED_TOPKS,
            f"top-k is not one of {_SUPPORTED_TOPKS}",
        ),
        (kwargs["page_size"] == _PAGE_SIZE, "C4 page size is not 64"),
        (
            _MIN_CONTEXT <= kwargs["max_context"] <= _MAX_CONTEXT,
            "C4 context is outside [65536, 196608]",
        ),
        (kwargs["attn_cp_size"] == 1, "attention context parallelism is active"),
        (kwargs["use_sgl_topk"], "SGL top-k backend is not selected"),
        (not kwargs["use_prefill_graph"], "prefill graph metadata is active"),
        (not kwargs["capturing"], "stream capture is active"),
        (not kwargs["in_piecewise_graph"], "piecewise graph is active"),
        (not kwargs["in_breakable_graph"], "breakable graph is active"),
        (not kwargs["has_tbo_parent"], "TBO parent range is active"),
        (not kwargs["has_tbo_children"], "TBO child batches are active"),
        (not kwargs["has_spec_info"], "speculative metadata is active"),
        (not kwargs["has_spec_algorithm"], "speculative decoding is configured"),
        (not kwargs["enable_multi_stream"], "multi-stream indexer is active"),
        (not kwargs["skip_compressor"], "compressor execution is skipped"),
        (not kwargs["has_hisparse"], "HiSparse is active"),
        (
            kwargs["has_prefill_workspace"],
            "prefill workspace is missing or incompatible",
        ),
        (kwargs["aiter_available"], "AITER LiteTopK API is unavailable"),
    )
    return next((reason for passed, reason in checks if not passed), None)


def is_fp4_litetopk_prefill_workspace_compatible(
    workspace: object | None,
    *,
    rows: int,
    max_seq_len: int,
    device: torch.device,
) -> bool:
    """Return whether cached prefill metadata can serve this LiteTopK call."""
    if workspace is None or rows <= 0 or max_seq_len <= 0:
        return False
    try:
        guarded_page_table = workspace.guarded_page_table
        row_to_batch = workspace.row_to_batch
        local_starts = workspace.local_starts
        workspace_max_seq_len = int(workspace.max_seq_len)
    except (AttributeError, TypeError, ValueError):
        return False
    device = torch.device(device)
    return (
        isinstance(guarded_page_table, torch.Tensor)
        and guarded_page_table.dtype == torch.int32
        and guarded_page_table.device == device
        and guarded_page_table.ndim == 2
        and guarded_page_table.shape[0] >= rows
        and guarded_page_table.is_contiguous()
        and isinstance(row_to_batch, torch.Tensor)
        and row_to_batch.dtype == torch.int32
        and row_to_batch.device == device
        and row_to_batch.ndim == 1
        and row_to_batch.shape[0] >= rows
        and row_to_batch.is_contiguous()
        and isinstance(local_starts, torch.Tensor)
        and local_starts.dtype == torch.int32
        and local_starts.device == device
        and local_starts.ndim == 1
        and local_starts.shape[0] >= rows
        and local_starts.is_contiguous()
        and workspace_max_seq_len >= max_seq_len
        and guarded_page_table.shape[1] * _PAGE_SIZE >= max_seq_len
    )


def validate_fp4_litetopk_static_configuration(
    *,
    required: bool,
    is_hip_platform: bool,
    use_fp4_indexer: bool,
    arch: str = "gfx950",
    heads: int = _HEADS,
    packed_head_dim: int = _PACKED_HEAD_DIM,
    topk: int = _SUPPORTED_TOPKS[0],
    use_sgl_topk: bool = True,
    aiter_available: bool = True,
) -> None:
    """Fail early when REQUIRED cannot be served by this model runner."""
    if not required:
        return
    if not is_hip_platform:
        raise RuntimeError(
            "SGLANG_DSV4_FP4_LITETOPK_REQUIRED=1 requires a HIP platform"
        )
    if not use_fp4_indexer:
        raise RuntimeError(
            "SGLANG_DSV4_FP4_LITETOPK_REQUIRED=1 requires the AITER FP4 indexer"
        )
    if arch != "gfx950":
        raise RuntimeError(
            f"SGLANG_DSV4_FP4_LITETOPK_REQUIRED=1 requires gfx950, got {arch!r}"
        )
    if (
        heads != _HEADS
        or packed_head_dim != _PACKED_HEAD_DIM
        or topk not in _SUPPORTED_TOPKS
    ):
        raise RuntimeError(
            "SGLANG_DSV4_FP4_LITETOPK_REQUIRED=1 requires H=64, packed D=64, "
            f"and K in {_SUPPORTED_TOPKS}; got H={heads}, "
            f"packed D={packed_head_dim}, K={topk}"
        )
    if not use_sgl_topk:
        raise RuntimeError(
            "SGLANG_DSV4_FP4_LITETOPK_REQUIRED=1 requires the SGL top-k backend"
        )
    if not aiter_available:
        raise RuntimeError(
            "SGLANG_DSV4_FP4_LITETOPK_REQUIRED=1 requires the complete AITER "
            "LiteTopK API"
        )


def max_c4_context_from_seq_lens(
    seq_lens_cpu: Sequence[int] | torch.Tensor | None,
) -> int:
    """Return the largest compressed C4 length using metadata's floor rule."""
    if seq_lens_cpu is None:
        return 0
    values = (
        seq_lens_cpu.tolist()
        if isinstance(seq_lens_cpu, torch.Tensor)
        else list(seq_lens_cpu)
    )
    return max((int(length) for length in values), default=0) // 4


def fp4_litetopk_must_dispatch(
    *, required: bool, is_plain_extend: bool, max_context: int
) -> bool:
    """Return whether REQUIRED applies to this forward phase.

    Early prefill chunks below the crossover and decode are deliberate legacy
    fallbacks. Once a plain EXTEND reaches the crossover, REQUIRED makes every
    other eligibility failure fatal, including contexts above the supported cap.
    """
    return required and is_plain_extend and max_context >= _MIN_CONTEXT


def prepare_fp4_litetopk_scratch(
    *,
    rows: int,
    topk: int,
    device: torch.device,
    scratch: FP4LiteTopKScratch | None = None,
) -> FP4LiteTopKScratch:
    if rows <= 0 or rows > _MAX_WORKSPACE_ROWS:
        raise ValueError(
            f"LiteTopK workspace rows must be in [1, {_MAX_WORKSPACE_ROWS}], got {rows}"
        )
    if topk not in _SUPPORTED_TOPKS:
        raise ValueError(
            f"FP4 LiteTopK top-k must be one of {_SUPPORTED_TOPKS}, got {topk}"
        )
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"FP4 LiteTopK scratch requires a GPU device, got {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device_index = device.index
    assert device_index is not None

    api = get_aiter_fp4_litetopk()
    if api is None:
        raise RuntimeError("The installed AITER does not expose FP4 LiteTopK")
    _, allocate_workspace, workspace_size, _, _, supported_topks = api
    if topk not in supported_topks:
        raise RuntimeError(
            f"The installed AITER does not support FP4 LiteTopK K={topk}"
        )
    stream = torch.cuda.current_stream(device)
    if (
        scratch is not None
        and scratch.rows >= rows
        and scratch.topk == topk
        and scratch.device_index == device_index
        and scratch.stream_id == stream.cuda_stream
    ):
        return scratch
    required_bytes = workspace_size(rows, topk=topk)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    cached_bytes = max(
        0,
        torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device),
    )
    available_bytes = free_bytes + cached_bytes
    if required_bytes + _ALLOCATION_HEADROOM_BYTES > available_bytes:
        raise torch.OutOfMemoryError(
            "insufficient free memory for FP4 LiteTopK workspace: "
            f"required={required_bytes}, driver_free={free_bytes}, "
            f"allocator_cached={cached_bytes}, "
            f"reserved_headroom={_ALLOCATION_HEADROOM_BYTES}"
        )
    return FP4LiteTopKScratch(
        workspace=allocate_workspace(rows, device, topk=topk, stream=stream),
        rows=rows,
        topk=topk,
        device_index=device_index,
        stream_id=stream.cuda_stream,
    )


def prepare_fp4_litetopk_scratch_for_dispatch(
    *,
    rows: int,
    topk: int,
    device: torch.device,
    scratch: FP4LiteTopKScratch | None,
    required: bool,
) -> FP4LiteTopKScratch | None:
    try:
        return prepare_fp4_litetopk_scratch(
            rows=rows,
            topk=topk,
            device=device,
            scratch=scratch,
        )
    except RuntimeError as exc:
        if required:
            raise RuntimeError(
                "FP4 LiteTopK workspace setup failed in required mode"
            ) from exc
        return None


def aiter_fp4_litetopk(
    *,
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    guarded_page_table: torch.Tensor,
    row_to_batch: torch.Tensor,
    row_starts: torch.Tensor,
    c4_seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    weight_scale: float,
    out_page_indices: torch.Tensor,
    out_raw_indices: torch.Tensor,
    scratch: FP4LiteTopKScratch | None = None,
) -> FP4LiteTopKScratch:
    """Run AITER LiteTopK and write SGLang's logical and physical outputs."""
    api = get_aiter_fp4_litetopk()
    if api is None:
        raise RuntimeError("The installed AITER does not expose FP4 LiteTopK")
    run_litetopk, _, _, _, result_type, supported_topks = api
    if topk not in supported_topks:
        raise RuntimeError(
            f"The installed AITER does not support FP4 LiteTopK K={topk}"
        )
    rows = q_fp4.shape[0]
    scratch = prepare_fp4_litetopk_scratch(
        rows=rows,
        topk=topk,
        device=q_fp4.device,
        scratch=scratch,
    )
    result = result_type(
        values=scratch.workspace.selected_values[:rows],
        raw_indices=out_raw_indices,
        physical_indices=out_page_indices,
        counts=scratch.workspace.output_counts[:rows],
        candidate_counts=scratch.workspace.candidate_counts[:rows],
        status=scratch.workspace.status[:rows],
    )
    run_litetopk(
        q_fp4.view(torch.uint8),
        q_scale,
        k_payload.view(torch.uint8),
        k_scale,
        guarded_page_table,
        weights,
        row_to_batch,
        row_starts,
        c4_seq_lens.reshape(-1).to(torch.int32).contiguous(),
        max_seq_len,
        topk=topk,
        weight_scale=weight_scale,
        workspace=scratch.workspace,
        out=result,
        enforce_status=True,
    )
    return scratch


def run_aiter_fp4_litetopk_chunks(
    *,
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    guarded_page_table: torch.Tensor,
    row_to_batch: torch.Tensor,
    row_starts: torch.Tensor,
    c4_seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    weight_scale: float,
    out_page_indices: torch.Tensor,
    out_raw_indices: torch.Tensor,
    rows_per_chunk: int,
    scratch: FP4LiteTopKScratch,
) -> FP4LiteTopKScratch:
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    query_rows = q_fp4.shape[0]
    for start in range(0, query_rows, rows_per_chunk):
        rows = slice(start, min(start + rows_per_chunk, query_rows))
        scratch = aiter_fp4_litetopk(
            q_fp4=q_fp4[rows],
            q_scale=q_scale[rows],
            k_payload=k_payload,
            k_scale=k_scale,
            weights=weights[rows],
            guarded_page_table=guarded_page_table,
            row_to_batch=row_to_batch[rows],
            row_starts=row_starts[rows],
            c4_seq_lens=c4_seq_lens[rows],
            max_seq_len=max_seq_len,
            topk=topk,
            weight_scale=weight_scale,
            out_page_indices=out_page_indices[rows, :topk],
            out_raw_indices=out_raw_indices[rows, :topk],
            scratch=scratch,
        )
    return scratch


def fp4_litetopk_rows_per_chunk() -> int:
    return _MAX_WORKSPACE_ROWS
