"""CUDA-graph-safe, per-row tracing for Kimi-K3 target verification.

Python model hooks run while a graph is captured, but not when it is replayed.
The helpers below therefore capture small device-side result tensors and keep
Python references to them.  Every graph replay refreshes those tensors; the
NPU graph runner reads them only after replay and only when the operator marker
exists.

No collectives are introduced by this module.  Each rank logs its local rows so
the current branch and the 0728 baseline can be compared rank-for-rank without
changing the graph's communication order.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_TRACE_ENV = "SGLANG_K3_GRAPH_ROW_TRACE"
_TRACE_MARKER_ENVS = (
    "SGLANG_K3_TRACE_HIDDEN_FILE",
    "SGLANG_K3_TRACE_STATE_FILE",
)


@dataclass
class _RowTraceBuffer:
    mode: str
    capture_bs: int
    capture_num_tokens: int
    layer_id: int
    stage: str
    source_shape: tuple[int, ...]
    row_dim: int
    row_kind: str
    row_start: int = 0
    stats: Optional[torch.Tensor] = None
    exact: Optional[torch.Tensor] = None


_ROW_TRACE_BUFFERS: dict[
    tuple[str, int, int, int, str, str], _RowTraceBuffer
] = {}


def graph_row_trace_enabled() -> bool:
    return os.environ.get(_TRACE_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def graph_row_trace_marker_enabled() -> bool:
    for name in _TRACE_MARKER_ENVS:
        marker = os.environ.get(name)
        if marker and os.path.exists(marker):
            return True
    return False


def _mode_name(forward_batch) -> str:
    mode = forward_batch.forward_mode
    return getattr(mode, "name", str(mode)).lower()


def _target_verify_trace_key(
    *, forward_batch, layer_id: int, stage: str, row_kind: str
) -> Optional[tuple[str, int, int, int, str, str]]:
    if not graph_row_trace_enabled():
        return None
    if not forward_batch.forward_mode.is_target_verify():
        return None
    return (
        _mode_name(forward_batch),
        int(forward_batch.batch_size),
        int(forward_batch.input_ids.shape[0]),
        int(layer_id),
        str(stage),
        str(row_kind),
    )


def capture_graph_row_stats(
    *,
    forward_batch,
    layer_id: int,
    stage: str,
    tensor: Optional[torch.Tensor],
    row_dim: int = 0,
    row_kind: str = "token",
    row_start: int = 0,
) -> None:
    """Capture five independent summaries for every logical tensor row.

    The retained tensor has columns ``abs_sum, signed_sum, square_sum,
    min, max``.  Unlike a verify-batch checksum, a wrong row selection cannot
    cancel or hide behind correct neighbouring rows.
    """
    key = _target_verify_trace_key(
        forward_batch=forward_batch,
        layer_id=layer_id,
        stage=stage,
        row_kind=row_kind,
    )
    if key is None or tensor is None or tensor.numel() == 0:
        return

    value = tensor.detach()
    normalized_row_dim = row_dim if row_dim >= 0 else value.ndim + row_dim
    if not 0 <= normalized_row_dim < value.ndim:
        raise ValueError(
            f"invalid row_dim={row_dim} for K3 trace tensor shape={tuple(value.shape)}"
        )
    rows = value.movedim(normalized_row_dim, 0)
    flat = rows.reshape(rows.shape[0], -1)
    with torch.no_grad():
        row_abs = flat.abs().sum(dim=-1, dtype=torch.float32)
        row_sum = flat.sum(dim=-1, dtype=torch.float32)
        row_sq = (flat * flat).sum(dim=-1, dtype=torch.float32)
        # aclnnMinDim/aclnnMaxDim cannot be captured by the NPU graph used by
        # K3 target verify.  The comparator triggers only on abs/square sums;
        # keep two graph-safe reserved columns so the log schema stays stable.
        row_min = row_abs * 0.0
        row_max = row_abs * 0.0
        stats = torch.stack((row_abs, row_sum, row_sq, row_min, row_max), dim=-1)

    _ROW_TRACE_BUFFERS[key] = _RowTraceBuffer(
        mode=key[0],
        capture_bs=key[1],
        capture_num_tokens=key[2],
        layer_id=int(layer_id),
        stage=str(stage),
        source_shape=tuple(int(dim) for dim in value.shape),
        row_dim=normalized_row_dim,
        row_kind=str(row_kind),
        row_start=int(row_start),
        stats=stats,
    )


def capture_graph_exact_rows(
    *,
    forward_batch,
    layer_id: int,
    stage: str,
    tensor: Optional[torch.Tensor],
    row_dim: int = 0,
    row_kind: str = "request",
    row_start: int = 0,
) -> None:
    """Capture small integer metadata tensors whose exact values matter."""
    key = _target_verify_trace_key(
        forward_batch=forward_batch,
        layer_id=layer_id,
        stage=stage,
        row_kind=row_kind,
    )
    if key is None or tensor is None or tensor.numel() == 0:
        return

    value = tensor.detach()
    normalized_row_dim = row_dim if row_dim >= 0 else value.ndim + row_dim
    if not 0 <= normalized_row_dim < value.ndim:
        raise ValueError(
            f"invalid row_dim={row_dim} for K3 exact trace shape={tuple(value.shape)}"
        )
    exact = value.movedim(normalized_row_dim, 0).clone()
    _ROW_TRACE_BUFFERS[key] = _RowTraceBuffer(
        mode=key[0],
        capture_bs=key[1],
        capture_num_tokens=key[2],
        layer_id=int(layer_id),
        stage=str(stage),
        source_shape=tuple(int(dim) for dim in value.shape),
        row_dim=normalized_row_dim,
        row_kind=str(row_kind),
        row_start=int(row_start),
        exact=exact,
    )


def _dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def dump_graph_row_traces(
    *,
    replay_id: int,
    mode: str,
    capture_bs: int,
    capture_num_tokens: int,
    raw_bs: int,
    raw_num_tokens: int,
    dense_verify_tokens: Optional[int] = None,
) -> None:
    """Materialize the buffers refreshed by the just-finished graph replay."""
    if not graph_row_trace_marker_enabled():
        return

    normalized_mode = str(mode).lower()
    rank = _dist_rank()
    records = [
        record
        for record in _ROW_TRACE_BUFFERS.values()
        if record.mode == normalized_mode
        and record.capture_bs == int(capture_bs)
        and record.capture_num_tokens == int(capture_num_tokens)
    ]
    records.sort(key=lambda item: (item.layer_id, item.stage, item.row_kind))
    prepared = []
    stat_parts = []
    exact_parts = []
    stat_offset = 0
    exact_offset = 0
    for record in records:
        if record.row_kind == "request":
            total_real_rows = int(raw_bs)
        elif record.row_kind == "dense_verify":
            total_real_rows = int(dense_verify_tokens or 0)
        else:
            total_real_rows = int(raw_num_tokens)

        available_rows = (
            record.stats.shape[0]
            if record.stats is not None
            else record.exact.shape[0]
        )
        real_rows = max(
            0,
            min(int(available_rows), total_real_rows - int(record.row_start)),
        )
        row_indices = list(
            range(int(record.row_start), int(record.row_start) + real_rows)
        )

        if record.stats is not None:
            flat_values = record.stats[:real_rows].reshape(-1)
            stat_parts.append(flat_values)
            prepared.append(
                (record, real_rows, row_indices, "stats", stat_offset, flat_values.numel())
            )
            stat_offset += flat_values.numel()
        elif record.exact is not None:
            flat_values = record.exact[:real_rows].reshape(-1).to(torch.int64)
            exact_parts.append(flat_values)
            prepared.append(
                (record, real_rows, row_indices, "exact", exact_offset, flat_values.numel())
            )
            exact_offset += flat_values.numel()

    # One device-to-host transfer per dtype avoids hundreds of serial NPU
    # synchronizations when all 93 layers are traced.
    stats_cpu = (
        torch.cat(stat_parts).detach().cpu() if stat_parts else torch.empty(0)
    )
    exact_cpu = (
        torch.cat(exact_parts).detach().cpu()
        if exact_parts
        else torch.empty(0, dtype=torch.int64)
    )

    for record, real_rows, row_indices, kind, offset, numel in prepared:
        if kind == "stats":
            values = stats_cpu[offset : offset + numel].view(real_rows, 5)
            logger.warning(
                "K3_GRAPH_ROW_TRACE rank=%d replay_id=%d mode=%s capture_bs=%d "
                "capture_num_tokens=%d layer_id=%d "
                "stage=%s row_kind=%s source_shape=%s row_dim=%d row_start=%d "
                "real_rows=%d row_indices=%s "
                "row_abs=%s row_sum=%s row_sq=%s row_min=%s row_max=%s",
                rank,
                replay_id,
                record.mode,
                record.capture_bs,
                record.capture_num_tokens,
                record.layer_id,
                record.stage,
                record.row_kind,
                record.source_shape,
                record.row_dim,
                record.row_start,
                real_rows,
                row_indices,
                values[:, 0].tolist(),
                values[:, 1].tolist(),
                values[:, 2].tolist(),
                values[:, 3].tolist(),
                values[:, 4].tolist(),
            )
        else:
            tail_shape = record.exact.shape[1:]
            values = exact_cpu[offset : offset + numel].view(
                real_rows, *tail_shape
            ).tolist()
            logger.warning(
                "K3_GRAPH_EXACT_ROW_TRACE rank=%d replay_id=%d mode=%s capture_bs=%d "
                "capture_num_tokens=%d "
                "layer_id=%d stage=%s row_kind=%s source_shape=%s row_dim=%d "
                "row_start=%d real_rows=%d row_indices=%s values=%s",
                rank,
                replay_id,
                record.mode,
                record.capture_bs,
                record.capture_num_tokens,
                record.layer_id,
                record.stage,
                record.row_kind,
                record.source_shape,
                record.row_dim,
                record.row_start,
                real_rows,
                row_indices,
                values,
            )
