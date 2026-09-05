"""Bounded logits workspace for the DeepSeek-V4 HIP FP4 indexer.

The FP4 indexer materializes an FP32 ``[query_rows, c4_context]`` tensor
between its score and top-k kernels.  The storage is deliberately persistent:
reallocating a slightly wider tensor for every growing-context forward leaves
unusable size classes in the caching allocator.  Unlike the old module-global
2 GiB slab, this workspace has an explicit owner, a workload-aware capacity,
and a stream-ordered lease lifetime.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import torch

FP32_BYTES = 4
KV_BLOCK_SIZE = 64
PAGE_TABLE_GROUP = 4
MAX_FUSED_ROWS = 4096


def guarded_page_table_width(logical_width: int) -> int:
    """Page columns after 256-token scheduling alignment."""
    if logical_width < 0:
        raise ValueError(f"logical_width must be non-negative, got {logical_width}")
    return max(
        PAGE_TABLE_GROUP, -(-logical_width // PAGE_TABLE_GROUP) * PAGE_TABLE_GROUP
    )


def fp4_logits_width_from_page_table(logical_width: int) -> int:
    """Return the padded C4 logits width for a page-table width."""
    return guarded_page_table_width(logical_width) * KV_BLOCK_SIZE


def fp4_logits_width_for_context(context_len: int, page_size: int) -> int:
    """Return the largest padded C4 logits width for a full-token context."""
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    logical_pages = -(-context_len // page_size)
    return fp4_logits_width_from_page_table(logical_pages)


@dataclass(frozen=True)
class FP4LogitsWorkspacePlan:
    """Resolved capacity and the geometry used to derive it."""

    capacity_bytes: int
    desired_bytes: int
    max_seq_len: int
    max_query_rows: int
    rows_at_max_width: int
    limiting_reason: str

    @property
    def capacity_elems(self) -> int:
        return self.capacity_bytes // FP32_BYTES


def plan_fp4_logits_workspace(
    *,
    max_seq_len: int,
    max_query_rows: int,
    runtime_headroom_bytes: int,
    free_memory_fraction: float,
    max_workspace_bytes: Optional[int],
    max_fused_rows: int = MAX_FUSED_ROWS,
) -> FP4LogitsWorkspacePlan:
    """Plan a persistent workspace without exceeding runtime headroom.

    The capacity is rounded down to complete rows at the widest supported
    context.  A configured MB value is a ceiling, not the amount allocated.
    """
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if max_query_rows <= 0:
        raise ValueError(f"max_query_rows must be positive, got {max_query_rows}")
    if runtime_headroom_bytes <= 0:
        raise ValueError(
            f"runtime_headroom_bytes must be positive, got {runtime_headroom_bytes}"
        )
    if not 0.0 < free_memory_fraction <= 1.0:
        raise ValueError(
            f"free_memory_fraction must be in (0, 1], got {free_memory_fraction}"
        )
    if max_workspace_bytes is not None and max_workspace_bytes <= 0:
        raise ValueError(
            f"max_workspace_bytes must be positive or None, got {max_workspace_bytes}"
        )
    if max_fused_rows <= 0:
        raise ValueError(f"max_fused_rows must be positive, got {max_fused_rows}")

    row_bytes = max_seq_len * FP32_BYTES
    desired_rows = min(max_query_rows, max_fused_rows)
    desired_bytes = desired_rows * row_bytes
    candidates = [
        ("workload", desired_bytes),
        ("runtime_headroom", int(runtime_headroom_bytes * free_memory_fraction)),
    ]
    if max_workspace_bytes is not None:
        candidates.append(("user_ceiling", max_workspace_bytes))

    limiting_reason, raw_capacity = min(candidates, key=lambda item: item[1])
    rows_at_max_width = raw_capacity // row_bytes
    if rows_at_max_width < 1:
        raise ValueError(
            "The DeepSeek-V4 FP4 logits workspace cannot hold one row at the "
            f"configured maximum context: row={row_bytes / (1 << 20):.2f} MiB, "
            f"budget={raw_capacity / (1 << 20):.2f} MiB. Increase runtime "
            "headroom or SGLANG_DSV4_FP4_LOGITS_BUDGET_MB."
        )

    capacity_bytes = min(rows_at_max_width, desired_rows) * row_bytes
    return FP4LogitsWorkspacePlan(
        capacity_bytes=capacity_bytes,
        desired_bytes=desired_bytes,
        max_seq_len=max_seq_len,
        max_query_rows=max_query_rows,
        rows_at_max_width=capacity_bytes // row_bytes,
        limiting_reason=limiting_reason,
    )


def limit_plan_to_available_memory(
    plan: FP4LogitsWorkspacePlan,
    available_bytes: int,
    *,
    safety_fraction: float = 0.8,
) -> FP4LogitsWorkspacePlan:
    """Shrink a profiled plan if live free memory is unexpectedly lower."""
    if available_bytes <= 0:
        raise ValueError(f"available_bytes must be positive, got {available_bytes}")
    if not 0.0 < safety_fraction <= 1.0:
        raise ValueError(f"safety_fraction must be in (0, 1], got {safety_fraction}")
    live_cap = int(available_bytes * safety_fraction)
    if plan.capacity_bytes <= live_cap:
        return plan
    row_bytes = plan.max_seq_len * FP32_BYTES
    live_rows = live_cap // row_bytes
    if live_rows < 1:
        raise RuntimeError(
            "Live free memory cannot safely hold one DeepSeek-V4 FP4 logits row: "
            f"row={row_bytes} bytes, free={available_bytes} bytes"
        )
    capacity_bytes = live_rows * row_bytes
    return FP4LogitsWorkspacePlan(
        capacity_bytes=capacity_bytes,
        desired_bytes=plan.desired_bytes,
        max_seq_len=plan.max_seq_len,
        max_query_rows=plan.max_query_rows,
        rows_at_max_width=live_rows,
        limiting_reason="live_free_memory",
    )


class FP4LogitsLease:
    """A borrowed workspace view valid through the consumer's top-k enqueue."""

    def __init__(
        self,
        workspace: FP4LogitsWorkspace,
        tensor: torch.Tensor,
        stream,
    ):
        self._workspace = workspace
        self.tensor = tensor
        self._stream = stream
        self._released = False

    def __enter__(self) -> torch.Tensor:
        return self.tensor

    def release(self) -> None:
        if not self._released:
            self._workspace._release(self._stream)
            self._released = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class FP4LogitsWorkspace:
    """One bounded arena whose reuse is ordered across CUDA/HIP streams."""

    def __init__(
        self,
        *,
        plan: FP4LogitsWorkspacePlan,
        device: torch.device | str,
    ):
        self.plan = plan
        self.device = torch.device(device)
        self._condition = threading.Condition()
        self._in_use = False
        self._closed = False
        self._last_stream = None
        self._reuse_event = (
            torch.cuda.Event(blocking=False) if self.device.type == "cuda" else None
        )
        self._storage: Optional[torch.Tensor] = torch.empty(
            plan.capacity_elems,
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def capacity_bytes(self) -> int:
        return self.plan.capacity_bytes

    @property
    def data_ptr(self) -> int:
        storage = self._require_storage()
        return storage.data_ptr()

    def rows_per_chunk(
        self, max_seq_len: int, *, max_rows: int = MAX_FUSED_ROWS
    ) -> int:
        """Return a strict whole-row chunk size for ``max_seq_len``."""
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if max_seq_len > self.plan.max_seq_len:
            raise RuntimeError(
                "Runtime FP4 logits width exceeds the workspace plan: "
                f"runtime={max_seq_len}, planned={self.plan.max_seq_len}"
            )
        if max_rows <= 0:
            raise ValueError(f"max_rows must be positive, got {max_rows}")
        rows = self.plan.capacity_elems // max_seq_len
        if rows < 1:
            required = max_seq_len * FP32_BYTES
            raise RuntimeError(
                "The FP4 logits workspace cannot hold one runtime row: "
                f"required={required} bytes, capacity={self.capacity_bytes} bytes"
            )
        return min(rows, max_rows)

    def acquire(
        self,
        rows: int,
        max_seq_len: int,
        *,
        stream=None,
    ) -> FP4LogitsLease:
        """Borrow a shaped view and order it after the prior stream consumer."""
        if rows <= 0:
            raise ValueError(f"rows must be positive, got {rows}")
        if max_seq_len <= 0 or max_seq_len > self.plan.max_seq_len:
            raise RuntimeError(
                "FP4 logits lease width is outside the workspace plan: "
                f"runtime={max_seq_len}, planned={self.plan.max_seq_len}"
            )
        required_elems = rows * max_seq_len
        if required_elems > self.plan.capacity_elems:
            raise RuntimeError(
                "FP4 logits lease exceeds the planned capacity: "
                f"required={required_elems * FP32_BYTES} bytes, "
                f"capacity={self.capacity_bytes} bytes"
            )
        if self.device.type == "cuda":
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "The eager FP4 logits workspace must not be used during graph capture"
                )
            if stream is None:
                stream = torch.cuda.current_stream(self.device)

        with self._condition:
            while self._in_use and not self._closed:
                self._condition.wait()
            storage = self._require_storage()
            self._in_use = True
            prior_stream = self._last_stream

        try:
            if (
                prior_stream is not None
                and stream is not None
                and not self._same_stream(prior_stream, stream)
            ):
                # Record lazily at the old stream's current tail. Same-stream
                # reuse already has the required producer -> top-k -> producer
                # ordering and pays no event operations.
                event = self._reuse_event
                event.record(prior_stream)
                stream.wait_event(event)
            tensor = storage[:required_elems].view(rows, max_seq_len)
            return FP4LogitsLease(self, tensor, stream)
        except Exception:
            with self._condition:
                self._in_use = False
                self._condition.notify()
            raise

    def _release(self, stream) -> None:
        if self.device.type == "cuda":
            if stream is None:
                stream = torch.cuda.current_stream(self.device)
        with self._condition:
            self._last_stream = stream
            self._in_use = False
            self._condition.notify()

    def close(self) -> None:
        """Release the arena after all enqueued consumers have completed."""
        with self._condition:
            if self._closed:
                return
            if self._in_use:
                raise RuntimeError(
                    "Cannot close an FP4 logits workspace with an active lease"
                )
            self._closed = True
            last_stream = self._last_stream
        event = self._reuse_event
        if self.device.type == "cuda" and last_stream is not None:
            event.record(last_stream)
            event.synchronize()
        self._storage = None
        self._reuse_event = None

    @staticmethod
    def _same_stream(lhs, rhs) -> bool:
        lhs_handle = getattr(lhs, "cuda_stream", None)
        rhs_handle = getattr(rhs, "cuda_stream", None)
        if lhs_handle is not None and rhs_handle is not None:
            return lhs_handle == rhs_handle
        return lhs is rhs

    def _require_storage(self) -> torch.Tensor:
        if self._closed or self._storage is None:
            raise RuntimeError("The FP4 logits workspace is closed")
        return self._storage
