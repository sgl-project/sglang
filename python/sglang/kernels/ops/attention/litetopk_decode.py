"""LiteTopK: exact decode top-k of paged-MQA index scores from a producer histogram.

DeepGEMM's paged MQA logits count every live, non-NaN score of a row into 1024
ordered coarse bins (``histogram=``); ``select.cuh`` uses them to select the exact
top-k as physical KV slots (``table[row, i // page] * page + i % page``), unordered,
padded with ``-1``, ties to the lower slot. A histogram that disagrees with the
scores falls back to an exact radix select, so the result never depends on it.
The selector leaves the histogram and workspace zeroed, so buffers are reused
across calls and CUDA graph replays without a reset. Configurations: FP32 scores,
k = 2048, 64-slot pages (DSA) and BF16, k = 512, 128-slot pages (DeepSeek-V4.1);
launch shapes are tuned for the 148-SM B200.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HISTOGRAM_BINS = 1024
DEFAULT_CANDIDATE_CAPACITY = 8192
# Per row: a 16-byte hand-off state (litetopk::RowState), then the 8-byte candidates.
_ROW_STATE_BYTES = 16
_CANDIDATE_BYTES = 8


class LiteTopKConfig(NamedTuple):
    name: str  # the selector's configuration struct, litetopk::<name>
    topk: int
    page_size: int


FP32_TOP2048 = LiteTopKConfig("Fp32Top2048", 2048, 64)
BF16_TOP512 = LiteTopKConfig("Bf16Top512", 512, 128)


def _workspace_bytes(rows: int, candidate_capacity: int) -> int:
    return rows * (_ROW_STATE_BYTES + _CANDIDATE_BYTES * candidate_capacity)


def unsupported_reason(config: LiteTopKConfig) -> Optional[str]:
    """Why ``config`` cannot run here (device, DeepGEMM producer), or None."""
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return "requires CUDA"
    if torch.cuda.get_device_capability()[0] != 10:
        return "requires an SM100 GPU"
    try:
        import deep_gemm
    except ImportError as e:
        return f"DeepGEMM is not importable ({e})"
    if config is FP32_TOP2048:
        try:
            params = inspect.signature(deep_gemm.fp8_paged_mqa_logits).parameters
        except (AttributeError, TypeError, ValueError):
            params = {}
        if "histogram" not in params:
            return "DeepGEMM's fp8_paged_mqa_logits takes no histogram"
        return None
    # The BF16 producer always takes a histogram.
    for name in ("fp4_paged_mqa_logits_bf16", "get_paged_mqa_logits_bf16_metadata"):
        if getattr(deep_gemm, name, None) is None:
            return f"DeepGEMM has no {name}"
    return None


@cache_once
def _jit_module(config: LiteTopKConfig) -> Module:
    return load_jit(
        "litetopk_decode",
        config.name,
        cuda_files=["litetopk_decode/select.cuh"],
        cuda_wrappers=[("select", f"litetopk::select<litetopk::{config.name}>")],
    )


class LiteTopKStorage:
    """Histogram and workspace buffers for up to ``max_rows`` rows. Plans of fewer
    rows are views whose hand-off states end where the shared candidates begin.
    Serial calls may share a storage; concurrent calls need separate storages."""

    def __init__(
        self,
        config: LiteTopKConfig,
        max_rows: int,
        device: torch.device,
        candidate_capacity: int = DEFAULT_CANDIDATE_CAPACITY,
    ):
        if max_rows < 1 or candidate_capacity < 1:
            raise ValueError("max_rows and candidate_capacity must be positive")
        self.config = config
        self.max_rows = max_rows
        self.candidate_capacity = candidate_capacity
        self.histogram = torch.zeros(
            (max_rows, HISTOGRAM_BINS), dtype=torch.int32, device=device
        )
        self.workspace = torch.zeros(
            _workspace_bytes(max_rows, candidate_capacity),
            dtype=torch.uint8,
            device=device,
        )
        self.module = _jit_module(config)

    @property
    def nbytes(self) -> int:
        return self.histogram.nbytes + self.workspace.nbytes

    def plan(self, rows: int) -> LiteTopKPlan:
        return LiteTopKPlan(self, rows)


class LiteTopKPlan:
    """Views of a storage for exactly ``rows`` score rows. The producer adds the
    rows' counts to ``histogram``; the next ``select`` consumes and clears them."""

    def __init__(self, storage: LiteTopKStorage, rows: int):
        if not 1 <= rows <= storage.max_rows:
            raise ValueError(f"rows must be in [1, {storage.max_rows}], got {rows}")
        self.storage = storage
        self.rows = rows
        self.histogram = storage.histogram[:rows]
        offset = (storage.max_rows - rows) * _ROW_STATE_BYTES
        size = _workspace_bytes(rows, storage.candidate_capacity)
        self.workspace = storage.workspace[offset : offset + size]

    def select(
        self,
        scores: torch.Tensor,
        lengths: torch.Tensor,
        table: torch.Tensor,
        *,
        out: torch.Tensor,
        rows_per_table_row: int = 1,
    ) -> torch.Tensor:
        """Writes the physical KV slots of each row's exact top-k to ``out`` (int32
        ``[rows, topk]``) and returns it. ``scores``: ``[rows, width]``, unit column
        stride, 16-byte aligned rows, live columns ``[0, lengths[row])`` (int32,
        clamped to ``[0, width]``). Row ``r`` maps pages through ``table`` row
        ``r // rows_per_table_row``. ``histogram`` must count exactly these scores."""
        self.storage.module.select(
            scores,
            lengths,
            self.histogram,
            table,
            rows_per_table_row,
            out,
            self.workspace,
            self.storage.candidate_capacity,
        )
        return out
