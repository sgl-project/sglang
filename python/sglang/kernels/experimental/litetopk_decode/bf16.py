"""Paired DeepGEMM BF16 producer + exact LiteTopK top-512 on page-128 rows.

Requires the paired producer patch: each histogram counts every live non-NaN
BF16 score once. The selector consumes this histogram in one launch and returns
exact physical slots. Buffers
return to rest after each call. Build and warm plans before CUDA graph capture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

TOPK = 512
PAGE_SIZE = 128
STATE_BYTES = 16


def dsv41_decode_unsupported_reason() -> Optional[str]:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return "requires CUDA"
    if torch.cuda.get_device_capability()[0] != 10:
        return "requires an SM100 GPU"
    try:
        import deep_gemm
    except ImportError as exc:
        return str(exc)
    for name in (
        "get_paged_mqa_logits_bf16_metadata",
        "fp4_paged_mqa_logits_bf16",
    ):
        if not callable(getattr(deep_gemm, name, None)):
            return f"the paired DeepGEMM BF16 API {name} is missing"
    return None


@cache_once
def _load_selector():
    source = Path(__file__).resolve().parent / "bf16_exact" / "select.cuh"
    return load_jit(
        "litetopk_dsv41_bf16_exact",
        cuda_files=[str(source)],
        cuda_wrappers=[("select", "litetopk::select_bf16_top512_page128")],
    )


class Bf16Dsv41DecodeStorage:
    """Shared buffers for serial calls of different row counts.

    Plans place their states before a common candidate region, so states never
    overlap another plan's candidates. Keep a storage alive while captured
    graphs use it; use separate storages for concurrent calls.
    """

    def __init__(
        self, max_rows: int, device, candidate_capacity: int = 8192, *, producer=None
    ):
        if max_rows < 1 or candidate_capacity < 1:
            raise ValueError("max_rows and candidate_capacity must be positive")
        if producer is None:
            import deep_gemm as producer
        self.producer = producer
        self.max_rows = max_rows
        self.candidate_capacity = candidate_capacity
        self.histogram = torch.zeros((max_rows, 1024), dtype=torch.int32, device=device)
        self.workspace = torch.zeros(
            max_rows * (STATE_BYTES + 8 * candidate_capacity),
            dtype=torch.uint8,
            device=device,
        )
        self.output = torch.empty((max_rows, TOPK), dtype=torch.int32, device=device)
        self.selector = _load_selector()

    @property
    def nbytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.histogram, self.workspace, self.output)
        )

    def plan(self, rows: int):
        return Bf16Dsv41DecodePlan(
            rows, self.histogram.device, self.candidate_capacity, storage=self
        )


class Bf16Dsv41DecodePlan:
    """A fixed row count, with the same scores/select split as Dsv41DecodePlan.

    Q, KV, BF16 weights and scheduler metadata use the paged MQA layouts.
    ``indices`` identifies requests shared by DSpark rows and must match the
    indices used to build ``schedule_metadata``. The selector always receives
    one page-table row and one causal length per score row.
    """

    def __init__(
        self,
        rows: int,
        device,
        candidate_capacity: int = 8192,
        *,
        storage: Optional[Bf16Dsv41DecodeStorage] = None,
        producer=None,
    ):
        if storage is None:
            storage = Bf16Dsv41DecodeStorage(
                rows, device, candidate_capacity, producer=producer
            )
        if not 1 <= rows <= storage.max_rows:
            raise ValueError(f"rows must be in [1, {storage.max_rows}]")
        if candidate_capacity != storage.candidate_capacity:
            raise ValueError("the storage has another candidate capacity")
        if producer is not None and producer is not storage.producer:
            raise ValueError("the storage has another producer")
        self.rows, self.candidate_capacity, self.storage = (
            rows,
            candidate_capacity,
            storage,
        )
        self.histogram = storage.histogram[:rows]
        offset = (storage.max_rows - rows) * STATE_BYTES
        size = rows * (STATE_BYTES + 8 * candidate_capacity)
        self.workspace = storage.workspace[offset : offset + size]
        self.output = storage.output[:rows]
        self.selector = storage.selector

    def scores(
        self,
        *,
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_context_len: int,
        indices=None,
        tokens_per_request: int = 1,
    ):
        if weights.dtype != torch.bfloat16:
            raise ValueError("the BF16 path requires BF16 MQA weights")
        if tuple(context_lens.shape) != (self.rows, 1):
            raise ValueError("context_lens must be [rows, 1]")
        if block_table.shape[0] != self.rows:
            raise ValueError("block_table must have one row per score row")
        logits = self.storage.producer.fp4_paged_mqa_logits_bf16(
            q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_metadata,
            max_context_len,
            indices=indices,
            histogram=self.histogram,
            tokens_per_request=tokens_per_request,
        )
        if logits.dtype != torch.bfloat16:
            raise RuntimeError("the paired DeepGEMM producer must return BF16 logits")
        return logits

    def select(self, *, scores, context_lens, block_table, out=None):
        """Exact slots by (BF16 ordered key descending, physical slot ascending).

        Output is unordered int32 [rows, 512], padded with -1. The histogram must
        be from the immediately preceding scores call, with no external edits.
        """
        out = self.output if out is None else out
        self.selector.select(
            scores,
            context_lens.view(-1),
            self.histogram,
            block_table,
            1,
            out,
            self.workspace,
            self.candidate_capacity,
        )
        return out

    def __call__(
        self,
        *,
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_context_len: int,
        indices=None,
        tokens_per_request: int = 1,
        out=None,
    ):
        logits = self.scores(
            q=q,
            kv_cache=kv_cache,
            weights=weights,
            context_lens=context_lens,
            block_table=block_table,
            schedule_metadata=schedule_metadata,
            max_context_len=max_context_len,
            indices=indices,
            tokens_per_request=tokens_per_request,
        )
        slots = self.select(
            scores=logits, context_lens=context_lens, block_table=block_table, out=out
        )
        return logits, slots
