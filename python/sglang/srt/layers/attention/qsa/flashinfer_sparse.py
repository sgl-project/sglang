"""FlashInfer QToken-KvBlock-Sparse-Attention for QSA decode (Qwen4-Exp).

The default QSA decode path gathers the selected K/V rows into a page-aligned
scratch (``fused_kv_prepare``) and runs FlashInfer's dense trtllm-gen paged
decode over the copy, so every selected byte is read, written and read again.
FlashInfer's prims-ts ``QTokenKvBlockSparsePagedTSWrapper`` reads the selected
blocks in place from the paged pool instead: one CuTe-DSL launch per layer plus
a split-KV reducer, no scratch and no gather. It consumes the indexer's deferred
block ids ``[rows, block_topk]`` directly and derives the causal tail of the
open block from the sequence lengths, which is what ``expand_qsa_block_indices``
produces (``count_current_block_in_topk=True``).

Scope: CUDA compute capability 10.x (SM100/SM103/SM107), an unquantized BF16
``MHATokenToKVPool`` storage (flat ``[slots, Hkv, D]`` with ``page_size`` slots
per physical page), decode with one query row per request, a power-of-two
compress ratio in [4, 128] and ``block_topk <= 512``. Everything else keeps the
default path. Opt in with ``--qsa-decode-attention-backend flashinfer_sparse``.
FP8 pools are left to a follow-up: the kernel has no BF16-query/FP8-KV route, so
they would take an FP8 query, which changes the numerics of the default path.

The wrapper plans capacity per batch size and binds the live tensor ABI on its
first eager run; SGLang's CUDA-graph capture is preceded by two eager warmups,
so every captured batch size is planned before capture.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, NamedTuple, Tuple

import torch

from sglang.srt.utils import is_cuda, is_sm100_supported

logger = logging.getLogger(__name__)

_MAX_BLOCK_TOPK = 512
_SPARSE_BLOCK_SIZES = (4, 8, 16, 32, 64, 128)


@lru_cache(maxsize=1)
def resolve_flashinfer_sparse():
    """The prims-ts wrapper class and workspace sizer, or None when unavailable.

    prims-ts compiles for compute capability 10.0, 10.3 and 10.7, the set
    ``is_sm100_supported`` admits (major 10, CUDA >= 12.8).
    """
    if not is_cuda() or not is_sm100_supported():
        return None
    try:
        from flashinfer.attention.prims_ts import (
            QTokenKvBlockSparsePagedTSWrapper,
            get_q_token_kv_block_sparse_workspace_size,
        )
    except ImportError:
        return None
    return QTokenKvBlockSparsePagedTSWrapper, get_q_token_kv_block_sparse_workspace_size


def flashinfer_sparse_supports(
    *, compress_ratio: int, block_topk: int, head_dim: int, page_size: int
) -> bool:
    """Whether the prims-ts kernel covers this QSA geometry."""
    return (
        compress_ratio in _SPARSE_BLOCK_SIZES
        and 0 < block_topk <= _MAX_BLOCK_TOPK
        and head_dim in (64, 128, 256)
        and page_size >= 4
        and page_size % 4 == 0
        and page_size % compress_ratio == 0
    )


def plan_rows(rows: int) -> int:
    """Batch-size bucket for eager decode, so plans stay few while padding stays small.

    CUDA-graph batches are already padded to their capture size and plan exactly;
    this ladder covers eager decode above the graph range. Padded rows are
    inactive (length 0) and cost the kernel almost nothing.
    """
    if rows <= 16:
        return rows
    if rows <= 64:
        return -(-rows // 8) * 8
    if rows <= 256:
        return -(-rows // 32) * 32
    return -(-rows // 64) * 64


class _Plan(NamedTuple):
    wrapper: object
    workspace: torch.Tensor
    out: torch.Tensor


class _Padded(NamedTuple):
    q: torch.Tensor
    seq_lens: torch.Tensor
    block_ids: torch.Tensor


class FlashInferSparseDecode:
    """Planned prims-ts wrappers, one per (batch size, layer), over the paged pool."""

    def __init__(
        self,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        compress_ratio: int,
        block_topk: int,
        max_context_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        resolved = resolve_flashinfer_sparse()
        if resolved is None:
            raise RuntimeError(
                "FlashInfer prims-ts QToken sparse attention is unavailable on this device"
            )
        if not flashinfer_sparse_supports(
            compress_ratio=compress_ratio,
            block_topk=block_topk,
            head_dim=head_dim,
            page_size=page_size,
        ):
            raise ValueError(
                "FlashInfer sparse decode does not support this QSA geometry: "
                f"ratio={compress_ratio}, block_topk={block_topk}, head_dim={head_dim}, "
                f"page_size={page_size}"
            )
        if max_context_len <= 0:
            raise ValueError("max_context_len must be positive")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"FlashInfer sparse decode takes a BF16 or FP16 pool and query, got {dtype}"
            )
        self._wrapper_cls, self._workspace_size = resolved
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.compress_ratio = compress_ratio
        self.block_topk = block_topk
        self.max_context_len = max_context_len
        self.max_pages = -(-max_context_len // page_size)
        self.device = torch.device(device)
        self.dtype = dtype
        # Each wrapper binds one layer's K/V storage, so plans are per (rows, layer).
        # Workspaces stay under 1 MiB each (<= 512 rows, 256K context).
        self._plans: Dict[Tuple[int, int], _Plan] = {}
        self._tables: Dict[int, torch.Tensor] = {}
        self._padded: Dict[int, _Padded] = {}

    def hnd_view(self, buffer: torch.Tensor) -> torch.Tensor:
        """``[slots, Hkv, D]`` pool -> HND ``[pages, Hkv, page_size, D]`` view."""
        slots = buffer.shape[0]
        if slots % self.page_size:
            raise ValueError(
                f"KV pool has {slots} slots, not a multiple of page_size {self.page_size}"
            )
        return buffer.view(
            slots // self.page_size, self.page_size, self.num_kv_heads, self.head_dim
        ).permute(0, 2, 1, 3)

    def block_table(self, page_table: torch.Tensor, rows: int) -> torch.Tensor:
        """An Int32 ``[rows, >= max_pages]`` table of physical page ids.

        CUDA-graph metadata already holds a full-width table that passes through.
        The eager ``compressed_decode_view`` table is only as wide as the batch's
        longest request and is copied into a cached full-width buffer; entries
        past a request's length are never read.
        """
        if (
            page_table.shape[0] == rows
            and page_table.shape[1] >= self.max_pages
            and page_table.dtype == torch.int32
            and page_table.stride(1) == 1
        ):
            return page_table
        table = self._tables.get(rows)
        if table is None:
            table = torch.zeros(
                (rows, self.max_pages), dtype=torch.int32, device=self.device
            )
            self._tables[rows] = table
        width = min(page_table.shape[1], self.max_pages)
        table[: page_table.shape[0], :width].copy_(page_table[:, :width])
        return table

    def _padded_inputs(
        self,
        rows: int,
        q5: torch.Tensor,
        block_ids: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad an eager batch to its plan size; padded rows are inactive."""
        padded = self._padded.get(rows)
        if padded is None:
            padded = _Padded(
                q=torch.zeros(
                    (rows, 1, 1, self.num_qo_heads, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                seq_lens=torch.zeros(rows, dtype=torch.int32, device=self.device),
                block_ids=torch.full(
                    (rows, self.block_topk), -1, dtype=torch.int32, device=self.device
                ),
            )
            self._padded[rows] = padded
        live = q5.shape[0]
        padded.q[:live].copy_(q5)
        padded.seq_lens[:live].copy_(seq_lens)
        padded.seq_lens[live:].zero_()
        padded.block_ids[:live].copy_(block_ids)
        return padded.q, padded.block_ids, padded.seq_lens

    def _plan(
        self,
        rows: int,
        layer_id: int,
        q5: torch.Tensor,
        k_hnd: torch.Tensor,
        table: torch.Tensor,
    ) -> _Plan:
        plan = self._plans.get((rows, layer_id))
        if plan is not None:
            return plan
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInfer sparse decode must run once eagerly for batch size "
                f"{rows} before CUDA-graph capture"
            )
        workspace_bytes = self._workspace_size(
            q5,
            k_hnd,
            table,
            block_topk=self.block_topk,
            max_seq_len_kv=self.max_context_len,
            out_dtype=self.dtype,
            group_size=1,
            kv_block_size=self.compress_ratio,
            split_kv=True,
        )
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=self.device)
        wrapper = self._wrapper_cls()
        wrapper.plan(
            rows,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.compress_ratio,
            self.page_size,
            self.block_topk,
            self.max_context_len,
            device=self.device,
            workspace_buffer=workspace,
            group_size=1,
            packed_query=False,
            split_kv=True,
            q_dtype=self.dtype,
            kv_dtype=self.dtype,
            out_dtype=self.dtype,
        )
        out = torch.empty(
            (rows, 1, 1, self.num_qo_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        plan = _Plan(wrapper, workspace, out)
        self._plans[(rows, layer_id)] = plan
        logger.debug(
            "QSA FlashInfer sparse decode planned: rows=%d layer=%d workspace=%d B",
            rows,
            layer_id,
            workspace_bytes,
        )
        return plan

    def forward(
        self,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        page_table: torch.Tensor,
        block_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        layer_id: int,
        *,
        exact_rows: bool = False,
    ) -> torch.Tensor:
        """Sparse decode attention over the pool; returns ``[rows, Hq * D]``.

        ``page_table`` maps each request's logical ``page_size``-token pages to
        physical pool pages, ``block_ids`` are the indexer's compressed block ids
        ``[rows, block_topk]`` (valid ids first, ``-1`` padding after), and
        ``seq_lens`` are the Int32 KV lengths including the query token.
        ``exact_rows`` plans for exactly ``rows`` (CUDA-graph batches); otherwise
        eager batches are padded to :func:`plan_rows`.
        """
        rows = q.shape[0]
        q5 = q.reshape(rows, 1, 1, self.num_qo_heads, self.head_dim)
        planned_rows = rows if exact_rows else plan_rows(rows)
        if planned_rows != rows:
            q5, block_ids, seq_lens = self._padded_inputs(
                planned_rows, q5, block_ids, seq_lens
            )
        k_hnd = self.hnd_view(k_buffer)
        v_hnd = self.hnd_view(v_buffer)
        table = self.block_table(page_table, planned_rows)
        plan = self._plan(planned_rows, layer_id, q5, k_hnd, table)
        plan.wrapper.run(
            q5,
            (k_hnd, v_hnd),
            table,
            block_ids,
            seq_lens=seq_lens,
            sm_scale=sm_scale,
            out=plan.out,
        )
        return plan.out[:rows].view(rows, self.num_qo_heads * self.head_dim)


__all__ = [
    "FlashInferSparseDecode",
    "flashinfer_sparse_supports",
    "plan_rows",
    "resolve_flashinfer_sparse",
]
