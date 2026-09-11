from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Iterator, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import (
    mqa_logits_budget_bytes,
    mqa_logits_needs_budget_check,
    mqa_logits_row_bytes,
    mqa_logits_rows_per_chunk,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import is_hip, is_sm120_supported, is_xpu

logger = logging.getLogger(__name__)

_IS_SM120 = is_sm120_supported()

"""
Some comments on the common terms used in DeepSeekV4Backend:

topk_lengths:
    NOTE: TL;DR: topk_lengths == seq_lens
    The FlashMLA sparse decode kernel will attend to `k` tokens for each query.
    `topk_lengths` indicates how many tokens each query will attend to.
    This should be named as `seq_lens`, but we simply follow the naming convention.

page_table:
    The page table indicates which pages each request is assigned to.
    Each value in the page table is the page index in the TokenToKVPool.
    This page index is irrelevant to the actual `page_size`.

page_indices:
    The real indices used to index into the KV cache.
    This can be computed from the `page_table` and `page_size`.
    e.g. page_indices[i, j] = page_table[i, j // page_size] * page_size + (j % page_size)
    For sparse C4 top-512 attention, the indices will be selected from the C4 page indices.
    In implementation, we don't materialize the full C4 `page_indices`,
    but calculate them from `page_table` on-the-fly in the attention kernel.

positions:
    The position of the last token for each request.
    For compress token, the positions must be times of compress ratio.
    For example, for C4, raw_position=11 will trigger a compression,
    But the RoPE's position, during compression, must be 8 instead of 11.

Some other notes:
    c4_ / c128_: means "compressed by 4" / "compressed by 128".
    compressed_page_size: physical indexer pool page size
    compressed_seq_lens: seq_lens // 4, but bounded by at least 1, due to flash_mla requirement.
    c4_sparse: means "compressed by 4" but only attend to top-512 tokens.
               all related length will be clipped to 512.
"""
_LARGE_INDEXER_QUERY_THRESHOLD = 11673

# DeepGEMM's paged-MQA metadata kernel cannot schedule more rows than this on
# SM120 (shared-memory cap), so SM120 always splits larger batches.
_SM120_INDEXER_M_CHUNK = 4096


def plan_indexer_row_chunks(
    *,
    num_rows: int,
    num_cols: int,
    budget_bytes: Optional[int],
    sm120_row_cap: Optional[int],
) -> Optional[int]:
    """Query rows per paged-indexer chunk; None runs the whole batch in one call.

    The fp32 logits are [num_rows, num_cols] per layer, so the chunk is the
    smaller of the SM120 kernel cap and what the memory budget allows.
    """
    rows_per_chunk = None
    if sm120_row_cap is not None and num_rows > sm120_row_cap:
        rows_per_chunk = sm120_row_cap
    if budget_bytes is not None:
        by_budget = mqa_logits_rows_per_chunk(
            num_rows=num_rows,
            row_bytes=mqa_logits_row_bytes(num_cols),
            budget_bytes=budget_bytes,
        )
        if by_budget is not None:
            rows_per_chunk = (
                by_budget if rows_per_chunk is None else min(rows_per_chunk, by_budget)
            )
    return rows_per_chunk


def iter_row_chunks(*, num_rows: int, rows_per_chunk: Optional[int]) -> Iterator[slice]:
    if rows_per_chunk is None or rows_per_chunk >= num_rows:
        yield slice(0, num_rows)
        return
    for start in range(0, num_rows, rows_per_chunk):
        yield slice(start, min(start + rows_per_chunk, num_rows))


def copy_metadata(
    *,
    src,
    dst,
    check_eq_fields: List[str],
    copy_fields: List[str],
    assign_fields: Optional[List[str]] = None,
):
    assign_fields = assign_fields or []

    for field_name in check_eq_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        assert src_val == dst_val, f"{field_name=} {src_val=} {dst_val=}"

    for field_name in copy_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        if src_val is None and dst_val is None:
            continue
        assert dst_val is not None, f"{field_name=} {src_val=} {dst_val=}"
        if hasattr(dst_val, "copy_"):
            dst_val.copy_(src_val)
        else:
            warnings.warn(
                f"{field_name=} {type(dst_val)=} does not have copy_, use setattr"
            )
            setattr(dst, field_name, src_val)

    for field_name in assign_fields:
        setattr(dst, field_name, getattr(src, field_name))

    provided_fields = check_eq_fields + copy_fields + assign_fields
    provided_fields_unique = set(provided_fields)
    assert len(provided_fields) == len(provided_fields_unique), (
        f"{provided_fields=} has dup"
    )
    all_fields = {f.name for f in fields(src)}
    provided_fields = set(provided_fields)
    assert provided_fields == all_fields, (
        f"{provided_fields - all_fields=}, {all_fields - provided_fields=}"
    )


@dataclass
class NonPagedIndexerPlan:
    page_table: torch.Tensor
    gather_seq_lens: torch.Tensor
    ks: torch.Tensor
    ke: torch.Tensor
    seq_len_sum: int
    max_seq_len: int
    max_seqlen_k: int
    query_rows: int
    # None runs all query rows in one fp8_mqa_logits call.
    rows_per_chunk: Optional[int] = None


@dataclass
class PagedIndexerMetadata:
    page_size: int
    compressed_page_size: int
    page_table: torch.Tensor
    compressed_seq_lens: torch.Tensor
    use_topk_v2: bool
    force_deep_gemm_metadata: bool = False
    use_prefill_cuda_graph: bool = False
    # A list when the forward is row-chunked: one schedule per chunk.
    deep_gemm_metadata: Any = field(init=False, repr=False)
    topk_metadata: torch.Tensor = field(init=False, repr=False)
    nonpaged_plan: Optional[NonPagedIndexerPlan] = field(
        init=False, repr=False, default=None
    )
    # Decided once per forward and shared by every layer's indexer call.
    rows_per_chunk: Optional[int] = field(init=False, repr=False, default=None)
    mqa_logits_budget_bytes: Optional[int] = field(init=False, repr=False, default=None)
    # The top-k v2 plan routes rows by index within the batch it was built for,
    # so a row-chunked forward needs one plan per chunk.
    topk_metadata_chunks: Optional[List[torch.Tensor]] = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):
        if (
            is_hip() or is_xpu() or envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        ) and not self.force_deep_gemm_metadata:
            self.deep_gemm_metadata = None
        else:
            import deep_gemm

            use_jit_indexer = not self.force_deep_gemm_metadata and (
                envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.get()
                or self.compressed_seq_lens.numel() > _LARGE_INDEXER_QUERY_THRESHOLD
            )
            if use_jit_indexer:
                from sglang.kernels.ops.attention.dsv4 import (
                    get_paged_mqa_logits_metadata,
                )
            else:
                from deep_gemm import get_paged_mqa_logits_metadata

            compressed_seq_lens = self.compressed_seq_lens.to(torch.int32)
            if compressed_seq_lens.dim() == 1:
                compressed_seq_lens = compressed_seq_lens.unsqueeze(-1)
            num_rows = compressed_seq_lens.shape[0]
            self.mqa_logits_budget_bytes = self._mqa_logits_budget(num_rows=num_rows)
            self.rows_per_chunk = plan_indexer_row_chunks(
                num_rows=num_rows,
                num_cols=self.max_compressed_seq_len,
                budget_bytes=self.mqa_logits_budget_bytes,
                sm120_row_cap=_SM120_INDEXER_M_CHUNK if _IS_SM120 else None,
            )
            if self.rows_per_chunk is not None:
                logger.debug(
                    "DSV4 indexer chunks %d query rows x %d compressed cols into "
                    "%d-row chunks (logits budget %s bytes)",
                    num_rows,
                    self.max_compressed_seq_len,
                    self.rows_per_chunk,
                    self.mqa_logits_budget_bytes,
                )
                # Chunk metadata is shared by all indexer layers in this forward.
                self.deep_gemm_metadata = [
                    get_paged_mqa_logits_metadata(
                        compressed_seq_lens[rows],
                        self.compressed_page_size,
                        deep_gemm.get_num_sms(),
                    )
                    for rows in iter_row_chunks(
                        num_rows=num_rows, rows_per_chunk=self.rows_per_chunk
                    )
                ]
            else:
                self.deep_gemm_metadata = get_paged_mqa_logits_metadata(
                    compressed_seq_lens,
                    self.compressed_page_size,
                    deep_gemm.get_num_sms(),
                )

            assert isinstance(self.deep_gemm_metadata, (torch.Tensor, list))

        if self.use_topk_v2:
            from sglang.kernels.ops.attention.dsv4 import plan_topk_v2

            self.topk_metadata = plan_topk_v2(self.compressed_seq_lens)
            if self.rows_per_chunk is not None:
                self.topk_metadata_chunks = [
                    plan_topk_v2(self.compressed_seq_lens[rows])
                    for rows in iter_row_chunks(
                        num_rows=self.compressed_seq_lens.shape[0],
                        rows_per_chunk=self.rows_per_chunk,
                    )
                ]
        else:
            self.topk_metadata = torch.empty((0,))

        assert self.page_size == 256, "the system hardcodes page_size=256"

    def _mqa_logits_budget(self, *, num_rows: int) -> Optional[int]:
        """Free-memory budget for this forward's logits; None disables chunking.

        Graph-backed forwards keep a single call: their shapes are fixed at
        capture and the free-memory read would sync the host mid-capture.
        """
        if self.use_prefill_cuda_graph or not self.compressed_seq_lens.is_cuda:
            return None
        if not mqa_logits_needs_budget_check(
            num_rows=num_rows, num_cols=self.max_compressed_seq_len
        ):
            return None
        if (
            torch.cuda.is_current_stream_capturing()
            or is_in_breakable_cuda_graph()
            or is_in_tc_piecewise_cuda_graph()
        ):
            return None
        return mqa_logits_budget_bytes(
            device_index=self.compressed_seq_lens.device.index, allow_sync=True
        )

    @property
    def max_seq_len(self) -> int:
        return self.page_table.shape[1] * self.page_size

    @property
    def max_compressed_seq_len(self) -> int:
        return self.page_table.shape[1] * self.compressed_page_size

    def copy_(self, other: PagedIndexerMetadata):
        # A chunked schedule list has no in-place copy; rebind it instead.
        chunked = isinstance(self.deep_gemm_metadata, list) or isinstance(
            other.deep_gemm_metadata, list
        )
        if is_hip() or chunked:
            copy_fields = ["page_table", "compressed_seq_lens"]
            assign_fields = ["deep_gemm_metadata", "nonpaged_plan"]
        else:
            copy_fields = ["page_table", "compressed_seq_lens", "deep_gemm_metadata"]
            assign_fields = ["nonpaged_plan"]
        copy_fields += ["topk_metadata"]
        assign_fields += [
            "rows_per_chunk",
            "mqa_logits_budget_bytes",
            "topk_metadata_chunks",
        ]
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[
                "page_size",
                "compressed_page_size",
                "force_deep_gemm_metadata",
                "use_prefill_cuda_graph",
                "use_topk_v2",
            ],
            copy_fields=copy_fields,
            assign_fields=assign_fields,
        )
        self.nonpaged_plan = None


def maybe_copy_inplace(dst, *, src) -> None:
    assert type(src) == type(dst)
    if dst is not None:
        dst.copy_(src)
