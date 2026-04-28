from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta


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
    c4_page_size: page_size // 4
    c4_seq_lens: seq_lens // 4, but bounded by at least 1, due to flash_mla requirement.
    c4_sparse: means "compressed by 4" but only attend to top-512 tokens.
               all related length will be clipped to 512.
"""


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
        assert dst_val is not None, f"{field_name=} {src_val=} {dst_val=}"
        dst_val.copy_(src_val)

    for field_name in assign_fields:
        setattr(dst, field_name, getattr(src, field_name))

    provided_fields = check_eq_fields + copy_fields + assign_fields
    assert len(provided_fields) == len(
        set(provided_fields)
    ), f"{provided_fields=} has dup"
    all_fields = {f.name for f in fields(src)}
    assert set(provided_fields) == all_fields, f"{provided_fields=} {all_fields=}"


def create_flashmla_metadata():
    if is_hip():
        return None
    else:
        import flash_mla

        return flash_mla.get_mla_metadata()[0]


@dataclass
class CoreMetadata:
    positions: torch.Tensor
    swa_slice: Optional[torch.Tensor]
    swa_out_loc_sliced: torch.Tensor
    c4_out_loc: torch.Tensor
    c128_out_loc: torch.Tensor

    def init_swa_slice(self, swa_slice: torch.Tensor):
        assert self.swa_slice is None, "can only update once"
        self.swa_slice = swa_slice
        self.swa_out_loc_sliced = self.swa_out_loc_sliced[swa_slice]

    def copy_(self, other):
        raise NotImplementedError


@dataclass
class IndexerMetadata:
    def copy_(self, other):
        raise NotImplementedError


@dataclass
class PagedIndexerMetadata(IndexerMetadata):
    page_size: int
    page_table: torch.Tensor
    c4_seq_lens: torch.Tensor
    deep_gemm_metadata: Any = field(init=False, repr=False)
    topk_metadata: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        if envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
            self.deep_gemm_metadata = None
        else:
            import deep_gemm

            if envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get() != -1:
                from sglang.srt.layers.deep_gemm_wrapper.paged_mqa_logits import (
                    get_paged_mqa_logits_metadata_chunked as get_paged_mqa_logits_metadata,
                )
            elif envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.get():
                from sglang.jit_kernel.deepseek_v4 import get_paged_mqa_logits_metadata
            else:
                from deep_gemm import get_paged_mqa_logits_metadata

            _c4 = self.c4_seq_lens.to(torch.int32)
            if _c4.dim() == 1:
                _c4 = _c4.unsqueeze(-1)
            self.deep_gemm_metadata = get_paged_mqa_logits_metadata(
                _c4,
                self.c4_page_size,
                deep_gemm.get_num_sms(),
            )

            if envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get() != -1:
                pass
            else:
                assert isinstance(self.deep_gemm_metadata, torch.Tensor)

        from sglang.jit_kernel.deepseek_v4 import plan_topk_v2

        if envs.SGLANG_OPT_USE_TOPK_V2.get():
            self.topk_metadata = plan_topk_v2(self.c4_seq_lens)
        else:
            self.topk_metadata = torch.empty((0,))

        assert self.page_size == 256

    @property
    def c4_page_size(self) -> int:
        return self.page_size // 4

    @property
    def max_seq_len(self) -> int:
        return self.page_table.shape[1] * self.page_size

    @property
    def max_c4_seq_len(self) -> int:
        return self.page_table.shape[1] * self.c4_page_size

    def copy_(self, other: "PagedIndexerMetadata"):
        if is_hip():
            copy_fields = ["page_table", "c4_seq_lens"]
        else:
            copy_fields = ["page_table", "c4_seq_lens", "deep_gemm_metadata"]
        copy_fields += ["topk_metadata"]
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=["page_size"],
            copy_fields=copy_fields,
        )


@dataclass
class PagedCoreMetadata(CoreMetadata):
    page_table: torch.Tensor
    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor
    c128_page_indices: torch.Tensor
    c128_topk_lengths_clamp1: torch.Tensor
    c4_topk_lengths_raw: torch.Tensor
    c4_topk_lengths_clamp1: torch.Tensor
    c4_sparse_topk: int
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)
    c4_sparse_page_indices: torch.Tensor = field(init=False)
    c1_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c4_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c128_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)

    def get_flashmla_metadata(self, compress_ratio: int):
        if compress_ratio == 0:
            return self.c1_flashmla_metadata
        elif compress_ratio == 4:
            return self.c4_flashmla_metadata
        elif compress_ratio == 128:
            return self.c128_flashmla_metadata
        else:
            raise ValueError(f"invalid {compress_ratio=}")

    def __post_init__(self):
        # c4_sparse_topk is set from model_config.index_topk per-model
        # (small model: 512, large model: 1024).
        assert self.c4_sparse_topk in (512, 1024), (
            f"unexpected c4_sparse_topk={self.c4_sparse_topk}; "
            "supported: 512 (small) or 1024 (large)"
        )
        self.c4_sparse_topk_lengths = torch.clamp(
            self.c4_topk_lengths_clamp1, max=self.c4_sparse_topk
        )
        self.c4_sparse_page_indices = torch.full(
            (self.c4_topk_lengths_clamp1.size(0), self.c4_sparse_topk),
            -1,
            dtype=torch.int32,
            device=self.c4_topk_lengths_clamp1.device,
        )
        self.c1_flashmla_metadata = create_flashmla_metadata()
        self.c4_flashmla_metadata = create_flashmla_metadata()
        self.c128_flashmla_metadata = create_flashmla_metadata()

    def copy_(self, other: PagedCoreMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=["c4_sparse_topk", "swa_slice"],
            copy_fields=[
                "positions",
                "swa_out_loc_sliced",
                "c4_out_loc",
                "c128_out_loc",
                "page_table",
                "swa_page_indices",
                "swa_topk_lengths",
                "c128_page_indices",
                "c128_topk_lengths_clamp1",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_page_indices",
            ],
            assign_fields=[
                "c1_flashmla_metadata",
                "c4_flashmla_metadata",
                "c128_flashmla_metadata",
            ],
        )




@dataclass
class RaggedCoreMetadata(CoreMetadata):
    swa_ragged_indices: torch.Tensor
    swa_c4_ragged_indices: torch.Tensor
    swa_c128_ragged_indices: torch.Tensor


@dataclass
class RaggedIndexerMetadata(IndexerMetadata):
    c4_k_start: torch.Tensor
    c4_k_finish: torch.Tensor


@dataclass
class DeepseekV4Metadata:
    core_metadata: CoreMetadata
    indexer_metadata: IndexerMetadata
    debug_seq_lens_expanded: torch.Tensor

    def copy_(self, other: "DeepseekV4Metadata"):
        self.core_metadata.copy_(other.core_metadata)
        self.indexer_metadata.copy_(other.indexer_metadata)


def maybe_copy_inplace(dst, *, src) -> None:
    assert type(src) == type(dst)
    if dst is not None:
        dst.copy_(src)
