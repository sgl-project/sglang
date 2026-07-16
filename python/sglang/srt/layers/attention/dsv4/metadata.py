from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip, is_xpu

if TYPE_CHECKING:
    pass

_TOPK_V2_MAX_SUPPORTED_LENGTH = 262144


def _maybe_copy_flashmla_sched_meta(dst, src) -> bool:
    if not (
        hasattr(dst, "have_initialized")
        and hasattr(src, "have_initialized")
        and hasattr(dst, "tile_scheduler_metadata")
        and hasattr(src, "tile_scheduler_metadata")
        and hasattr(dst, "num_splits")
        and hasattr(src, "num_splits")
    ):
        return False

    if dst is None or src is None:
        return False

    # CUDA graphs capture the initialized FlashMLA metadata tensors by pointer.
    # Replay preparation often builds a fresh, uninitialized metadata object;
    # replacing the captured object would drop the tensors still referenced by
    # the graph. Keep the captured object alive in that case.
    if getattr(dst, "have_initialized", False) and not getattr(
        src, "have_initialized", False
    ):
        return True

    if not getattr(dst, "have_initialized", False) or not getattr(
        src, "have_initialized", False
    ):
        return False

    for field_name in ("tile_scheduler_metadata", "num_splits"):
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        if src_val is None and dst_val is None:
            continue
        if src_val is None or dst_val is None or not hasattr(dst_val, "copy_"):
            return False
        if tuple(src_val.shape) != tuple(dst_val.shape):
            return False
        dst_val.copy_(src_val)

    dst.have_initialized = src.have_initialized
    dst.config = src.config
    return True


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
_LARGE_INDEXER_QUERY_THRESHOLD = 11673


def get_dcp_sharded_c4_seq_lens(
    c4_seq_lens: torch.Tensor,
    c4_page_size: int,
    dcp_world_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """Return lengths after interleaving logical C4 pages across DCP ranks."""

    assert c4_page_size > 0
    assert dcp_world_size > 0
    assert 0 <= dcp_rank < dcp_world_size

    shape = c4_seq_lens.shape
    seq_lens = c4_seq_lens.reshape(-1).to(torch.int64)
    full_pages = seq_lens // c4_page_size
    tail = seq_lens % c4_page_size

    # Count full logical pages p where p % world_size == rank.
    local_full_pages = torch.clamp(
        (full_pages + dcp_world_size - 1 - dcp_rank) // dcp_world_size,
        min=0,
    )
    owns_tail = (tail > 0) & ((full_pages % dcp_world_size) == dcp_rank)
    local_tail = torch.where(owns_tail, tail, torch.zeros_like(tail))
    return (local_full_pages * c4_page_size + local_tail).to(torch.int32).view(shape)


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
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        if not _maybe_copy_flashmla_sched_meta(dst_val, src_val):
            setattr(dst, field_name, src_val)

    provided_fields = check_eq_fields + copy_fields + assign_fields
    provided_fields_unique = set(provided_fields)
    assert len(provided_fields) == len(
        provided_fields_unique
    ), f"{provided_fields=} has dup"
    all_fields = {f.name for f in fields(src)}
    provided_fields = set(provided_fields)
    assert (
        provided_fields == all_fields
    ), f"{provided_fields - all_fields=}, {all_fields - provided_fields=}"


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


@dataclass
class PagedIndexerMetadata:
    page_size: int
    page_table: torch.Tensor
    c4_seq_lens: torch.Tensor
    use_prefill_cuda_graph: bool = False
    deep_gemm_metadata: Any = field(init=False, repr=False)
    dcp_world_size: int = field(init=False, repr=False, default=1)
    dcp_rank: int = field(init=False, repr=False, default=0)
    dcp_local_page_table: Optional[torch.Tensor] = field(
        init=False, repr=False, default=None
    )
    dcp_local_c4_seq_lens: Optional[torch.Tensor] = field(
        init=False, repr=False, default=None
    )
    dcp_deep_gemm_metadata: Any = field(init=False, repr=False, default=None)
    topk_metadata: torch.Tensor = field(init=False, repr=False)
    nonpaged_plan: Optional[NonPagedIndexerPlan] = field(
        init=False, repr=False, default=None
    )

    def _make_deep_gemm_metadata(self, c4_seq_lens: torch.Tensor):
        if (
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
            or is_xpu()
            or envs.SGLANG_OPT_USE_AITER_INDEXER.get()
        ):
            return None

        import deep_gemm

        use_jit_indexer = (
            envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.get()
            or c4_seq_lens.numel() > _LARGE_INDEXER_QUERY_THRESHOLD
        )
        if use_jit_indexer:
            from sglang.jit_kernel.dsv4 import get_paged_mqa_logits_metadata
        else:
            from deep_gemm import get_paged_mqa_logits_metadata

        _c4 = c4_seq_lens.to(torch.int32)
        if _c4.dim() == 1:
            _c4 = _c4.unsqueeze(-1)
        metadata = get_paged_mqa_logits_metadata(
            _c4,
            self.c4_page_size,
            deep_gemm.get_num_sms(),
        )
        assert isinstance(metadata, torch.Tensor)
        return metadata

    def __post_init__(self):
        self.deep_gemm_metadata = self._make_deep_gemm_metadata(self.c4_seq_lens)

        if envs.SGLANG_DSV4_DCP_SHARD_C4_INDEXER.get() and not is_hip():
            from sglang.srt.distributed.parallel_state import get_dcp_group_no_assert

            dcp_group = get_dcp_group_no_assert()
            if dcp_group is not None and dcp_group.world_size > 1:
                self.dcp_world_size = dcp_group.world_size
                self.dcp_rank = dcp_group.rank_in_group
                self.dcp_local_page_table = self.page_table[
                    :, self.dcp_rank :: self.dcp_world_size
                ].contiguous()
                self.dcp_local_c4_seq_lens = get_dcp_sharded_c4_seq_lens(
                    self.c4_seq_lens,
                    self.c4_page_size,
                    self.dcp_world_size,
                    self.dcp_rank,
                )
                self.dcp_deep_gemm_metadata = self._make_deep_gemm_metadata(
                    self.dcp_local_c4_seq_lens
                )

        from sglang.jit_kernel.dsv4 import plan_topk_v2

        if envs.SGLANG_OPT_USE_TOPK_V2.get():
            self.topk_metadata = plan_topk_v2(self.c4_seq_lens)
        else:
            self.topk_metadata = torch.empty((0,))

        assert self.page_size == 256, "the system hardcodes page_size=256"

    @property
    def c4_page_size(self) -> int:
        return self.page_size // 4

    @property
    def max_seq_len(self) -> int:
        return self.page_table.shape[1] * self.page_size

    @property
    def max_c4_seq_len(self) -> int:
        if self.c4_seq_lens.numel() == 0:
            return self.c4_page_size

        # During CUDA graph capture, ``.item()`` forces a device->host sync,
        # which is illegal on a capturing stream. The captured logits buffer
        # must also be sized for the worst case so replay shapes stay valid.
        # Fall back to the static max-capacity bound in that case.
        if torch.cuda.is_current_stream_capturing():
            return self.page_table.shape[1] * self.c4_page_size

        # CUDA graph/raw metadata may keep a max-capacity page table. The indexer
        # should only materialize logits up to the active compressed length.
        max_c4_seq_len = int(self.c4_seq_lens.max().item())
        max_c4_pages = max(
            1, (max_c4_seq_len + self.c4_page_size - 1) // self.c4_page_size
        )
        return min(max_c4_pages, self.page_table.shape[1]) * self.c4_page_size

    def copy_(self, other: PagedIndexerMetadata):
        if is_hip():
            copy_fields = ["page_table", "c4_seq_lens"]
            assign_fields = ["deep_gemm_metadata", "nonpaged_plan"]
        else:
            copy_fields = ["page_table", "c4_seq_lens", "deep_gemm_metadata"]
            assign_fields = ["nonpaged_plan"]
        copy_fields += [
            "dcp_local_page_table",
            "dcp_local_c4_seq_lens",
            "dcp_deep_gemm_metadata",
            "topk_metadata",
        ]
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[
                "page_size",
                "use_prefill_cuda_graph",
                "dcp_world_size",
                "dcp_rank",
            ],
            copy_fields=copy_fields,
            assign_fields=assign_fields,
        )
        self.nonpaged_plan = None


def maybe_copy_inplace(dst, *, src) -> None:
    assert type(src) == type(dst)
    if dst is not None:
        dst.copy_(src)
