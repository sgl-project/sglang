from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    INDEX_HEAD_DIM,
    build_pooled_page_table_64,
    kpool_build_ragged_layout,
    kpool_max_closed_pools,
    update_kpool_write_plan_cuda_graph,
)
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.model_executor.forward_context import get_req_to_token_pool
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
    from sglang.srt.layers.attention.dsa_backend import DSAMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


_RAGGED_SCRATCH_K_U8: Optional[torch.Tensor] = None
_RAGGED_SCRATCH_K_SCALE: Optional[torch.Tensor] = None


def _get_ragged_scratch(
    total_k_rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    global _RAGGED_SCRATCH_K_U8, _RAGGED_SCRATCH_K_SCALE
    cur = _RAGGED_SCRATCH_K_U8
    grow = (
        cur is None
        or cur.device.type != device.type
        or (device.index is not None and cur.device.index != device.index)
        or cur.shape[0] < total_k_rows
    )
    if grow:
        _RAGGED_SCRATCH_K_U8 = torch.empty(
            (total_k_rows, INDEX_HEAD_DIM), dtype=torch.uint8, device=device
        )
        _RAGGED_SCRATCH_K_SCALE = torch.empty(
            (total_k_rows,), dtype=torch.float32, device=device
        )
    return _RAGGED_SCRATCH_K_U8[:total_k_rows], _RAGGED_SCRATCH_K_SCALE[:total_k_rows]


@dataclass(frozen=True)
class PoolWriteRows:
    req: torch.Tensor
    pool_id: torch.Tensor
    n_from_tail: torch.Tensor
    chunk_src: torch.Tensor
    tail_logical_base: torch.Tensor
    write_loc: torch.Tensor

    @property
    def is_empty(self) -> bool:
        return self.pool_id.shape[0] == 0


@dataclass(frozen=True)
class TailWriteRows:
    req: torch.Tensor
    dst_logical_start: torch.Tensor
    chunk_src: torch.Tensor
    n_write: torch.Tensor

    @property
    def is_empty(self) -> bool:
        return self.req.shape[0] == 0


@dataclass(frozen=True)
class KPoolCpInfo:
    size: int
    rank: int
    owner_rank: torch.Tensor
    local_write_mask: torch.Tensor


@dataclass(frozen=True)
class KPoolExtendPlan:
    writes: PoolWriteRows
    tails: TailWriteRows
    pooled_seq_lens_expanded: torch.Tensor
    seq_lens_expanded: torch.Tensor
    ragged_concat_page_table: torch.Tensor
    ragged_q_ks: torch.Tensor
    ragged_q_ke: torch.Tensor
    ragged_total_k_rows: int
    ragged_k_u8: Optional[torch.Tensor]
    ragged_k_scale: Optional[torch.Tensor]
    ragged_paged_page_table: Optional[torch.Tensor]
    ragged_paged_page_table_row_index: Optional[torch.Tensor]
    cp: Optional[KPoolCpInfo] = None


@dataclass(frozen=True)
class KPoolWritePlan:
    """``write_loc[b, p]`` is the compression destination for candidate closed
    pool ``base_pool[b] + p``; the kernel decides which candidates closed."""

    req: torch.Tensor
    write_start: torch.Tensor
    tail_logical_start: torch.Tensor
    write_loc: torch.Tensor  # int64 [B, max_closed_pools]
    num_draft_tokens: int
    pool_seqlens_per_q: Optional[torch.Tensor] = None
    seqlens_per_q: Optional[torch.Tensor] = None
    pool_schedule_metadata: Optional[torch.Tensor] = None
    effective_n_per_batch: Optional[torch.Tensor] = None


def _is_kpool_layout_enabled(kpool: int, physical_page_size: int) -> bool:
    return kpool > 1 and physical_page_size == 64 and physical_page_size % kpool == 0


@dataclass
class _KPoolCpuPlan:
    pool_batch_idx: List[int] = field(default_factory=list)
    pool_req: List[int] = field(default_factory=list)
    pool_pool_id: List[int] = field(default_factory=list)
    pool_n_from_tail: List[int] = field(default_factory=list)
    pool_chunk_src: List[int] = field(default_factory=list)
    pool_tail_logical_base: List[int] = field(default_factory=list)

    tail_req: List[int] = field(default_factory=list)
    tail_dst_logical_start: List[int] = field(default_factory=list)
    tail_chunk_src: List[int] = field(default_factory=list)
    tail_n_write: List[int] = field(default_factory=list)

    ragged_q_len: List[int] = field(default_factory=list)
    ragged_pool_pages: List[int] = field(default_factory=list)
    cu_pages_excl: List[int] = field(default_factory=list)
    cu_q_len_excl: List[int] = field(default_factory=list)
    total_pool_pages: int = 0


class _KPoolDecompose(NamedTuple):
    first_slot: int
    base_pool: int
    n_pool: int
    tail_n_write: int


def _decompose_compress(start: int, length: int, kpool: int) -> _KPoolDecompose:
    first_slot = start % kpool
    base_pool = start // kpool
    n_pool = (start + length) // kpool - base_pool
    consumed = max(0, n_pool * kpool - first_slot)
    tail_n = length - consumed
    return _KPoolDecompose(
        first_slot=first_slot,
        base_pool=base_pool,
        n_pool=n_pool,
        tail_n_write=tail_n,
    )


def _append_compress_rows(
    plan: _KPoolCpuPlan,
    kpool: int,
    batch_size: int,
    extend_seq_lens_cpu: List[int],
    seq_lens_cpu: List[int],
    req_pool_indices_cpu: List[int],
) -> None:
    q_offset = 0
    for i in range(batch_size):
        q_len = extend_seq_lens_cpu[i]
        assert q_len > 0, f"extend_seq_lens_cpu[{i}] = {q_len}; expected > 0"

        seq_len = seq_lens_cpu[i]
        req = req_pool_indices_cpu[i]
        d = _decompose_compress(seq_len - q_len, q_len, kpool)

        if d.n_pool > 0:
            plan.pool_batch_idx.extend([i] * d.n_pool)
            plan.pool_req.extend([req] * d.n_pool)
            plan.pool_pool_id.extend(range(d.base_pool, d.base_pool + d.n_pool))
            plan.pool_n_from_tail.append(d.first_slot)
            plan.pool_n_from_tail.extend([0] * (d.n_pool - 1))
            bulk_start = q_offset + kpool - d.first_slot
            plan.pool_chunk_src.append(q_offset)
            plan.pool_chunk_src.extend(
                range(bulk_start, bulk_start + (d.n_pool - 1) * kpool, kpool)
            )
            plan.pool_tail_logical_base.extend(
                range(
                    d.base_pool * kpool,
                    (d.base_pool + d.n_pool) * kpool,
                    kpool,
                )
            )

        if d.tail_n_write > 0:
            consumed = q_len - d.tail_n_write
            plan.tail_req.append(req)
            plan.tail_dst_logical_start.append(seq_len - q_len + consumed)
            plan.tail_chunk_src.append(q_offset + consumed)
            plan.tail_n_write.append(d.tail_n_write)

        q_offset += q_len


def _append_local_rows(
    plan: _KPoolCpuPlan,
    kpool: int,
    index_page_size: int,
    local_extend_seq_lens_cpu: List[int],
    local_seq_lens_cpu: List[int],
) -> None:
    q_offset = 0
    for q_len, seq_len in zip(
        local_extend_seq_lens_cpu, local_seq_lens_cpu, strict=True
    ):
        assert q_len > 0, f"local_extend_seq_lens_cpu has non-positive {q_len = }"
        plan.ragged_q_len.append(q_len)
        pool_seq_len = seq_len // kpool
        pool_pages_i = (pool_seq_len + index_page_size - 1) // index_page_size
        plan.ragged_pool_pages.append(pool_pages_i)
        plan.cu_pages_excl.append(plan.total_pool_pages)
        plan.cu_q_len_excl.append(q_offset)
        plan.total_pool_pages += pool_pages_i
        q_offset += q_len


def _kpool_cpu_plan(
    forward_batch: ForwardBatch,
    kpool: int,
    index_page_size: int,
    *,
    local_extend_seq_lens_cpu: Optional[List[int]] = None,
    local_seq_lens_cpu: Optional[List[int]] = None,
) -> _KPoolCpuPlan:
    plan = _KPoolCpuPlan()

    extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
    if isinstance(extend_seq_lens_cpu, torch.Tensor):
        extend_seq_lens_cpu = extend_seq_lens_cpu.tolist()
    seq_lens_cpu = forward_batch.seq_lens_cpu.tolist()
    req_pool_indices_cpu = getattr(forward_batch, "req_pool_indices_cpu", None)
    if req_pool_indices_cpu is None:
        req_pool_indices_cpu = forward_batch.req_pool_indices
    req_pool_indices_cpu = req_pool_indices_cpu.tolist()

    _append_compress_rows(
        plan,
        kpool,
        forward_batch.batch_size,
        extend_seq_lens_cpu,
        seq_lens_cpu,
        req_pool_indices_cpu,
    )

    if local_extend_seq_lens_cpu is None:
        local_extend_seq_lens_cpu = extend_seq_lens_cpu
        local_seq_lens_cpu = seq_lens_cpu

    _append_local_rows(
        plan,
        kpool,
        index_page_size,
        local_extend_seq_lens_cpu,
        local_seq_lens_cpu,
    )
    return plan


def _kpool_plan_to_gpu(
    cpu: _KPoolCpuPlan,
    forward_batch: ForwardBatch,
    full_real_page_table: torch.Tensor,
    local_real_page_table: torch.Tensor,
    local_seqlens_expanded: torch.Tensor,
    local_req_pool_indices: torch.Tensor,
    kpool: int,
    index_page_size: int,
    topk_transform_method: TopkTransformMethod,
) -> KPoolExtendPlan:
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod

    device = forward_batch.seq_lens.device
    n_pool = len(cpu.pool_pool_id)
    n_tail = len(cpu.tail_req)
    n_rag = len(cpu.ragged_q_len)

    total_pool_pages = cpu.total_pool_pages
    ragged_total_k_rows = total_pool_pages * index_page_size

    need_paged = (
        topk_transform_method == TopkTransformMethod.PAGED
        and envs.SGLANG_DSA_FUSE_TOPK.get()
        and n_rag > 0
    )

    i64_total = 4 * n_pool + 2 * n_tail
    if i64_total > 0:
        i64_cpu = torch.tensor(
            cpu.pool_req
            + cpu.pool_pool_id
            + cpu.pool_chunk_src
            + cpu.pool_batch_idx
            + cpu.tail_req
            + cpu.tail_chunk_src,
            dtype=torch.int64,
            pin_memory=True,
        )
        i64_gpu = i64_cpu.to(device, non_blocking=True)
        c = 0
        pool_req_t = i64_gpu[c : c + n_pool]
        c += n_pool
        pool_pool_id_t = i64_gpu[c : c + n_pool]
        c += n_pool
        pool_chunk_src_t = i64_gpu[c : c + n_pool]
        c += n_pool
        pool_batch_idx_t = i64_gpu[c : c + n_pool]
        c += n_pool
        tail_req_t = i64_gpu[c : c + n_tail]
        c += n_tail
        tail_chunk_src_t = i64_gpu[c : c + n_tail]
    else:
        empty_i64 = torch.empty((0,), dtype=torch.int64, device=device)
        pool_req_t = pool_pool_id_t = pool_chunk_src_t = pool_batch_idx_t = empty_i64
        tail_req_t = tail_chunk_src_t = empty_i64

    i32_total = 2 * n_pool + 2 * n_tail + 4 * n_rag
    if i32_total > 0:
        i32_cpu = torch.tensor(
            cpu.pool_n_from_tail
            + cpu.pool_tail_logical_base
            + cpu.tail_dst_logical_start
            + cpu.tail_n_write
            + cpu.ragged_pool_pages
            + cpu.ragged_q_len
            + cpu.cu_pages_excl
            + cpu.cu_q_len_excl,
            dtype=torch.int32,
            pin_memory=True,
        )
        i32_gpu = i32_cpu.to(device, non_blocking=True)
        c = 0
        pool_n_from_tail_t = i32_gpu[c : c + n_pool]
        c += n_pool
        pool_tail_logical_base_t = i32_gpu[c : c + n_pool]
        c += n_pool
        tail_dst_logical_start_t = i32_gpu[c : c + n_tail]
        c += n_tail
        tail_n_write_t = i32_gpu[c : c + n_tail]
        c += n_tail
        ragged_pool_pages_t = i32_gpu[c : c + n_rag]
        c += n_rag
        ragged_q_len_t = i32_gpu[c : c + n_rag]
        c += n_rag
        cu_pages_excl_t = i32_gpu[c : c + n_rag]
        c += n_rag
        cu_q_len_excl_t = i32_gpu[c : c + n_rag]
    else:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        pool_n_from_tail_t = pool_tail_logical_base_t = empty_i32
        tail_dst_logical_start_t = tail_n_write_t = empty_i32
        ragged_pool_pages_t = ragged_q_len_t = empty_i32
        cu_pages_excl_t = cu_q_len_excl_t = empty_i32

    if n_pool > 0:
        pool_page_group = torch.div(
            pool_pool_id_t, index_page_size, rounding_mode="floor"
        )
        token_page_row = pool_page_group * kpool
        packed_page = (
            full_real_page_table[pool_batch_idx_t, token_page_row].to(torch.int64)
            // kpool
        )
        pool_write_locs = packed_page * index_page_size + torch.remainder(
            pool_pool_id_t, index_page_size
        )
    else:
        pool_write_locs = torch.empty((0,), dtype=torch.int64, device=device)

    pooled_seq_lens_expanded = torch.div(
        local_seqlens_expanded, kpool, rounding_mode="floor"
    ).to(torch.int32)

    if n_rag > 0:
        (
            ragged_concat_page_table,
            ragged_q_ks,
            ragged_q_ke,
        ) = kpool_build_ragged_layout(
            full_page_table=local_real_page_table,
            cu_pages_excl=cu_pages_excl_t,
            ragged_pool_pages=ragged_pool_pages_t,
            cu_q_len_excl=cu_q_len_excl_t,
            ragged_q_len=ragged_q_len_t,
            pooled_seq_lens_expanded=pooled_seq_lens_expanded,
            index_page_size=index_page_size,
            total_pool_pages=total_pool_pages,
            total_q=pooled_seq_lens_expanded.shape[0],
            kpool=kpool,
        )
    else:
        empty_i32_dev = torch.empty((0,), dtype=torch.int32, device=device)
        ragged_concat_page_table = empty_i32_dev
        ragged_q_ks = empty_i32_dev
        ragged_q_ke = empty_i32_dev

    ragged_paged_page_table = None
    ragged_paged_page_table_row_index = None
    if need_paged:
        req_to_token = get_req_to_token_pool().req_to_token
        ragged_paged_page_table_row_index = torch.repeat_interleave(
            local_req_pool_indices.to(torch.int32),
            ragged_q_len_t,
            output_size=sum(cpu.ragged_q_len),
        )
        ragged_paged_page_table = req_to_token

    if ragged_total_k_rows > 0:
        ragged_k_u8, ragged_k_scale = _get_ragged_scratch(ragged_total_k_rows, device)
    else:
        ragged_k_u8 = None
        ragged_k_scale = None

    return KPoolExtendPlan(
        writes=PoolWriteRows(
            req=pool_req_t,
            pool_id=pool_pool_id_t,
            n_from_tail=pool_n_from_tail_t,
            chunk_src=pool_chunk_src_t,
            tail_logical_base=pool_tail_logical_base_t,
            write_loc=pool_write_locs,
        ),
        tails=TailWriteRows(
            req=tail_req_t,
            dst_logical_start=tail_dst_logical_start_t,
            chunk_src=tail_chunk_src_t,
            n_write=tail_n_write_t,
        ),
        pooled_seq_lens_expanded=pooled_seq_lens_expanded,
        seq_lens_expanded=local_seqlens_expanded,
        ragged_concat_page_table=ragged_concat_page_table,
        ragged_q_ks=ragged_q_ks,
        ragged_q_ke=ragged_q_ke,
        ragged_total_k_rows=ragged_total_k_rows,
        ragged_k_u8=ragged_k_u8,
        ragged_k_scale=ragged_k_scale,
        ragged_paged_page_table=ragged_paged_page_table,
        ragged_paged_page_table_row_index=ragged_paged_page_table_row_index,
        cp=_kpool_cp_owner_rank(forward_batch, n_pool, device),
    )


def _kpool_cp_owner_rank(
    forward_batch: ForwardBatch,
    n_pool: int,
    device: torch.device,
) -> Optional[KPoolCpInfo]:
    if not dsa_use_prefill_cp(forward_batch):
        return None

    cp_size = get_parallel().attn_cp_size
    if cp_size <= 1:
        return None

    cp_rank = get_parallel().attn_cp_rank
    if n_pool > 0:
        owner = torch.arange(n_pool, dtype=torch.int32, device=device) % cp_size
        local_write_mask = owner == cp_rank
    else:
        owner = torch.empty((0,), dtype=torch.int32, device=device)
        local_write_mask = torch.empty((0,), dtype=torch.bool, device=device)
    return KPoolCpInfo(
        size=cp_size,
        rank=cp_rank,
        owner_rank=owner,
        local_write_mask=local_write_mask,
    )


def init_kpool_extend_metadata(
    metadata: DSAMetadata,
    forward_batch: ForwardBatch,
    *,
    kpool: int,
    physical_page_size: int,
    index_page_size: int,
    topk_transform_method: TopkTransformMethod,
    full_real_page_table: torch.Tensor,
    full_seqlens_expanded: torch.Tensor,
    local_real_page_table: Optional[torch.Tensor] = None,
    local_seqlens_expanded: Optional[torch.Tensor] = None,
    local_extend_seq_lens_cpu: Optional[List[int]] = None,
    local_seq_lens_cpu: Optional[List[int]] = None,
    local_req_pool_indices: Optional[torch.Tensor] = None,
) -> DSAMetadata:
    mode = forward_batch.forward_mode
    is_extend_like = mode.is_extend_without_speculative() or mode.is_draft_extend_v2()
    if (
        not _is_kpool_layout_enabled(kpool, physical_page_size)
        or not is_extend_like
        or forward_batch.extend_seq_lens_cpu is None
        or forward_batch.seq_lens_cpu is None
    ):
        return metadata

    if local_real_page_table is None:
        local_real_page_table = full_real_page_table
    if local_seqlens_expanded is None:
        local_seqlens_expanded = full_seqlens_expanded
    if local_req_pool_indices is None:
        local_req_pool_indices = forward_batch.req_pool_indices

    cpu = _kpool_cpu_plan(
        forward_batch,
        kpool,
        index_page_size,
        local_extend_seq_lens_cpu=local_extend_seq_lens_cpu,
        local_seq_lens_cpu=local_seq_lens_cpu,
    )
    plan = _kpool_plan_to_gpu(
        cpu,
        forward_batch,
        full_real_page_table,
        local_real_page_table,
        local_seqlens_expanded,
        local_req_pool_indices,
        kpool,
        index_page_size,
        topk_transform_method,
    )
    return dataclasses.replace(metadata, kpool_extend_plan=plan)


_DEEP_GEMM_MODULE = None
_DEEP_GEMM_IMPORT_FAILED = False


def _get_deep_gemm():
    # Cache import failure too, because this helper runs on every graph-replay
    # metadata refresh.
    global _DEEP_GEMM_MODULE, _DEEP_GEMM_IMPORT_FAILED
    if _DEEP_GEMM_MODULE is None and not _DEEP_GEMM_IMPORT_FAILED:
        try:
            import deep_gemm
        except (ImportError, ModuleNotFoundError):
            _DEEP_GEMM_IMPORT_FAILED = True
        else:
            _DEEP_GEMM_MODULE = deep_gemm
    return _DEEP_GEMM_MODULE


def _compute_pool_schedule_metadata(
    pool_seqlens: torch.Tensor,
    *,
    index_page_size: int,
) -> Optional[torch.Tensor]:
    if not is_cuda():
        return None
    deep_gemm = _get_deep_gemm()
    if deep_gemm is None:
        return None
    return deep_gemm.get_paged_mqa_logits_metadata(
        pool_seqlens.contiguous().view(-1, 1).clamp(min=1),
        index_page_size,
        deep_gemm.get_num_sms(),
    )


def init_pooled_paged_mqa_metadata(
    metadata: DSAMetadata,
    seqlens_32: torch.Tensor,
    forward_mode: ForwardMode,
    *,
    kpool: int,
    physical_page_size: int,
    index_page_size: int,
    build_schedule_metadata: bool = True,
) -> DSAMetadata:
    if (
        not _is_kpool_layout_enabled(kpool, physical_page_size)
        or not is_cuda()
        or not (forward_mode.is_decode_or_idle() or forward_mode.is_target_verify())
    ):
        return metadata

    pooled_page_table = build_pooled_page_table_64(
        metadata.real_page_table, kpool
    ).contiguous()
    if forward_mode.is_target_verify():
        return dataclasses.replace(
            metadata,
            pooled_index_kpool=kpool,
            pooled_real_page_table=pooled_page_table,
        )

    pool_seqlens = torch.div(seqlens_32, kpool, rounding_mode="floor").to(torch.int32)
    schedule = (
        _compute_pool_schedule_metadata(
            pool_seqlens,
            index_page_size=index_page_size,
        )
        if build_schedule_metadata
        else None
    )
    return dataclasses.replace(
        metadata,
        pooled_index_kpool=kpool,
        pooled_cache_seqlens_int32=pool_seqlens,
        pooled_real_page_table=pooled_page_table,
        pooled_paged_mqa_schedule_metadata=schedule,
    )


def update_pooled_paged_mqa_metadata(
    metadata: DSAMetadata,
    seqlens_32: torch.Tensor,
    forward_mode: ForwardMode,
    *,
    kpool: int,
    physical_page_size: int,
    index_page_size: int,
    build_schedule_metadata: bool = True,
) -> None:
    if (
        not _is_kpool_layout_enabled(kpool, physical_page_size)
        or not is_cuda()
        or not (forward_mode.is_decode_or_idle() or forward_mode.is_target_verify())
    ):
        return

    if forward_mode.is_target_verify():
        metadata.pooled_real_page_table.copy_(metadata.real_page_table[:, ::kpool])
        return

    if (
        metadata.pooled_index_kpool != kpool
        or metadata.pooled_cache_seqlens_int32 is None
        or metadata.pooled_real_page_table is None
    ):
        return

    pool_seqlens = torch.div(seqlens_32, kpool, rounding_mode="floor").to(torch.int32)
    metadata.pooled_cache_seqlens_int32[: pool_seqlens.shape[0]].copy_(pool_seqlens)
    pooled_page_table = build_pooled_page_table_64(
        metadata.real_page_table, kpool
    ).contiguous()
    metadata.pooled_real_page_table[
        : pooled_page_table.shape[0], : pooled_page_table.shape[1]
    ].copy_(pooled_page_table)

    if (
        build_schedule_metadata
        and metadata.pooled_paged_mqa_schedule_metadata is not None
    ):
        new_schedule = _compute_pool_schedule_metadata(
            metadata.pooled_cache_seqlens_int32,
            index_page_size=index_page_size,
        )
        if new_schedule is not None:
            metadata.pooled_paged_mqa_schedule_metadata.copy_(new_schedule)


def _alloc_kpool_write_plan_buffers(
    *,
    max_bs: int,
    num_draft_tokens: int,
    kpool: int,
    device: torch.device,
    is_verify: bool,
    is_v2: bool = False,
) -> KPoolWritePlan:
    max_closed_pools = kpool_max_closed_pools(num_draft_tokens, kpool)
    verify_extras = {}
    if is_verify:
        n_rows = max_bs * num_draft_tokens
        verify_extras = dict(
            pool_seqlens_per_q=torch.zeros(n_rows, dtype=torch.int32, device=device),
            seqlens_per_q=torch.zeros(n_rows, dtype=torch.int32, device=device),
        )
    if is_v2:
        verify_extras["effective_n_per_batch"] = torch.zeros(
            max_bs, dtype=torch.int32, device=device
        )
    return KPoolWritePlan(
        req=torch.zeros(max_bs, dtype=torch.int64, device=device),
        write_start=torch.zeros(max_bs, dtype=torch.int32, device=device),
        tail_logical_start=torch.zeros(max_bs, dtype=torch.int32, device=device),
        write_loc=torch.zeros(
            max_bs, max_closed_pools, dtype=torch.int64, device=device
        ),
        num_draft_tokens=num_draft_tokens,
        **verify_extras,
    )


def init_kpool_write_plan_capture(
    metadata: DSAMetadata,
    *,
    max_bs: int,
    kpool: int,
    physical_page_size: int,
    num_draft_tokens: int,
    device: torch.device,
    is_verify: bool,
    index_page_size: int,
    is_v2: bool = False,
    build_schedule_metadata: bool = True,
) -> DSAMetadata:
    if not _is_kpool_layout_enabled(kpool, physical_page_size) or num_draft_tokens == 0:
        return metadata

    plan = _alloc_kpool_write_plan_buffers(
        max_bs=max_bs,
        num_draft_tokens=num_draft_tokens,
        kpool=kpool,
        device=device,
        is_verify=is_verify,
        is_v2=is_v2,
    )
    if is_verify and build_schedule_metadata:
        schedule = _compute_pool_schedule_metadata(
            plan.pool_seqlens_per_q,
            index_page_size=index_page_size,
        )
        plan = dataclasses.replace(plan, pool_schedule_metadata=schedule)
    return dataclasses.replace(metadata, kpool_write_plan=plan)


def update_kpool_write_plan(
    metadata: DSAMetadata,
    *,
    write_start: torch.Tensor,
    req_pool_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    kpool: int,
    physical_page_size: int,
    num_draft_tokens: int,
    forward_mode: ForwardMode,
    index_page_size: int,
    effective_n_per_batch: Optional[torch.Tensor] = None,
) -> None:
    if not _is_kpool_layout_enabled(kpool, physical_page_size) or not (
        is_cuda() or is_hip()
    ):
        return
    is_verify = forward_mode.is_target_verify()
    is_decode = forward_mode.is_decode_or_idle()
    is_v2 = forward_mode.is_draft_extend_v2()
    if not (is_verify or is_decode or is_v2):
        return

    plan = metadata.kpool_write_plan
    assert plan is not None, "kpool_write_plan must be allocated before update"
    update_kpool_write_plan_cuda_graph(
        write_start=write_start,
        req_pool_indices=req_pool_indices,
        real_page_table=real_page_table,
        req_out=plan.req,
        write_start_out=plan.write_start,
        tail_logical_start_out=plan.tail_logical_start,
        write_loc_out=plan.write_loc,
        pool_seqlens_per_q_out=plan.pool_seqlens_per_q,
        seqlens_per_q_out=plan.seqlens_per_q,
        kpool=kpool,
        num_draft_tokens=num_draft_tokens,
        index_page_size=index_page_size,
    )

    if (
        is_v2
        and effective_n_per_batch is not None
        and plan.effective_n_per_batch is not None
    ):
        plan.effective_n_per_batch[: effective_n_per_batch.shape[0]].copy_(
            effective_n_per_batch.to(torch.int32)
        )

    if plan.pool_schedule_metadata is not None:
        new_schedule = _compute_pool_schedule_metadata(
            plan.pool_seqlens_per_q,
            index_page_size=index_page_size,
        )
        if new_schedule is not None:
            plan.pool_schedule_metadata.copy_(new_schedule)


def init_kpool_write_plan(
    metadata: DSAMetadata,
    forward_batch: ForwardBatch,
    *,
    kpool: int,
    physical_page_size: int,
    real_page_table: torch.Tensor,
    num_draft_tokens: int,
    write_start: torch.Tensor,
    index_page_size: int,
    effective_n_per_batch: Optional[torch.Tensor] = None,
    build_schedule_metadata: bool = True,
) -> DSAMetadata:
    forward_mode = forward_batch.forward_mode
    is_verify = forward_mode.is_target_verify()
    is_decode = forward_mode.is_decode_or_idle()
    is_v2 = forward_mode.is_draft_extend_v2()
    is_ring_write = is_verify or is_decode or is_v2
    if not _is_kpool_layout_enabled(kpool, physical_page_size) or not is_ring_write:
        return metadata

    pool = getattr(forward_batch, "token_to_kv_pool", None)
    if (is_verify or is_v2) and pool is not None:
        assert pool.tail_extra_slots == num_draft_tokens, (
            f"tail_extra_slots mismatch: pool={pool.tail_extra_slots}, "
            f"forward={num_draft_tokens}"
        )

    metadata = init_kpool_write_plan_capture(
        metadata,
        max_bs=forward_batch.seq_lens.shape[0],
        kpool=kpool,
        physical_page_size=physical_page_size,
        num_draft_tokens=num_draft_tokens,
        device=forward_batch.seq_lens.device,
        is_verify=is_verify or is_v2,
        index_page_size=index_page_size,
        is_v2=is_v2,
        build_schedule_metadata=build_schedule_metadata,
    )
    update_kpool_write_plan(
        metadata,
        write_start=write_start,
        req_pool_indices=forward_batch.req_pool_indices,
        real_page_table=real_page_table,
        kpool=kpool,
        physical_page_size=physical_page_size,
        num_draft_tokens=num_draft_tokens,
        forward_mode=forward_mode,
        index_page_size=index_page_size,
        effective_n_per_batch=effective_n_per_batch,
    )
    return metadata
