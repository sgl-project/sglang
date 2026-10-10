"""Owner-side sparse attention without materializing global K/V."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.mem_cache.qsa_kv_pool import (
    QSARawKVSharding,
    assert_qsa_indices_in_bounds,
)

OWNER_PARTIAL_TRANSPORT = os.environ.get(
    "SGLANG_QSA_OWNER_PARTIAL_TRANSPORT", "packed_a2a"
)


@triton.jit
def _owner_localize_token_slots_kernel(
    global_slots_ptr,
    compact_ptr,
    lengths_ptr,
    selected_width: tl.constexpr,
    compact_width: tl.constexpr,
    global_size: tl.constexpr,
    world_size: tl.constexpr,
    owner_rank: tl.constexpr,
    block_width: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_width)
    input_mask = offsets < selected_width
    global_ids = tl.load(
        global_slots_ptr + row * selected_width + offsets,
        mask=input_mask,
        other=-1,
    ).to(tl.int64)
    owned = (
        input_mask
        & (global_ids >= 0)
        & (global_ids < global_size)
        & (global_ids % world_size == owner_rank)
    )
    destinations = tl.cumsum(owned.to(tl.int32), axis=0) - 1
    output_mask = offsets < compact_width
    tl.store(compact_ptr + row * compact_width + offsets, -1, mask=output_mask)
    tl.store(
        compact_ptr + row * compact_width + destinations,
        global_ids // world_size,
        mask=owned & (destinations < compact_width),
    )
    tl.store(lengths_ptr + row, tl.sum(owned.to(tl.int32), axis=0))


def _owner_localize_token_slots_triton(
    global_token_slots: torch.Tensor,
    sharding: QSARawKVSharding,
    *,
    fixed_width: int,
    compact: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, selected_width = global_token_slots.shape
    if compact is None:
        compact = torch.empty(
            (rows, fixed_width),
            dtype=torch.int64,
            device=global_token_slots.device,
        )
    if lengths is None:
        lengths = torch.empty(
            (rows,), dtype=torch.int32, device=global_token_slots.device
        )
    block_width = triton.next_power_of_2(max(selected_width, fixed_width))
    _owner_localize_token_slots_kernel[(rows,)](
        global_token_slots,
        compact,
        lengths,
        selected_width=selected_width,
        compact_width=fixed_width,
        global_size=sharding.global_capacity,
        world_size=sharding.world_size,
        owner_rank=sharding.rank,
        block_width=block_width,
        num_warps=4 if block_width <= 128 else 8,
    )
    if fixed_width < selected_width:
        torch._assert_async(
            torch.all(lengths <= fixed_width),
            "owner token count exceeds interleaved fixed-width capacity",
        )
    return compact, lengths


def select_kv_heads_for_query_shard(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    total_query_heads: int,
    total_kv_heads: int,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the global KV-head group covered by one contiguous Q-head shard."""

    if k.ndim != 3 or v.shape != k.shape:
        raise ValueError("K/V must have matching [tokens, heads, dim] shapes")
    if (
        min(total_query_heads, total_kv_heads, tp_size) <= 0
        or total_query_heads % total_kv_heads
        or total_query_heads % tp_size
        or not 0 <= tp_rank < tp_size
    ):
        raise ValueError("invalid query/KV tensor-parallel head geometry")
    query_heads_per_rank = total_query_heads // tp_size
    queries_per_kv = total_query_heads // total_kv_heads
    query_start = tp_rank * query_heads_per_rank
    query_stop = query_start + query_heads_per_rank
    first_kv = query_start // queries_per_kv
    last_kv = (query_stop - 1) // queries_per_kv
    if first_kv != last_kv:
        raise ValueError(
            "query shard crosses KV-head groups; QSA requires group-aligned TP"
        )
    if k.shape[1] == 1:
        return k, v
    if k.shape[1] != total_kv_heads:
        raise ValueError(
            "K/V head layout must be either owner-local or globally replicated: "
            f"got {k.shape[1]}, expected 1 or {total_kv_heads}"
        )
    return (
        k[:, first_kv : first_kv + 1].contiguous(),
        v[:, first_kv : first_kv + 1].contiguous(),
    )


@triton.jit
def _pack_owner_kv_contiguous_kernel(
    k_ptr,
    v_ptr,
    slots_ptr,
    lengths_ptr,
    out_k_ptr,
    out_v_ptr,
    out_slots_ptr,
    width: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1)
    head = tl.program_id(2)
    dimensions = tl.arange(0, block_dim)
    slot = tl.load(slots_ptr + row * width + column)
    length = tl.load(lengths_ptr + row)
    valid = (column < length) & (slot >= 0)
    source = slot.to(tl.int64) * num_kv_heads * head_dim + head * head_dim + dimensions
    packed_index = row * width + column
    destination = (
        packed_index.to(tl.int64) * num_kv_heads * head_dim
        + head * head_dim
        + dimensions
    )
    dimension_mask = dimensions < head_dim
    tl.store(
        out_k_ptr + destination,
        tl.load(k_ptr + source, mask=valid & dimension_mask, other=0.0),
        mask=dimension_mask,
    )
    tl.store(
        out_v_ptr + destination,
        tl.load(v_ptr + source, mask=valid & dimension_mask, other=0.0),
        mask=dimension_mask,
    )
    if head == 0:
        tl.store(
            out_slots_ptr + packed_index,
            tl.where(valid, packed_index, -1),
        )


def pack_owner_kv_contiguous(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    local_token_slots: torch.Tensor,
    owner_lengths: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
    out_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather owner-selected K/V into fixed-address row-contiguous storage."""

    rows, width = local_token_slots.shape
    expected = (rows * width, k_cache.shape[1], k_cache.shape[2])
    if out_k.shape != expected or out_v.shape != expected:
        raise ValueError(f"owner packed K/V shape must be {expected}")
    if out_slots.shape != local_token_slots.shape:
        raise ValueError("owner packed slots must match local slot shape")
    _pack_owner_kv_contiguous_kernel[(rows, width, k_cache.shape[1])](
        k_cache,
        v_cache,
        local_token_slots,
        owner_lengths,
        out_k,
        out_v,
        out_slots,
        width=width,
        num_kv_heads=k_cache.shape[1],
        head_dim=k_cache.shape[2],
        block_dim=triton.next_power_of_2(k_cache.shape[2]),
        num_warps=4,
    )
    return out_k, out_v, out_slots


def _validate_attention_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    local_token_slots: torch.Tensor,
) -> None:
    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("q, k_cache and v_cache must be rank-3 tensors")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have matching shapes")
    if local_token_slots.ndim != 2 or local_token_slots.shape[0] != q.shape[0]:
        raise ValueError("local_token_slots must have shape [query_tokens, tokens]")
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match")
    if k_cache.shape[1] == 0 or q.shape[1] % k_cache.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("q, k_cache and v_cache must share a device")
    if local_token_slots.device != q.device:
        raise ValueError("local_token_slots must be on the Q/K/V device")
    assert_qsa_indices_in_bounds(
        local_token_slots,
        k_cache.shape[0],
        valid_mask=local_token_slots >= 0,
        label="owner local KV read locations",
    )


def owner_sparse_attention_partial(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    local_token_slots: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the production GPU owner-local kernel without gathering K/V."""

    _validate_attention_inputs(q, k_cache, v_cache, local_token_slots)
    if not q.is_cuda:
        raise ValueError("production owner sparse attention requires CUDA tensors")
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        sparse_gqa_packed_decode_with_lse_triton,
    )

    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    return sparse_gqa_packed_decode_with_lse_triton(
        q.contiguous(),
        k_cache,
        v_cache,
        local_token_slots,
        scale,
    )


def merge_owner_sparse_attention(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    *,
    group,
    static_buffers=None,
    transport: str = "packed_a2a",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange FP32 output/LSE once and stably merge this rank's head shard."""

    if partial_output.ndim != 3 or partial_lse.shape != partial_output.shape[:2]:
        raise ValueError("partial output/LSE must be [rows, heads, dim]/[rows, heads]")
    if partial_lse.dtype != torch.float32:
        raise TypeError("partial LSE must be float32")
    world_size = int(group.world_size)
    rows, num_heads, head_dim = partial_output.shape
    if transport not in ("packed_a2a", "all_gather"):
        raise ValueError(f"unsupported owner partial transport: {transport}")
    if transport == "packed_a2a":
        if num_heads % world_size:
            raise ValueError("owner partial heads must be divisible by group size")
        local_heads = num_heads // world_size
        if static_buffers is None:
            packed = torch.empty(
                (world_size, rows, local_heads, head_dim + 1),
                dtype=torch.float32,
                device=partial_output.device,
            )
            recv = torch.empty_like(packed)
        else:
            static_buffers = static_buffers.for_graph(rows)
            static_buffers.for_batch(rows)
            packed = static_buffers.owner_a2a_send
            recv = static_buffers.owner_a2a_recv
            packed[..., :-1].zero_()
            packed[..., -1].fill_(float("-inf"))
        packed[:, :rows, ..., :-1].copy_(
            partial_output.float()
            .view(rows, world_size, local_heads, head_dim)
            .permute(1, 0, 2, 3)
        )
        packed[:, :rows, ..., -1].copy_(
            partial_lse.view(rows, world_size, local_heads).permute(1, 0, 2)
        )
        group.all_to_all_single(recv.view(-1), packed.view(-1))
        gathered = recv[:, :rows]
    elif static_buffers is None:
        packed = torch.cat(
            (partial_output.float(), partial_lse[..., None]), dim=-1
        ).unsqueeze(0)
        gathered = group.all_gather(packed, dim=0)
    else:
        static_buffers = static_buffers.for_graph(rows)
        static_buffers.for_batch(rows)
        packed = static_buffers.owner_transport
        packed[..., :-1].zero_()
        packed[..., -1].fill_(float("-inf"))
        packed[:rows, ..., :-1].copy_(partial_output.float())
        packed[:rows, ..., -1].copy_(partial_lse)
        recv = static_buffers.owner_recv.view(
            world_size * static_buffers.max_rows,
            static_buffers.num_heads,
            static_buffers.head_dim + 1,
        )
        group.all_gather_into_tensor(recv, packed)
        gathered = static_buffers.owner_recv[:, :rows]
    if gathered.shape[0] != world_size:
        raise ValueError("owner collective returned an unexpected owner dimension")
    if not gathered.is_cuda:
        raise ValueError("production owner sparse attention requires CUDA tensors")
    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton

    output, lse = dcp_lse_combine_triton(
        gathered[..., :-1],
        gathered[..., -1],
        is_lse_base_on_e=True,
        return_lse=True,
    )
    assert lse is not None
    return output.to(partial_output.dtype), lse


def owner_sparse_attention(
    q: torch.Tensor,
    local_k_cache: torch.Tensor,
    local_v_cache: torch.Tensor,
    global_token_slots: torch.Tensor,
    sharding: QSARawKVSharding,
    *,
    group=None,
    softmax_scale: float | None = None,
    static_buffers=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run owner filtering, local attention, and optional cross-owner merge."""

    local_query_heads = q.shape[1]
    if static_buffers is not None:
        static_buffers = static_buffers.for_graph(q.shape[0])
        expected_local_heads = static_buffers.num_heads // static_buffers.world_size
        if local_query_heads != expected_local_heads:
            raise ValueError(
                "static Q transport head geometry does not match the query: "
                f"got {local_query_heads}, expected {expected_local_heads}"
            )
    gathered_q = q
    if group is not None:
        with torch.profiler.record_function("qsa.owner.q_gather"):
            if static_buffers is None:
                gathered_q = group.all_gather(q.contiguous(), dim=1).contiguous()
            else:
                rows = q.shape[0]
                static_buffers.for_batch(rows)
                static_buffers.q_send.zero_()
                static_buffers.q_send[:, :rows].copy_(q.permute(1, 0, 2))
                device_group = getattr(group, "device_group", None)
                if device_group is None:
                    group.all_gather_into_tensor(
                        static_buffers.q_recv, static_buffers.q_send
                    )
                else:
                    work = all_gather_single(
                        static_buffers.q_recv,
                        static_buffers.q_send,
                        group=device_group,
                        async_op=True,
                    )
                    work.block_current_stream()
    with torch.profiler.record_function("qsa.owner.localize_slots"):
        owner_width = global_token_slots.shape[1]
        if static_buffers is not None and global_token_slots.is_cuda:
            if owner_width > static_buffers.owner_topk:
                raise ValueError(
                    "selected owner width exceeds static buffer capacity: "
                    f"{owner_width} > {static_buffers.owner_topk}"
                )
            local_slots, owner_lengths = _owner_localize_token_slots_triton(
                global_token_slots,
                sharding,
                fixed_width=owner_width,
                compact=static_buffers.local_slots[
                    : global_token_slots.shape[0], :owner_width
                ],
                lengths=static_buffers.owner_lengths[: global_token_slots.shape[0]],
            )
        else:
            local_slots, owner_lengths = _owner_localize_token_slots_triton(
                global_token_slots, sharding, fixed_width=owner_width
            )
    if static_buffers is not None and group is not None:
        static_buffers.q_gathered[: q.shape[0]].copy_(
            static_buffers.q_recv[:, : q.shape[0]].permute(1, 0, 2)
        )
        gathered_q = static_buffers.q_gathered[: q.shape[0]]
    with torch.profiler.record_function("qsa.owner.kernel"):
        attention_k = local_k_cache
        attention_v = local_v_cache
        attention_slots = local_slots
        if (
            static_buffers is not None
            and q.shape[0] <= static_buffers.owner_pack_max_rows
            and local_k_cache.shape[1] == 1
            and local_k_cache.shape[2] == 128
        ):
            static_buffers.for_batch(q.shape[0])
            packed_tokens = q.shape[0] * owner_width
            attention_k, attention_v, attention_slots = pack_owner_kv_contiguous(
                local_k_cache,
                local_v_cache,
                local_slots,
                owner_lengths,
                static_buffers.owner_packed_k[:packed_tokens],
                static_buffers.owner_packed_v[:packed_tokens],
                static_buffers.owner_packed_slots[: q.shape[0], :owner_width],
            )
        output, lse = owner_sparse_attention_partial(
            gathered_q,
            attention_k,
            attention_v,
            attention_slots,
            softmax_scale,
        )
    if group is None:
        return output, lse
    with torch.profiler.record_function("qsa.owner.output_lse_merge"):
        output, lse = merge_owner_sparse_attention(
            output,
            lse,
            group=group,
            static_buffers=static_buffers,
            transport=OWNER_PARTIAL_TRANSPORT,
        )
    if OWNER_PARTIAL_TRANSPORT == "all_gather":
        rank = int(group.rank_in_group)
        head_start = rank * local_query_heads
        head_stop = head_start + local_query_heads
        output = output[:, head_start:head_stop]
        lse = lse[:, head_start:head_stop]
    return output, lse


__all__ = [
    "OWNER_PARTIAL_TRANSPORT",
    "merge_owner_sparse_attention",
    "owner_sparse_attention",
    "owner_sparse_attention_partial",
    "pack_owner_kv_contiguous",
    "select_kv_heads_for_query_shard",
]
