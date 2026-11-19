from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.mem_cache.qsa_kv_pool import (
    QSARawKVSharding,
    assert_qsa_indices_in_bounds,
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
    if static_buffers is not None:
        static_buffers = static_buffers.for_batch(q.shape[0])
        if q.shape[1] != static_buffers.num_heads // static_buffers.world_size:
            raise ValueError("static Q transport head geometry mismatch")
    gathered_q = q
    if group is not None:
        with torch.profiler.record_function("qsa.owner.q_gather"):
            if static_buffers is None:
                gathered_q = group.all_gather(q.contiguous(), dim=1).contiguous()
            else:
                rows = q.shape[0]
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
                raise ValueError("selected owner width exceeds static buffer capacity")
            local_slots, _ = _owner_localize_token_slots_triton(
                global_token_slots,
                sharding,
                fixed_width=owner_width,
                compact=static_buffers.local_slots[
                    : global_token_slots.shape[0], :owner_width
                ],
                lengths=static_buffers.owner_lengths[: global_token_slots.shape[0]],
            )
        else:
            local_slots, _ = _owner_localize_token_slots_triton(
                global_token_slots, sharding, fixed_width=owner_width
            )
    if static_buffers is not None and group is not None:
        static_buffers.q_gathered[: q.shape[0]].copy_(
            static_buffers.q_recv[:, : q.shape[0]].permute(1, 0, 2)
        )
        gathered_q = static_buffers.q_gathered[: q.shape[0]]
    with torch.profiler.record_function("qsa.owner.kernel"):
        assert_qsa_indices_in_bounds(
            local_slots,
            local_k_cache.shape[0],
            valid_mask=local_slots >= 0,
            label="owner local KV read locations",
        )
        from sglang.srt.layers.attention.qsa.sparse_attn import (
            sparse_gqa_packed_decode_with_lse_triton,
        )

        scale = gathered_q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        output, lse = sparse_gqa_packed_decode_with_lse_triton(
            gathered_q.contiguous(),
            local_k_cache,
            local_v_cache,
            local_slots,
            scale,
        )
    if group is None:
        return output, lse
    with torch.profiler.record_function("qsa.owner.output_lse_merge"):
        output_dtype = output.dtype
        world_size = int(group.world_size)
        rows, num_heads, head_dim = output.shape
        if num_heads % world_size:
            raise ValueError("owner partial heads must be divisible by group size")
        local_heads = num_heads // world_size
        if static_buffers is None:
            packed = torch.empty(
                (world_size, rows, local_heads, head_dim + 1),
                dtype=torch.float32,
                device=output.device,
            )
            recv = torch.empty_like(packed)
        else:
            packed = static_buffers.owner_a2a_send
            recv = static_buffers.owner_a2a_recv
            packed[..., :-1].zero_()
            packed[..., -1].fill_(float("-inf"))
        packed[:, :rows, ..., :-1].copy_(
            output.float()
            .view(rows, world_size, local_heads, head_dim)
            .permute(1, 0, 2, 3)
        )
        packed[:, :rows, ..., -1].copy_(
            lse.view(rows, world_size, local_heads).permute(1, 0, 2)
        )
        group.all_to_all_single(recv.view(-1), packed.view(-1))
        gathered = recv[:, :rows]
        from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton

        output, lse = dcp_lse_combine_triton(
            gathered[..., :-1],
            gathered[..., -1],
            is_lse_base_on_e=True,
            return_lse=True,
        )
        output = output.to(output_dtype)
    return output, lse
