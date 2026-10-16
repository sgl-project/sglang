from __future__ import annotations

import functools
from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_GATHER_BLOCK_SIZE = 64


class HiSparseSpecState(NamedTuple):
    """Persistent cache state and reusable miss workspace for speculative swap.

    ``cache_index`` stores the two int64 hash banks as
    ``[num_requests, 2, hash_size]``. ``cache_policy`` uses a control-plane row
    for the packed CLOCK states followed by one reference-epoch row per
    request: ``[1 + num_requests, hot_buffer_size]``.

    ``scratch_locs`` and ``scratch_state`` hold reusable miss locations,
    counters, and metadata shared by all layers.
    """

    cache_index: torch.Tensor
    cache_policy: torch.Tensor
    scratch_locs: torch.Tensor
    scratch_state: torch.Tensor


@cache_once
def _jit_spec_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    num_steps: int,
    record_miss_plan: bool,
    is_dsv4_layout: bool = False,
) -> Module:
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        item_size_bytes,
        num_steps,
        record_miss_plan,
        is_arch_support_pdl(),
        is_dsv4_layout,
    )
    return load_jit(
        "hisparse_spec",
        *template_args,
        cuda_files=["kvcacheio/hisparse_spec.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer_spec",
                f"load_cache_to_device_buffer_spec<{template_args}>",
            )
        ],
    )


def load_cache_to_device_buffer_spec_mla(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    state: HiSparseSpecState,
    num_real_reqs: torch.Tensor,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    is_dsv4_layout: bool = False,
) -> None:
    """Resolve all speculative steps and swap unique misses in one launch pair.

    Optional miss-plan outputs use the same protocol as the single-step HiSparse
    kernel, so shared-index layers can replay only the Host-to-GPU copies with
    ``copy_cache_planned_mla``.
    """
    _, num_steps, num_top_k = top_k_tokens.shape
    if is_dsv4_layout:
        if num_steps < 2 or num_top_k <= 0 or num_steps * num_top_k > 8192:
            raise ValueError(
                "HiSparse speculative swap requires at least two steps, positive "
                f"top_k, and steps * top_k <= 8192, got {num_steps} * {num_top_k}."
            )
    elif not 2 <= num_steps <= 4:
        raise ValueError(
            f"HiSparse speculative swap requires 2-4 steps, got {num_steps}."
        )
    hot_buffer_size = state.cache_policy.size(1)
    page_size = device_buffer_tokens.size(1) - hot_buffer_size
    # C4 rows are split into value and scale runs inside a padded page.
    # The paged copy computes both addresses and ignores the linear stride.
    item_size_bytes = (
        584 if is_dsv4_layout else host_cache.stride(0) * host_cache.element_size()
    )
    if is_dsv4_layout and page_size != 64:
        raise ValueError("HiSparse C4 speculative transfers require 64-slot V4 pages.")
    record_miss_plan = miss_src is not None
    if record_miss_plan:
        if miss_dst is None or miss_count is None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        if miss_src.dtype != torch.int64 or miss_dst.dtype != torch.int32:
            raise ValueError("miss_src must be int64 and miss_dst must be int32.")
        if miss_count.dtype != torch.int32:
            raise ValueError("miss_count must be int32.")
        plan_capacity = num_steps * num_top_k
        batch_size = top_k_tokens.size(0)
        if (
            miss_src.ndim != 2
            or miss_dst.ndim != 2
            or miss_src.size(0) < batch_size
            or miss_dst.size(0) < batch_size
            or miss_src.size(1) < plan_capacity
            or miss_dst.size(1) < plan_capacity
        ):
            raise ValueError(
                "speculative miss_src/miss_dst must have shape "
                f"[batch, >= steps * top_k] (capacity {plan_capacity})."
            )
        if miss_count.ndim != 1 or miss_count.numel() < batch_size:
            raise ValueError("speculative miss_count must have shape [batch].")
        if miss_src.stride(0) != miss_dst.stride(0):
            raise ValueError("miss_src/miss_dst row strides must match.")
    else:
        if miss_dst is not None or miss_count is not None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        empty = torch.empty(0)
        miss_src = miss_dst = miss_count = empty

    module = _jit_spec_module(
        item_size_bytes,
        _GATHER_BLOCK_SIZE,
        num_top_k,
        hot_buffer_size,
        num_steps,
        record_miss_plan,
        is_dsv4_layout,
    )

    module.load_cache_to_device_buffer_spec(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        state.cache_index,
        state.cache_policy,
        state.scratch_locs,
        state.scratch_state,
        num_real_reqs,
        page_size,
        miss_src,
        miss_dst,
        miss_count,
    )


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
    is_dsv4_layout: bool = False,
    top_k_block_size: int = 1,
    top_k_is_blocks: bool = False,
    record_miss_plan: bool = False,
    skip_io: bool = False,
) -> Module:
    # record_miss_plan / skip_io are compile-time kernel flags; the
    # (False, False) production instantiation stays byte-identical.
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        top_k_block_size,
        top_k_is_blocks,
        record_miss_plan,
        skip_io,
    )
    cache_args = make_cpp_args(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        top_k_block_size,
        top_k_is_blocks,
        record_miss_plan,
        skip_io,
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["kvcacheio/hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_copy_planned_module(
    block_size: int,
    is_mla: bool,
    is_dsv4_layout: bool,
    skip_io: bool,
) -> Module:
    template_args = make_cpp_args(block_size, is_mla, is_dsv4_layout, skip_io)
    return load_jit(
        "sparse_copy_planned",
        block_size,
        is_mla,
        is_dsv4_layout,
        skip_io,
        cuda_files=["kvcacheio/hisparse.cuh"],
        cuda_wrappers=[
            (
                "copy_cache_planned",
                f"copy_cache_planned<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_dsv4_transfer_module(block_size: int) -> Module:
    template_args = make_cpp_args(block_size)
    return load_jit(
        "sparse_cache_dsv4_transfer",
        block_size,
        cuda_files=["kvcacheio/hisparse.cuh"],
        cuda_wrappers=[
            (
                "transfer_cache_dsv4_mla",
                f"transfer_cache_dsv4_mla<{template_args}>",
            )
        ],
    )


def transfer_cache_dsv4_mla(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    block_size: int = 1024,
) -> None:
    """Copy C4 tokens with shared [item] or per-layer [layer, item] indices.

    Negative indices skip a copy, including its value and scale writes.
    """
    module = _jit_dsv4_transfer_module(block_size)
    module.transfer_cache_dsv4_mla(
        src_ptrs,
        dst_ptrs,
        src_indices,
        dst_indices,
    )


def _load_cache_to_device_buffer_mla(
    *,
    is_dsv4_layout: bool,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int,
    block_size: int,
    num_real_reqs: torch.Tensor | None,
    miss_src: torch.Tensor | None,
    miss_dst: torch.Tensor | None,
    miss_count: torch.Tensor | None,
    skip_io: bool,
) -> None:
    assert hot_buffer_size >= num_top_k, (
        f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"
    )

    record_miss_plan = miss_src is not None
    module = _jit_sparse_module(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla=True,
        is_dsv4_layout=is_dsv4_layout,
        record_miss_plan=record_miss_plan,
        skip_io=skip_io,
    )

    empty = torch.empty(0, device=top_k_tokens.device)

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )

    if record_miss_plan:
        assert miss_dst is not None and miss_count is not None
        assert miss_src.dtype == torch.int64 and miss_dst.dtype == torch.int32
        assert miss_count.dtype == torch.int32
        # The kernel indexes both plan rows with one stride.
        assert miss_src.stride(0) == miss_dst.stride(0)
    else:
        # Unused sentinels; the RecordMissPlan=false instantiation never reads them.
        miss_src = miss_dst = miss_count = empty

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
        miss_src,
        miss_dst,
        miss_count,
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    skip_io: bool = False,
) -> None:
    """Generic MLA hisparse swap-in: device + host both linear (stride=item_size_bytes).

    Optional miss_src/miss_dst/miss_count record the miss plan for replay by
    copy_cache_planned_mla; skip_io elides only the KV bytes (timing probe).
    """
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=False,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        skip_io=skip_io,
    )


def load_blocks_to_device_buffer_mha(
    top_k_blocks: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache_k: torch.Tensor,
    host_cache_v: torch.Tensor,
    device_buffer_k: torch.Tensor,
    device_buffer_v: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    hot_buffer_size: int,
    sparse_block_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    skip_io: bool = False,
) -> None:
    """Swap block-selected MHA K/V into the HiSparse device pool."""
    num_top_k_blocks = top_k_blocks.size(1)
    num_top_k_tokens = num_top_k_blocks * sparse_block_size
    assert hot_buffer_size >= num_top_k_tokens, (
        f"hot_buffer_size ({hot_buffer_size}) must be >= selected tokens "
        f"({num_top_k_tokens})"
    )
    assert top_k_device_locs.size(1) >= num_top_k_tokens
    k_stride = host_cache_k.stride(0) * host_cache_k.element_size()
    v_stride = host_cache_v.stride(0) * host_cache_v.element_size()
    assert k_stride == v_stride == item_size_bytes, (
        "K/V token strides must equal item_size_bytes: "
        f"k_stride={k_stride}, v_stride={v_stride}, "
        f"item_size_bytes={item_size_bytes}"
    )

    module = _jit_sparse_module(
        item_size_bytes,
        block_size,
        num_top_k_blocks,
        hot_buffer_size,
        is_mla=False,
        is_dsv4_layout=False,
        top_k_block_size=sparse_block_size,
        top_k_is_blocks=True,
        record_miss_plan=False,
        skip_io=skip_io,
    )
    empty = torch.empty(0, device=top_k_blocks.device)
    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_blocks.size(0)], dtype=torch.int32, device=top_k_blocks.device
        )

    module.load_cache_to_device_buffer(
        top_k_blocks,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache_k,
        host_cache_v,
        device_buffer_k,
        device_buffer_v,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
        empty,
        empty,
        empty,
    )


def copy_cache_planned_mla(
    *,
    miss_src: torch.Tensor,
    miss_dst: torch.Tensor,
    miss_count: torch.Tensor,
    num_real_reqs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    item_size_bytes: int,
    num_blocks: int = 4,
    block_size: int = 1024,
    is_dsv4_layout: bool = False,
    skip_io: bool = False,
) -> None:
    """Replay a recorded miss plan (host_cache -> device_buffer) for a skip layer.

    IO-only, no planning; the small fixed grid keeps the SM footprint low while
    overlapped on a side stream. The anchor's slot table stays valid (lockstep).
    """
    assert miss_src.dtype == torch.int64 and miss_dst.dtype == torch.int32
    assert miss_count.dtype == torch.int32
    module = _jit_copy_planned_module(block_size, True, is_dsv4_layout, skip_io)
    empty = torch.empty(0)
    module.copy_cache_planned(
        miss_src,
        miss_dst,
        miss_count,
        num_real_reqs,
        host_cache,
        empty,
        device_buffer,
        empty,
        num_blocks,
        item_size_bytes,
    )


def load_cache_to_device_buffer_dsv4_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    skip_io: bool = False,
) -> None:
    """DSv4 hisparse swap-in: page-padded device + page-padded host C4 layout."""
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=True,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        skip_io=skip_io,
    )


def prepare_hisparse_spec_verify(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    hisparse_out_locs: torch.Tensor,
    req_to_device_buffer: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    num_real_reqs: torch.Tensor,
    logical_to_device_mapping: torch.Tensor,
    verify_width: int,
    hot_buffer_size: int,
) -> None:
    """Bind completed C4 boundaries to the request's fixed verify tail."""
    batch = req_pool_indices.numel()
    if hisparse_out_locs.numel() != batch * verify_width:
        raise ValueError(
            "HiSparse speculative output locations must contain batch_size * verify_width entries."
        )
    assert prefix_lens.shape == req_pool_indices.shape
    assert device_buffer_tokens.shape[1:] == req_to_device_buffer.shape
    assert (
        req_pool_indices.dtype
        == prefix_lens.dtype
        == hisparse_out_locs.dtype
        == torch.int64
    )
    assert req_to_device_buffer.dtype == logical_to_device_mapping.dtype == torch.int64
    assert device_buffer_tokens.dtype == num_real_reqs.dtype == torch.int32
    assert num_real_reqs.numel() == 1
    slots = (verify_width + 3) // 4
    active = torch.arange(batch, device=req_pool_indices.device) < num_real_reqs
    # Padding may carry an invalid request ID. Mask before every pool lookup.
    rows = torch.where(active, req_pool_indices, 0)
    prefix = torch.where(active, prefix_lens, 0)[:, None]
    old_count = prefix // 4
    ordinal = torch.arange(slots, device=prefix.device)[None, :]
    generated = (prefix + verify_width) // 4 - old_count
    positions = torch.where(
        active[:, None] & (ordinal < generated), old_count + ordinal, -1
    ).to(torch.int32)
    tail = device_buffer_tokens[:, :, hot_buffer_size + 1 : hot_buffer_size + 1 + slots]
    # Padded rows safely reduce into reserved row zero with neutral value -1.
    tail.scatter_reduce_(
        1,
        rows[None, :, None].expand(tail.size(0), batch, slots),
        positions[None].expand(tail.size(0), batch, slots),
        reduce="amax",
        include_self=False,
    )
    seq_lens = prefix + torch.arange(verify_width, device=prefix.device)[None, :] + 1
    boundary = active[:, None] & (seq_lens % 4 == 0)
    columns = hot_buffer_size + 1 + (seq_lens // 4 - 1 - old_count).clamp(0, slots - 1)
    device_locs = req_to_device_buffer[rows[:, None], columns]
    logical_locs = torch.where(boundary, hisparse_out_locs.view(batch, verify_width), 0)
    # Mapping zero is the reserved null route, never an allocated C4 location.
    logical_to_device_mapping.scatter_reduce_(
        0,
        logical_locs.flatten(),
        torch.where(boundary, device_locs, 0).flatten(),
        reduce="amax",
        include_self=False,
    )


def finalize_hisparse_spec(
    *,
    device_ptrs: torch.Tensor,
    host_ptrs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    req_to_host_pool: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    req_to_token: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    logical_to_device_mapping: torch.Tensor,
    verify_width: int,
    hot_buffer_size: int,
) -> None:
    """Back up accepted C4 KV, promote persistent entries, then clear verify routes."""
    assert prefix_lens.shape == commit_lens.shape == req_pool_indices.shape
    assert device_buffer_locs.shape == device_buffer_tokens.shape
    assert (
        req_pool_indices.dtype
        == prefix_lens.dtype
        == req_to_host_pool.dtype
        == torch.int64
    )
    assert (
        commit_lens.dtype
        == device_buffer_locs.dtype
        == device_buffer_tokens.dtype
        == req_to_token.dtype
        == torch.int32
    )
    layers = device_buffer_locs.size(0)
    rows = req_pool_indices[:, None]
    ordinal = torch.arange((verify_width + 3) // 4, device=rows.device)[None, :]
    old_count = prefix_lens[:, None] // 4
    positions = old_count + ordinal
    accepted = (prefix_lens[:, None] + commit_lens[:, None]) // 4 - old_count
    valid = ordinal < accepted
    sources = device_buffer_locs[:, rows, hot_buffer_size + 1 + ordinal].to(torch.int64)
    source_plan = torch.where(valid, sources, -1).reshape(layers, -1)
    host_locs = req_to_host_pool[rows, torch.where(valid, positions, 0)]
    # Host slot zero can be allocated: rejected entries must skip, never write it.
    transfer_cache_dsv4_mla(
        device_ptrs, host_ptrs, source_plan, torch.where(valid, host_locs, -1).flatten()
    )
    persistent = valid & ((positions < hot_buffer_size) | (ordinal == accepted - 1))
    destinations = device_buffer_locs[:, rows, positions.clamp(max=hot_buffer_size)].to(
        torch.int64
    )
    transfer_cache_dsv4_mla(
        device_ptrs,
        device_ptrs,
        torch.where(persistent, sources, -1).reshape(layers, -1),
        torch.where(persistent, destinations, -1).reshape(layers, -1),
    )

    # All metadata changes follow both copies on the caller's current stream.
    tail = device_buffer_tokens[
        :, :, hot_buffer_size + 1 : hot_buffer_size + 1 + ordinal.numel()
    ]
    tail.index_fill_(1, req_pool_indices, -1)
    hot = valid & (positions < hot_buffer_size)
    # Invalid hot updates target distinct tail slots, avoiding duplicate writes
    # to H-1/C when a verify window crosses the end of the hot buffer.
    columns = torch.where(hot, positions, hot_buffer_size + 1 + ordinal)
    device_buffer_tokens[:, rows, columns] = torch.where(hot, positions, -1).to(
        torch.int32
    )[None]
    newest = (old_count + accepted - 1).flatten()
    canonical = device_buffer_tokens[:, req_pool_indices, hot_buffer_size]
    device_buffer_tokens[:, req_pool_indices, hot_buffer_size] = torch.where(
        accepted.flatten() > 0,
        torch.where(newest >= hot_buffer_size, newest, -1).to(torch.int32),
        canonical,
    )
    generated = ordinal < (prefix_lens[:, None] + verify_width) // 4 - old_count
    boundaries = torch.where(generated, (positions + 1) * 4 - 1, 0)
    logical_locs = req_to_token[rows, boundaries].to(torch.int64) // 4
    logical_to_device_mapping.index_fill_(
        0, torch.where(generated, logical_locs, 0).flatten(), 0
    )
