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
    pass_through_device_locs: bool = False,
    late_bound_host_base: bool = False,
    workspace_capacity: int | None = None,
) -> Module:
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        item_size_bytes,
        num_steps,
        record_miss_plan,
        is_arch_support_pdl(),
        pass_through_device_locs,
        late_bound_host_base,
        workspace_capacity if workspace_capacity is not None else num_steps * num_top_k,
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
    host_cache: torch.Tensor | None,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    state: HiSparseSpecState,
    num_real_reqs: torch.Tensor,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    device_locs: torch.Tensor | None = None,
    host_binding: torch.Tensor | None = None,
) -> None:
    """Resolve all speculative steps and swap unique misses in one launch pair.

    Optional miss-plan outputs use the same protocol as the single-step HiSparse
    kernel, so shared-index layers can replay only the Host-to-GPU copies with
    ``copy_cache_planned_mla``.

    ``device_locs`` enables resident KV passthrough and single-step draft calls
    using the same hot/scratch state. ``host_binding`` supplies the runtime
    [host base address, row stride in bytes] for the shared HiCache host pool.
    """
    _, num_steps, num_top_k = top_k_tokens.shape
    if not 2 <= num_steps <= 4 and not (num_steps == 1 and device_locs is not None):
        raise ValueError(
            f"HiSparse speculative swap requires 2-4 steps, or one resident-device step; got {num_steps}."
        )
    hot_buffer_size = state.cache_policy.size(1)
    page_size = device_buffer_tokens.size(1) - hot_buffer_size
    item_size_bytes = device_buffer.stride(0) * device_buffer.element_size()
    if host_cache is None and host_binding is None:
        raise ValueError(
            "HiSparse swap requires a host cache or a late-bound host address."
        )
    if (
        req_pool_indices.numel() != top_k_tokens.shape[0]
        or seq_lens.numel() != top_k_tokens.shape[0] * num_steps
    ):
        raise ValueError("HiSparse speculative request/query dimensions do not match.")
    if device_locs is not None and (
        device_locs.dtype != torch.int32
        or device_locs.shape != host_cache_locs.shape
        or device_locs.stride(1) != 1
        or device_locs.device != device_buffer.device
    ):
        raise ValueError("device_locs must be a GPU int32 request-to-token table.")
    if host_binding is not None and (
        host_binding.dtype != torch.int64
        or host_binding.shape != (2,)
        or not host_binding.is_contiguous()
        or host_binding.device != device_buffer.device
    ):
        raise ValueError("host_binding must be a contiguous GPU int64 pair.")
    if host_binding is None and (
        host_cache.stride(0) * host_cache.element_size() != item_size_bytes
    ):
        raise ValueError("Host and device KV row widths must match.")
    if top_k_tokens.size(0) == 0:
        return
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
        device_locs is not None,
        host_binding is not None,
        # Draft and verification reuse metadata without changing its hash layout.
        min(8192, state.scratch_state.size(1) // 5),
    )
    empty_device = torch.empty(0, device=device_buffer.device)

    module.load_cache_to_device_buffer_spec(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache if host_cache is not None else torch.empty(0),
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
        device_locs if device_locs is not None else empty_device,
        host_binding if host_binding is not None else empty_device,
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
    pass_through_device_locs: bool = False,
    late_bound_host_base: bool = False,
) -> Module:
    # The four bool flags are compile-time kernel flags; the all-False
    # private-host instantiation stays byte-identical.
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
        pass_through_device_locs,
        late_bound_host_base,
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
        pass_through_device_locs,
        late_bound_host_base,
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
    """Transfer DSv4 C4 tokens between page-padded C4 buffers."""
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
    host_cache: torch.Tensor | None,
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
    device_locs: torch.Tensor | None,
    host_binding: torch.Tensor | None,
) -> None:
    assert hot_buffer_size >= num_top_k, (
        f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"
    )

    record_miss_plan = miss_src is not None
    # Both extra sources are opt-in per caller: passing the tensor compiles that
    # branch in, omitting it keeps the two-source private-host kernel.
    pass_through_device_locs = device_locs is not None
    late_bound_host_base = host_binding is not None
    module = _jit_sparse_module(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla=True,
        is_dsv4_layout=is_dsv4_layout,
        record_miss_plan=record_miss_plan,
        skip_io=skip_io,
        pass_through_device_locs=pass_through_device_locs,
        late_bound_host_base=late_bound_host_base,
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

    if pass_through_device_locs:
        assert device_locs.dtype == torch.int32, (
            "device_locs must be int32 (it is the pool's own index table), got "
            f"{device_locs.dtype}"
        )
    else:
        # Unused sentinel; the PassThroughDeviceLocs=false instantiation never
        # reads it.
        device_locs = empty

    if late_bound_host_base:
        # Both elements are read on the device at launch time, so a short tensor
        # is a silently wrong stride rather than an error.
        assert host_binding.dtype == torch.int64 and host_binding.numel() == 2, (
            "host_binding must be a two-element int64 tensor [base address, row "
            f"stride in bytes], got {host_binding.dtype} {host_binding.numel()=}"
        )
    else:
        assert host_cache is not None, "no host source: pass host_cache or host_binding"
        # Unused sentinel; the LateBoundHostBase=false instantiation never reads it.
        host_binding = empty
    # None only under the binding, whose caller has no per-layer view to pass.
    host_cache = empty if host_cache is None else host_cache

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        device_locs,
        host_binding,
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
    host_cache: torch.Tensor | None,
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
    device_locs: torch.Tensor | None = None,
    host_binding: torch.Tensor | None = None,
) -> None:
    """Generic MLA hisparse swap-in: device + host both linear (stride=item_size_bytes).

    Optional miss_src/miss_dst/miss_count record the miss plan for replay by
    copy_cache_planned_mla; skip_io elides only the KV bytes (timing probe, or the
    plan-only half when the caller replays the recorded plan itself).

    Optional device_locs adds the pool as a third source: `device_locs[req, pos]`
    >= 0 means the position is live in the regular pool, so that slot is returned
    with no hot-buffer slot and no copy; it also masks (-1) a lane the selector
    left unfilled. Callers whose staging freed every pool slot leave it None.

    Optional host_binding replaces `host_cache` as the source layout, reading the
    host pool's [base address, row stride in bytes] from a two-element int64
    tensor at run time: required when the host tier attaches after cuda-graph
    capture, or when its rows are not one cell apart (HiCache's page_first
    interleaves layers within a token).
    """
    _load_cache_to_device_buffer_mla(
        device_locs=device_locs,
        host_binding=host_binding,
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
        empty,
        empty,
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
    Only the private-host backing plans, and it owns its host pool from init, so
    there is no late-bound host source here.
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
    host_cache: torch.Tensor | None,
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
    """DSv4 hisparse swap-in: page-padded device + page-padded host C4 layout.

    No pool source: DeepSeek V4 is private-host only, so every selected position
    is either in the request's hot buffer or on host.
    """
    _load_cache_to_device_buffer_mla(
        device_locs=None,
        host_binding=None,
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
