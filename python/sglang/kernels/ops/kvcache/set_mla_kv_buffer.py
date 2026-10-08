"""JIT TMA bulk-store path for ``set_mla_kv_buffer``.

Each warp scatter-writes one item's (nope, rope) row via a single
``cp.async.bulk.global.shared::cta`` store. Requires SM90+ (Hopper or later)
for the TMA bulk-store hardware. The host-side wrapper in
``sglang.srt.mem_cache.utils`` falls back to a Triton kernel for older arches.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def set_mla_kv_buffer_module(nope_bytes: int, rope_bytes: int, use_pdl: bool) -> Module:
    args = make_cpp_args(nope_bytes, rope_bytes, use_pdl)
    return load_jit(
        f"set_mla_kv_buffer_{nope_bytes}_{rope_bytes}",
        *args,
        cuda_files=["elementwise/set_mla_kv_buffer.cuh"],
        cuda_wrappers=[
            ("set_mla_kv_buffer", f"SetMlaKVBufferKernel<{args}>::run"),
        ],
    )


@cache_once
def can_use_set_mla_kv_buffer(nope_bytes: int, rope_bytes: int) -> bool:
    if (rope_bytes + nope_bytes) % 16 != 0:
        return False
    try:
        set_mla_kv_buffer_module(nope_bytes, rope_bytes, is_arch_support_pdl())
        return True
    except Exception as e:  # pragma: no cover - compile-time only
        logger.warning(
            "Failed to load JIT set_mla_kv_buffer kernel "
            "with nope_bytes=%d rope_bytes=%d: %s",
            nope_bytes,
            rope_bytes,
            e,
        )
        return False


def _pick_num_warps(n_loc: int) -> int:
    # Tuned on GB300: nw=4 wins below 1024 (more CTAs spread across SMs);
    # nw=8 wins above (each CTA amortises the bulk-group commit better).
    return 4 if n_loc <= 768 else 8


def set_mla_kv_buffer(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    num_warps: int = 0,
    *,
    reserved_skip_index: int = 0,
) -> None:
    """Write packed [k_nope | k_rope] rows into ``kv_buffer`` at ``loc`` indices
    via a TMA bulk-store. SM90+ only — the caller is expected to gate.

    Shapes (last dim is treated as the row payload; any leading singleton dims
    on the source tensors are flattened away):
        kv_buffer:    [num_pages, total_dim] or [num_pages, 1, total_dim]
        cache_k_nope: [n_loc, nope_dim] or [n_loc, 1, nope_dim]
        cache_k_rope: [n_loc, rope_dim] or [n_loc, 1, rope_dim]
        loc:          [n_loc]

    Writes targeting ``reserved_skip_index`` are skipped. Slot 0 is reserved
    for CUDA-graph padding by default; pass -1 to disable skipping.
    """
    n_loc = loc.shape[0]
    if n_loc == 0:
        return

    src_nope = cache_k_nope.view(n_loc, -1) if cache_k_nope.dim() != 2 else cache_k_nope
    src_rope = cache_k_rope.view(n_loc, -1) if cache_k_rope.dim() != 2 else cache_k_rope
    buf = kv_buffer.view(kv_buffer.shape[0], -1) if kv_buffer.dim() != 2 else kv_buffer

    nope_bytes = src_nope.shape[-1] * src_nope.element_size()
    rope_bytes = src_rope.shape[-1] * src_rope.element_size()
    if num_warps <= 0:
        num_warps = _pick_num_warps(n_loc)

    module = set_mla_kv_buffer_module(nope_bytes, rope_bytes, is_arch_support_pdl())
    module.set_mla_kv_buffer(
        buf,
        loc,
        src_nope,
        src_rope,
        num_warps,
        reserved_skip_index,
    )


@cache_once
def set_sharded_mla_kv_buffer_module(
    nope_bytes: int, rope_bytes: int, use_pdl: bool
) -> Module:
    args = make_cpp_args(nope_bytes, rope_bytes, use_pdl)
    return load_jit(
        f"set_sharded_mla_kv_buffer_{nope_bytes}_{rope_bytes}",
        *args,
        cuda_files=["elementwise/set_mla_kv_buffer.cuh"],
        cuda_wrappers=[("store", f"SetShardedMlaKVBufferKernel<{args}>::run")],
    )


@cache_once
def can_use_set_sharded_mla_kv_buffer(nope_bytes: int, rope_bytes: int) -> bool:
    """Compile-time capability gate for the raw-FP8 dual TMA store.

    Other layouts and devices retain the existing two-writer path. Only JIT
    construction errors are caught here; launch failures must propagate.
    """
    if (nope_bytes, rope_bytes) != (512, 64):
        return False
    if (
        torch.version.cuda is None
        or torch.version.hip is not None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] < 9
    ):
        return False
    try:
        set_sharded_mla_kv_buffer_module(nope_bytes, rope_bytes, is_arch_support_pdl())
        return True
    except Exception as e:  # pragma: no cover - compiler/toolchain dependent
        logger.warning("Failed to load sharded MLA TMA store: %s", e)
        return False


def _sharded_mla_kv_buffer_input_error(
    scratch: torch.Tensor,
    scratch_loc: torch.Tensor,
    shard_buffer: torch.Tensor,
    logical_loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    page_size: int,
    shard_size: int,
    shard_rank: int,
    *,
    scratch_reserved_skip_index: int = 0,
    local_reserved_skip_index: int = 0,
) -> str | None:
    if not all(
        isinstance(v, int)
        for v in (
            page_size,
            shard_size,
            shard_rank,
            scratch_reserved_skip_index,
            local_reserved_skip_index,
        )
    ):
        return "sharding and reserved-index parameters must be integers"
    if page_size <= 0 or shard_size <= 1 or not 0 <= shard_rank < shard_size:
        return "invalid page sharding parameters"
    if scratch_reserved_skip_index < -1 or local_reserved_skip_index < -1:
        return "reserved indices must be nonnegative, or -1 to disable skipping"
    for loc in (scratch_loc, logical_loc):
        if loc.ndim != 1 or loc.dtype not in (torch.int32, torch.int64):
            return "locations must be one-dimensional int32 or int64 tensors"
        if loc.stride(0) != 1:
            return "locations must be contiguous"
    if scratch_loc.shape != logical_loc.shape or scratch_loc.dtype != logical_loc.dtype:
        return "scratch and logical locations must have matching lengths and dtypes"
    n = logical_loc.numel()
    if n > 2**32 - 1:
        return "too many input rows"
    for t, width in ((cache_k_nope, 512), (cache_k_rope, 64)):
        if t.ndim not in (2, 3) or (t.ndim == 3 and t.shape[1] != 1):
            return "sources must have shape [n, dim] or [n, 1, dim]"
        if t.shape[0] != n or t.shape[-1] != width or t.dtype != torch.uint8:
            return "sources must be preconverted uint8 raw-FP8 rows matching loc length"
        if t.stride(-1) != 1 or t.stride(0) < width:
            return "source rows must be contiguous and non-overlapping"
        if n and (t.data_ptr() % 16 or t.stride(0) % 16):
            return "source rows must be 16-byte aligned"
    for t in (scratch, shard_buffer):
        if t.ndim not in (2, 3) or (t.ndim == 3 and t.shape[1] != 1):
            return "destinations must have shape [rows, dim] or [rows, 1, dim]"
        if t.dtype != torch.uint8 or t.shape[-1] < 576:
            return "destinations must be uint8 with at least 576 bytes per row"
        if n and t.shape[0] == 0:
            return "nonempty input requires nonempty destinations"
        if t.stride(-1) != 1 or t.stride(0) < t.shape[-1]:
            return "destination rows must be contiguous and non-overlapping"
        if t.data_ptr() % 16 or t.stride(0) % 16:
            return "destination rows must be 16-byte aligned"
    tensors = (
        scratch,
        scratch_loc,
        shard_buffer,
        logical_loc,
        cache_k_nope,
        cache_k_rope,
    )
    if scratch.device.type != "cuda" or any(
        t.device != scratch.device for t in tensors
    ):
        return "all inputs must be on the same CUDA device"
    if scratch.device.index != torch.cuda.current_device():
        return "the input device must match the current JIT CUDA device"
    return None


def sharded_mla_kv_buffer_inputs_supported(*args, **kwargs) -> bool:
    """Metadata-only gate; does not copy tensor data or synchronize CUDA.

    Location values, bounds, and uniqueness are guaranteed by the allocator,
    as for the existing single-destination writer. The two destinations must
    refer to separate allocations. This helper checks layouts before the
    caller chooses the fused path, avoiding exception-based runtime fallback.
    """
    return _sharded_mla_kv_buffer_input_error(*args, **kwargs) is None


def set_sharded_mla_kv_buffer(
    scratch: torch.Tensor,
    scratch_loc: torch.Tensor,
    shard_buffer: torch.Tensor,
    logical_loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    page_size: int,
    shard_size: int,
    shard_rank: int,
    *,
    scratch_reserved_skip_index: int = 0,
    local_reserved_skip_index: int = 0,
) -> None:
    """Reuse a raw-FP8 staging row for scratch and the owner-local TMA store.

    Keep PyTorch's original FP8 conversion in the caller: no conversion or
    saturation is performed here, including for overflow, infinities or NaNs.
    Reserved indices are independent physical indices in the two destination
    pools. Pass -1 separately for a pool whose slot zero is writable.
    """
    error = _sharded_mla_kv_buffer_input_error(
        scratch,
        scratch_loc,
        shard_buffer,
        logical_loc,
        cache_k_nope,
        cache_k_rope,
        page_size,
        shard_size,
        shard_rank,
        scratch_reserved_skip_index=scratch_reserved_skip_index,
        local_reserved_skip_index=local_reserved_skip_index,
    )
    if error is not None:
        raise ValueError(error)
    n = logical_loc.numel()
    if n == 0:
        return
    set_sharded_mla_kv_buffer_module(512, 64, is_arch_support_pdl()).store(
        scratch.view(scratch.shape[0], scratch.shape[-1]),
        scratch_loc,
        cache_k_nope.view(n, 512),
        cache_k_rope.view(n, 64),
        shard_buffer.view(shard_buffer.shape[0], shard_buffer.shape[-1]),
        logical_loc,
        page_size,
        shard_size,
        shard_rank,
        _pick_num_warps(n),
        scratch_reserved_skip_index,
        local_reserved_skip_index,
    )
