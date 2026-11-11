"""JIT-compiled Mamba KV cache transfer kernel.

Provides ``transfer_kv_mamba_pf_lf`` (load: page_first -> layer_first)
and ``transfer_kv_mamba_lf_pf`` (backup: layer_first -> page_first).

Uses the shared ``load_jit`` + ``cache_once`` infrastructure from
``sglang.kernels.jit.utils`` — the same mechanism used by ``hicache.py``
for MHA/MLA staged write-back kernels.  This ensures consistent
content-addressed caching, CUDA arch detection, and multi-worker
JIT compilation behavior across all JIT kernels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import cache_once, is_hip_runtime, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.kvcache.hicache import hicache_block_quota_from_env
from sglang.srt.environ import envs

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# Split chunks are whole 16 KiB block steps of the 1024-thread copy loop (16 B per thread).
_SPLIT_CHUNK_ALIGN = 16 * 1024


@cache_once
def mamba_split_block_quotas() -> tuple[int | None, int | None]:
    """(load, backup) CTA quotas of the split Mamba state copies; None keeps the stock kernel."""
    if is_hip_runtime():
        return None, None
    return (
        hicache_block_quota_from_env(envs.SGLANG_HICACHE_MAMBA_LOAD_BLOCK_QUOTA),
        hicache_block_quota_from_env(envs.SGLANG_HICACHE_MAMBA_BACKUP_BLOCK_QUOTA),
    )


def _split_chunk_bytes(*, total_bytes: int, item_size: int, block_quota: int) -> int:
    """One chunk per CTA for a small copy; whole items once every CTA gets one or more."""
    per_cta = -(-total_bytes // block_quota)
    aligned = max(-(-per_cta // _SPLIT_CHUNK_ALIGN), 1) * _SPLIT_CHUNK_ALIGN
    return min(aligned, item_size)


@cache_once
def _jit_transfer_mamba_split_module() -> Module:
    return load_jit(
        "transfer_mamba_split",
        cuda_files=["kvcacheio/transfer_mamba_split.cuh"],
        cuda_wrappers=[
            ("transfer_kv_mamba_pf_lf", "&TransferMambaSplitKernel::run_pf_lf"),
            ("transfer_kv_mamba_lf_pf", "&TransferMambaSplitKernel::run_lf_pf"),
        ],
    )


def warm_up_mamba_split_kernels() -> tuple[int | None, int | None]:
    """Build the split kernels now if a quota is set (else the first transfer compiles them)."""
    quotas = mamba_split_block_quotas()
    if any(q is not None for q in quotas):
        _jit_transfer_mamba_split_module()
    return quotas


_split_logged: set[str] = set()


def _log_split_once(
    *, direction: str, units: int, item_size: int, chunk_bytes: int, block_quota: int
) -> None:
    if direction in _split_logged:
        return
    _split_logged.add(direction)
    logger.info(
        "HiCache Mamba %s: split kernel quota=%d units=%d item_bytes=%d chunk_bytes=%d grid=%d",
        direction,
        block_quota,
        units,
        item_size,
        chunk_bytes,
        min(block_quota, units),
    )


@cache_once
def _jit_transfer_mamba_module() -> Module:
    return load_jit(
        "transfer_mamba",
        cuda_files=["kvcacheio/transfer_mamba.cuh"],
        cuda_wrappers=[
            ("transfer_kv_mamba_pf_lf", "&TransferMambaKernel::run_pf_lf"),
            ("transfer_kv_mamba_lf_pf", "&TransferMambaKernel::run_lf_pf"),
        ],
    )


@debug_kernel_api
def transfer_kv_mamba_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    num_warps_per_item: int = 32,
):
    block_quota = mamba_split_block_quotas()[0]
    if block_quota is not None:
        num_items = src_indices.numel()
        chunk_bytes = _split_chunk_bytes(
            total_bytes=num_items * item_size,
            item_size=item_size,
            block_quota=block_quota,
        )
        _log_split_once(
            direction="load",
            units=num_items * -(-item_size // chunk_bytes),
            item_size=item_size,
            chunk_bytes=chunk_bytes,
            block_quota=block_quota,
        )
        _jit_transfer_mamba_split_module().transfer_kv_mamba_pf_lf(
            src,
            dst,
            src_indices,
            dst_indices,
            layer_id,
            item_size,
            src_layout_dim,
            block_quota,
            chunk_bytes,
        )
        return
    module = _jit_transfer_mamba_module()
    module.transfer_kv_mamba_pf_lf(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
    )


@debug_kernel_api
def transfer_kv_mamba_lf_pf(
    src_ptrs: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    num_warps_per_item: int = 32,
):
    block_quota = mamba_split_block_quotas()[1]
    if block_quota is not None:
        num_items = src_indices.numel()
        chunk_bytes = _split_chunk_bytes(
            total_bytes=num_items * num_layers * item_size,
            item_size=item_size,
            block_quota=block_quota,
        )
        _log_split_once(
            direction="backup",
            units=num_items * num_layers * -(-item_size // chunk_bytes),
            item_size=item_size,
            chunk_bytes=chunk_bytes,
            block_quota=block_quota,
        )
        _jit_transfer_mamba_split_module().transfer_kv_mamba_lf_pf(
            src_ptrs,
            dst,
            src_indices,
            dst_indices,
            item_size,
            dst_layout_dim,
            num_layers,
            block_quota,
            chunk_bytes,
        )
        return
    module = _jit_transfer_mamba_module()
    module.transfer_kv_mamba_lf_pf(
        src_ptrs,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
    )
