from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def _jit_set_mla_kv_buffer_module(
    nope_bytes: int, rope_bytes: int, use_pdl: bool
) -> Module:
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
    if nope_bytes % 4 != 0 or rope_bytes % 4 != 0:
        logger.warning(
            "Unsupported nope_bytes=%d rope_bytes=%d for JIT set_mla_kv_buffer:"
            " both must be multiples of 4",
            nope_bytes,
            rope_bytes,
        )
        return False
    try:
        _jit_set_mla_kv_buffer_module(nope_bytes, rope_bytes, is_arch_support_pdl())
        return True
    except Exception as e:  # pragma: no cover - first-use compile failures only
        logger.warning(
            "Failed to load JIT set_mla_kv_buffer kernel "
            "with nope_bytes=%d rope_bytes=%d: %s",
            nope_bytes,
            rope_bytes,
            e,
        )
        return False


def _pick_dispatch(n_loc: int, nope_bytes: int) -> tuple[int, int, int]:
    """Return (k_split, num_warps_per_block, items_per_warp).

    Strategy:
      - Small n_loc (≤ 8): split each item across 5 warps so 5 CTAs go live per item
        (more SMs saturated than 1 CTA per item).
      - Mid n_loc (≤ 128): split into 3 warps per item (3 CTAs per item).
      - Large n_loc: fat warp (k_split=1). Pack 4-8 warps per CTA. At memory-bound
        sizes (≥ 4096), have each warp do 2 items so the warp scheduler has more
        in-flight memory ops, mimicking Triton's ``num_stages=2`` pipelining.

    Tuned on GB300 (148 SMs) against the tokenspeed reference.
    """
    can_split_5 = nope_bytes % 4 == 0
    can_split_3 = nope_bytes % 2 == 0

    if n_loc <= 8 and can_split_5:
        return 5, 1, 1
    if n_loc <= 128 and can_split_3:
        return 3, 1, 1
    if n_loc <= 1024:
        return 1, 4, 1
    return 1, 8, 1


def set_mla_kv_buffer(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    k_split: int = 0,
    num_warps: int = 0,
    items_per_warp: int = 0,
) -> None:
    """Write packed [k_nope | k_rope] rows into ``kv_buffer`` at ``loc`` indices.

    All non-loc tensors must share the same dtype and device (CUDA / ROCm).
    ``loc`` is int32 or int64.

    Tuning knobs (0 = auto, picked from ``loc.shape[0]``):
        k_split:        warps cooperating on one item (1, 3, or 5).
        num_warps:      warps per CTA.
        items_per_warp: items each warp loops over (only with k_split=1).

    Shapes (last dim is treated as the row payload; any leading singleton dims
    on the source tensors are flattened away):
        kv_buffer:    [num_pages, total_dim] or [num_pages, 1, total_dim]
        cache_k_nope: [n_loc, nope_dim] or [n_loc, 1, nope_dim]
        cache_k_rope: [n_loc, rope_dim] or [n_loc, 1, rope_dim]
        loc:          [n_loc]
    """
    n_loc = loc.shape[0]
    if n_loc == 0:
        return

    src_nope = cache_k_nope.view(n_loc, -1) if cache_k_nope.dim() != 2 else cache_k_nope
    src_rope = cache_k_rope.view(n_loc, -1) if cache_k_rope.dim() != 2 else cache_k_rope
    buf = kv_buffer.view(kv_buffer.shape[0], -1) if kv_buffer.dim() != 2 else kv_buffer

    nope_bytes = src_nope.shape[-1] * src_nope.element_size()
    rope_bytes = src_rope.shape[-1] * src_rope.element_size()

    if k_split <= 0 or num_warps <= 0 or items_per_warp <= 0:
        auto_split, auto_nw, auto_ipw = _pick_dispatch(n_loc, nope_bytes)
        if k_split <= 0:
            k_split = auto_split
        if num_warps <= 0:
            num_warps = auto_nw
        if items_per_warp <= 0:
            items_per_warp = auto_ipw

    module = _jit_set_mla_kv_buffer_module(
        nope_bytes, rope_bytes, is_arch_support_pdl()
    )
    module.set_mla_kv_buffer(
        buf, loc, src_nope, src_rope, k_split, num_warps, items_per_warp
    )
