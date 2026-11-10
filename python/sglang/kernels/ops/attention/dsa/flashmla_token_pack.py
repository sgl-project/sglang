"""Opt-in V32 prefill packing with per-token top-k ownership masks.

Input top-k slots must be unique within each row. The union changes key order,
so this path is disabled for deterministic inference. Q concatenation and tail
padding are fused with union construction in the JIT prepare kernel; the packed
decode runs on the sgl-flash-mla wheel (``heads_per_group``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# FlashMLA's token-group bits (26..30) cap a group at 5 tokens and 64 packed rows.
_MAX_GROUP = 5
_MAX_PACKED_HEADS = 64


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "flashmla_token_pack",
        cuda_files=["dsa/flashmla_token_pack.cuh"],
        cuda_wrappers=[("prepare", "flashmla_token_pack_prepare")],
        extra_cuda_cflags=["-O3"],
    )


def flashmla_token_pack_available() -> bool:
    from sglang.kernels.ops.attention import flash_mla_interface

    return flash_mla_interface.supports_native_heads()


def flashmla_token_pack_prepare(
    q: torch.Tensor, q_rope: Optional[torch.Tensor], indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``q`` [T, H, 576] (or [T, H, 512] plus ``q_rope``) and ``indices``
    [>= T, topk]: (packed_q [G, 1, group * H, 576], packed_indices [G, width],
    lengths [G])."""
    tokens, heads = q.shape[0], q.shape[1]
    topk = indices.shape[1]
    group = min(_MAX_PACKED_HEADS // heads, _MAX_GROUP)
    groups = (tokens + group - 1) // group
    width = (group * topk + 63) // 64 * 64
    use_view = q_rope is None and q.is_contiguous() and tokens % group == 0
    if use_view:
        packed_q = q.view(groups, 1, group * heads, 576)
    else:
        packed_q = q.new_empty((groups, 1, group * heads, 576))
    packed_indices = indices.new_empty((groups, width))
    lengths = indices.new_empty((groups,))
    _jit_module().prepare(
        q,
        q_rope,
        indices,
        None if use_view else packed_q,
        packed_indices,
        lengths,
        group,
    )
    return packed_q, packed_indices, lengths


def flashmla_kv_token_packed(q, q_rope, kv_cache, indices, sm_scale):
    """Return [T, 1, H, 512], or None for unsupported widths/layouts.

    The JIT launcher validates matching shapes, devices and index dtype.
    """
    if (
        q.dtype != torch.bfloat16
        or q.shape[0] == 0
        or not 1 <= q.shape[1] <= 32
        or q.shape[2] != (512 if q_rope is not None else 576)
        or kv_cache.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or kv_cache.shape[1:] != (64, 1, 656)
        or kv_cache.shape[0] * 64 > 1 << 26
        or not 1 <= indices.shape[1] <= 2048
        or indices.stride(1) != 1
    ):
        return None
    for part in (q, q_rope):
        if part is not None and (
            part.stride(-1) != 1
            or part.stride(0) % 8
            or part.stride(1) % 8
            or part.data_ptr() % 16
        ):
            return None
    from sglang.kernels.ops.attention import flash_mla_interface

    packed_q, packed_indices, lengths = flashmla_token_pack_prepare(q, q_rope, indices)
    out, _ = flash_mla_interface.flash_mla_with_kvcache(
        q=packed_q,
        k_cache=kv_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=flash_mla_interface.get_mla_metadata()[0],
        softmax_scale=sm_scale,
        is_fp8_kvcache=True,
        indices=packed_indices.unsqueeze(1),
        topk_length=lengths,
        kv_format="V32",
        heads_per_group=q.shape[1],
    )
    return out.view(-1, 1, q.shape[1], 512)[: q.shape[0]]
