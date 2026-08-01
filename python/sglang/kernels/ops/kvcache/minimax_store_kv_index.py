from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(head_bytes: int) -> Module:
    # Build marker is (head_bytes, kUsePDL); the index dtype (int32/int64) is a
    # runtime dispatch inside the C++ launcher.
    args = make_cpp_args(head_bytes, is_arch_support_pdl())
    return load_jit(
        "minimax_store_kv_index",
        *args,
        cuda_files=["minimax/fused_store_kv_index.cuh"],
        cuda_wrappers=[("store_kv_index", f"store_kv_index<{args}>")],
    )


def store_kv_index(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v: Optional[torch.Tensor],
    idx_v_cache: Optional[torch.Tensor],
    indices: torch.Tensor,
    *,
    num_kv_heads: int,
    head_bytes: int,
) -> None:
    """Fused store of the MiniMax-M3 sparse caches in one launch.

    Writes the main ``k``/``v`` (``num_kv_heads`` heads each), the single index
    ``idx_k`` head, and optionally the single ``idx_v`` head into their caches
    at the per-token rows given by ``indices`` (out_cache_loc). In-place on the
    four cache tensors.

    All tensors must share the same (store) dtype and a head_dim whose byte size
    equals ``head_bytes`` (a multiple of 16). ``k``/``idx_k`` are 2D rows
    ``[T, num_kv_heads*head_dim]`` / ``[T, head_dim]``; caches are the matching
    ``[num_pages, ...]`` buffers. When ``idx_v`` is None there is no index value
    head (the layer is a pure block selector).
    """
    has_v = idx_v is not None
    if not has_v:
        # Pass idx_k as a dummy for the unused index-V slot; heads_per_token is
        # set so the kernel never reaches the index-V branch.
        idx_v = idx_k
        idx_v_cache = idx_k_cache
    heads_per_token = 2 * num_kv_heads + 1 + (1 if has_v else 0)

    module = _jit_module(head_bytes)
    module.store_kv_index(
        k,
        v,
        k_cache,
        v_cache,
        idx_k,
        idx_k_cache,
        idx_v,
        idx_v_cache,
        indices,
        num_kv_heads,
        heads_per_token,
    )
