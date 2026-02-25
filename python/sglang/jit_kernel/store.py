from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_store_module() -> Module:
    return load_jit(
        "store",
        cuda_files=["memory/store.cuh"],
        cuda_wrappers=[("store_kv_cache", "store_kv_cache")],
    )


def store_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Store key and value tensors into KV cache at specified indices.

    Args:
        k_cache: Key cache tensor, first dim is max_tokens.
        v_cache: Value cache tensor, first dim is max_tokens.
        out_loc: Token indices, shape (num_tokens,), dtype int32 or int64.
        k: Key tensor, first dim is num_tokens.
        v: Value tensor, first dim is num_tokens.
    """
    max_tokens = k_cache.size(0)
    num_tokens = out_loc.size(0)
    module = _jit_store_module()
    module.store_kv_cache(
        k_cache.view(max_tokens, -1),
        v_cache.view(max_tokens, -1),
        out_loc,
        k.view(num_tokens, -1),
        v.view(num_tokens, -1),
    )
