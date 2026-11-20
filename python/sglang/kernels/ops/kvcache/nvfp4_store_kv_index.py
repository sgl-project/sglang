from __future__ import annotations

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


@cache_once
def _jit_module(head_dim: int) -> Module:
    # Build marker is (head_dim, kUsePDL); the index dtype (int32/int64) is a
    # runtime dispatch inside the C++ launcher.
    args = make_cpp_args(head_dim, is_arch_support_pdl())
    return load_jit(
        "minimax_nvfp4_store_kv_index",
        *args,
        cuda_files=["minimax/nvfp4_store_kv_index.cuh"],
        cuda_wrappers=[("nvfp4_store_kv_index", f"nvfp4_store_kv_index<{args}>")],
    )


def nvfp4_store_kv_index(
    k: torch.Tensor,
    v: torch.Tensor,
    idx_k: torch.Tensor,
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    v_payload: torch.Tensor,
    v_scale: torch.Tensor,
    idx_k_payload: torch.Tensor,
    idx_k_scale: torch.Tensor,
    indices: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """Quantize main K/V and index K to NVFP4 and scatter into their caches.

    ``k``/``v`` are ``[tokens, num_kv_heads, head_dim]`` BF16 and may be strided
    views of a combined projection; ``idx_k`` is ``[tokens, head_dim]``. The six
    cache tensors are the pool's packed-payload and E4M3 block-scale buffers,
    written in place at the physical slots given by ``indices``.

    Byte-identical to the eager path (``NVFP4KVQuantizeUtil.quantize`` followed
    by the pool scatter) on the architectures this kernel is enabled for.
    """
    head_dim = k.shape[-1]
    _jit_module(head_dim).nvfp4_store_kv_index(
        k,
        v,
        idx_k,
        k_payload,
        k_scale,
        v_payload,
        v_scale,
        idx_k_payload,
        idx_k_scale,
        indices,
        global_scale,
        page_size,
    )
