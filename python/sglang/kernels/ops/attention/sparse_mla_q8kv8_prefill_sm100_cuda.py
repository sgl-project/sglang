"""CUDA/tcgen05 launcher for SM100 Q8KV8 sparse MLA prefill."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _cuda_flags() -> list[str]:
    return [
        "-O3",
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "--use_fast_math",
        "--threads=4",
    ]


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "sparse_mla_q8kv8_prefill_sm100_cuda",
        cuda_files=["sparse_mla_q8kv8_prefill_sm100/entry.cuh"],
        cuda_wrappers=[
            ("dispatch", "sparse_prefill_q8kv8_dispatch"),
            ("dispatch_full", "sparse_prefill_q8kv8_dispatch_full"),
            ("dispatch_topk_length", "sparse_prefill_q8kv8_dispatch_topk_length"),
            ("dispatch_full_active", "sparse_prefill_q8kv8_dispatch_full_active"),
        ],
        extra_cuda_cflags=_cuda_flags(),
        extra_dependencies=["cutlass"],
    )


_entries: tuple | None = None


def _get_entries() -> tuple:
    global _entries
    if _entries is None:
        module = _jit_module()
        _entries = (
            module["dispatch"],
            module["dispatch_full"],
            module["dispatch_topk_length"],
            module["dispatch_full_active"],
        )
    return _entries


def sparse_mla_q8kv8_prefill_fwd_sm100_cuda(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor | None,
    out: torch.Tensor,
    max_logits: torch.Tensor,
    lse: torch.Tensor,
    active_heads: int | None = None,
) -> None:
    """Launch the caller-buffer SM100 CUDA implementation."""
    s_q, q_storage_heads, d_qk = q.shape
    h_q = (
        ((active_heads + 31) // 32) * 32
        if active_heads is not None
        else q_storage_heads
    )
    s_kv, h_kv, _ = kv.shape
    topk = indices.shape[-1]
    stream = torch._C._cuda_getCurrentRawStream(q.device.index)
    common = (
        q,
        kv,
        indices,
        q_scale,
        kv_scale,
    )
    tail = (
        out,
        max_logits,
        lse,
        s_q,
        s_kv,
        h_q,
        h_kv,
        d_qk,
        512,
        topk,
        sm_scale,
        stream,
    )
    dispatch, dispatch_full, dispatch_topk_length, dispatch_full_active = _get_entries()
    if active_heads is not None:
        assert attn_sink is not None and topk_length is not None
        dispatch_full_active(
            *common,
            attn_sink,
            topk_length,
            *tail[:-2],
            active_heads,
            *tail[-2:],
        )
        return
    if attn_sink is not None:
        assert topk_length is not None
        dispatch_full(*common, attn_sink, topk_length, *tail)
    elif topk_length is not None:
        dispatch_topk_length(*common, topk_length, *tail)
    else:
        dispatch(*common, *tail)
