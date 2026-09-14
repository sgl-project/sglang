"""SGL-owned native providers for MiniMax sparse main attention."""

from __future__ import annotations

from typing import Optional

import torch


class SglNativeQ8KV8UnavailableError(RuntimeError):
    """Raised when the initial native Q8KV8 provider cannot serve a call."""


def _validate_q8kv8_contract(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size_k: int,
) -> None:
    tensors = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "topk_idx": topk_idx,
        "req_to_token": req_to_token,
        "slot_ids": slot_ids,
        "cu_seqlens": cu_seqlens,
        "seq_lens": seq_lens,
        "prefix_lens": prefix_lens,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise SglNativeQ8KV8UnavailableError(f"{name} must be on CUDA")
        if name != "q" and not tensor.is_contiguous():
            raise SglNativeQ8KV8UnavailableError(f"{name} must be contiguous")
    if q.stride(-1) != 1:
        raise SglNativeQ8KV8UnavailableError(
            "q last dimension must be contiguous, "
            f"got shape={tuple(q.shape)}, stride={tuple(q.stride())}"
        )
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise SglNativeQ8KV8UnavailableError("SM90 is required")
    if q.dtype != torch.float8_e4m3fn:
        raise SglNativeQ8KV8UnavailableError(f"FP8 E4M3 Q is required, got {q.dtype}")
    if k_cache.dtype != torch.float8_e4m3fn:
        raise SglNativeQ8KV8UnavailableError(
            f"FP8 E4M3 K cache is required, got {k_cache.dtype}"
        )
    if v_cache.dtype != torch.float8_e4m3fn:
        raise SglNativeQ8KV8UnavailableError(
            f"FP8 E4M3 V cache is required, got {v_cache.dtype}"
        )
    if q.shape[-1] != 128 or k_cache.shape[-1] != 128:
        raise SglNativeQ8KV8UnavailableError("head_dim=128 is required")
    if block_size_k != 128:
        raise SglNativeQ8KV8UnavailableError("block_size_k=128 is required")
    if q.shape[1] % k_cache.shape[1] != 0:
        raise SglNativeQ8KV8UnavailableError(
            "local Q heads must be divisible by local KV heads"
        )
    group_size = q.shape[1] // k_cache.shape[1]
    if group_size not in (1, 2, 4, 8, 16):
        raise SglNativeQ8KV8UnavailableError(
            f"unsupported local GQA group size: {group_size}"
        )
    if topk_idx.shape[:2] != (k_cache.shape[1], q.shape[0]):
        raise SglNativeQ8KV8UnavailableError(
            "topk_idx must have shape [local_num_kv_heads, total_q, topk]"
        )
    if req_to_token.dtype != torch.int32:
        raise SglNativeQ8KV8UnavailableError("req_to_token must be int32")
    if slot_ids.dtype != torch.int64:
        raise SglNativeQ8KV8UnavailableError("slot_ids must be int64")
    for name, tensor in (
        ("topk_idx", topk_idx),
        ("cu_seqlens", cu_seqlens),
        ("seq_lens", seq_lens),
        ("prefix_lens", prefix_lens),
    ):
        if tensor.dtype != torch.int32:
            raise SglNativeQ8KV8UnavailableError(f"{name} must be int32")


def sgl_native_q8kv8_sparse_prefill_main(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size_k: int,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    """Adapt MiniMax Step-3 metadata to the model-independent native operator."""
    _validate_q8kv8_contract(
        q,
        k_cache,
        v_cache,
        topk_idx,
        req_to_token,
        slot_ids,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        block_size_k,
    )
    from sglang.kernels.ops.attention.minimax_sparse.prefill.sgl_native_q8kv8 import (
        SglNativeQ8KV8BuildError,
        sgl_native_q8kv8_sparse_prefill,
    )

    try:
        return sgl_native_q8kv8_sparse_prefill(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            topk_idx=topk_idx,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            block_size_k=block_size_k,
            sm_scale=sm_scale,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    except SglNativeQ8KV8BuildError as err:
        raise SglNativeQ8KV8UnavailableError(str(err)) from err
