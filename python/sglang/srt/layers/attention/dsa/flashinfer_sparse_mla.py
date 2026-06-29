from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Callable

import torch

_REQUIRED_FLASHINFER_KWARGS = {
    "backend",
    "kv_scale_format",
    "out",
    "sparse_mla_top_k",
}


def validate_flashinfer_sparse_mla_skip_softmax(
    phase: str, threshold_scale_factor: float | None
) -> None:
    if threshold_scale_factor is None:
        return

    env_name = (
        "SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR"
        if phase == "prefill"
        else "SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR"
    )
    raise ValueError(
        "flashinfer_sparse_mla does not support skip-softmax for sparse MLA; "
        f"unset {env_name}."
    )


@lru_cache(maxsize=1)
def get_flashinfer_sparse_mla_op() -> Callable:
    try:
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "flashinfer_sparse_mla requires FlashInfer's merged SM120 sparse MLA "
            "API (flashinfer-ai/flashinfer#3395). Install FlashInfer >=0.6.14rc1 "
            "or a newer source build with matching flashinfer-cubin artifacts."
        ) from exc

    parameters = inspect.signature(trtllm_batch_decode_with_kv_cache_mla).parameters
    missing = sorted(_REQUIRED_FLASHINFER_KWARGS.difference(parameters))
    if missing:
        raise RuntimeError(
            "flashinfer_sparse_mla requires FlashInfer's merged SM120 sparse MLA "
            "API (flashinfer-ai/flashinfer#3395); the installed API is missing "
            f"arguments {missing}. Install FlashInfer >=0.6.14rc1 or a newer "
            "source build with matching flashinfer-cubin artifacts."
        )

    return trtllm_batch_decode_with_kv_cache_mla


def flashinfer_sparse_mla_forward(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    page_size: int,
    kv_cache_dim: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    sm_scale: float,
) -> torch.Tensor:
    """Run FlashInfer's SM120 sparse MLA kernel on SGLang's packed DSA cache."""
    if page_size != 64:
        raise ValueError(
            f"flashinfer_sparse_mla requires page_size=64, got {page_size}."
        )
    if kv_lora_rank != 512 or qk_rope_head_dim != 64 or v_head_dim != 512:
        raise ValueError(
            "flashinfer_sparse_mla requires kv_lora_rank=512, "
            "qk_rope_head_dim=64, and v_head_dim=512; "
            f"got {kv_lora_rank=}, {qk_rope_head_dim=}, {v_head_dim=}."
        )
    if kv_cache_dim != 656:
        raise ValueError(
            "flashinfer_sparse_mla requires the 656-byte packed FP8 DSA KV "
            f"layout, got kv_cache_dim={kv_cache_dim}."
        )
    if q.ndim != 3 or q.shape[-1] != kv_lora_rank + qk_rope_head_dim:
        raise ValueError(
            "flashinfer_sparse_mla expects q with shape [tokens, heads, 576], "
            f"got {tuple(q.shape)}."
        )
    if q.dtype != torch.bfloat16:
        raise TypeError(f"flashinfer_sparse_mla requires BF16 queries, got {q.dtype}.")
    if kv_cache.element_size() != 1:
        raise TypeError(
            "flashinfer_sparse_mla requires a byte-addressable packed FP8 KV "
            f"cache, got dtype={kv_cache.dtype}."
        )

    num_tokens = q.shape[0]
    if indices.ndim != 2 or indices.shape[0] != num_tokens:
        raise ValueError(
            "flashinfer_sparse_mla expects indices with shape [tokens, topk], "
            f"got {tuple(indices.shape)} for {num_tokens} tokens."
        )
    if indices.dtype != torch.int32:
        raise TypeError(
            f"flashinfer_sparse_mla requires int32 physical indices, got {indices.dtype}."
        )
    if seq_lens.ndim != 1 or seq_lens.shape[0] != num_tokens:
        raise ValueError(
            "flashinfer_sparse_mla expects seq_lens with shape [tokens], "
            f"got {tuple(seq_lens.shape)} for {num_tokens} tokens."
        )
    if seq_lens.dtype != torch.int32:
        raise TypeError(
            f"flashinfer_sparse_mla requires int32 seq_lens, got {seq_lens.dtype}."
        )
    if indices.shape[1] == 0:
        raise ValueError("flashinfer_sparse_mla requires a non-empty topk dimension.")
    if workspace_buffer.dtype != torch.uint8:
        raise TypeError(
            "flashinfer_sparse_mla requires a uint8 workspace buffer, "
            f"got {workspace_buffer.dtype}."
        )

    tensors = (kv_cache, indices, seq_lens, workspace_buffer)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("flashinfer_sparse_mla inputs must be on the same device.")

    bytes_per_page = page_size * kv_cache_dim
    if kv_cache.numel() % bytes_per_page != 0:
        raise ValueError(
            f"Packed KV cache size {kv_cache.numel()} is not divisible by "
            f"page_size * kv_cache_dim ({bytes_per_page})."
        )

    if num_tokens == 0:
        return q.new_empty((0, q.shape[1], v_head_dim))

    topk = indices.shape[1]
    query = q.contiguous().unsqueeze(1)
    packed_kv = (
        kv_cache.view(torch.uint8).view(-1, page_size, kv_cache_dim).unsqueeze(1)
    )
    # FlashInfer uses -1 physical indices for inactive sparse slots and masks
    # the active prefix using seq_lens.
    block_tables = indices.contiguous().unsqueeze(1)
    output = q.new_empty((num_tokens, q.shape[1], v_head_dim))

    op = get_flashinfer_sparse_mla_op()
    result = op(
        query=query,
        kv_cache=packed_kv,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens.contiguous(),
        max_seq_len=topk,
        sparse_mla_top_k=topk,
        out=output.unsqueeze(1),
        bmm1_scale=float(sm_scale),
        bmm2_scale=1.0,
        backend="sparse",
        kv_scale_format="arbitrary_fp32",
    )
    return result.squeeze(1)
