"""aiter persistent sparse-MLA decode for ROCm (gfx950).

This module owns the *calling convention* for aiter's persistent sparse MLA
decode pair -- ``aiter::mla_a8w8_qh16_qseqlen1_gqaratio16_ps`` plus
``kn_mla_reduce_v1_ps`` -- and nothing else. The kernels themselves live in
aiter; what is hard to get right is how they are fed.

The argument set below is a direct port of ATOM, which is the only caller of
this kernel pair known to run correctly and at speed on GLM-5.2 / MI355X:

  * ``atom/model_ops/attentions/aiter_mla.py``
        ``AiterMLAMetadataBuilder.set_mla_persistent_worker_buffers``
  * ``atom/model_ops/attention_mla.py``
        ``MLAAttention._forward_decode``

Two properties are load-bearing and easy to lose:

1. **The metadata is built once per forward batch, not once per layer.**
   ``get_mla_metadata_v1`` depends only on the batch's KV layout, which every
   attention layer shares. Rebuilding it inside the per-layer attention call
   costs ~5 extra kernel launches per layer (~22 us), which on a 92-layer model
   is ~2 ms per decode iteration -- more than the kernel saves.

2. **The metadata arguments must match the sparse KV layout.** DSA packs the
   selected KV per query token at ``page_size=1``, so:

   - ``kv_indptr`` must be the *sparse* (top-k) indptr, not the dense one.
     With the dense indptr the asm kernel's ``kv_end`` runs past the written
     region of the sparse index buffer once the context exceeds ``index_topk``,
     giving an illegal KV-cache access.
   - ``kv_last_page_lens`` must be all ones (one token per page). The dense
     per-block value (1..block_size) makes ``get_mla_metadata_v1`` compute a KV
     extent up to ``block_size - 1`` pages past the written sparse region.
   - both must be sliced to ``[: bs + 1]`` / ``[: bs]``. aiter derives its
     batch count from the *length* of ``kv_indptr``
     (``get_mla_metadata_v1_0_device``: ``num_batches =
     seqlens_kv_indptr.size(0) - 1``), so handing over a full-capacity buffer
     makes it emit work items for every stale slot past ``bs``, with
     ``qo_start`` running past the end of Q and O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from aiter.mla import mla_decode_fwd
from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1

__all__ = [
    "AiterSparseMLADecodeMetadata",
    "aiter_sparse_mla_out_dtype",
    "alloc_sparse_mla_decode_metadata",
    "build_sparse_mla_decode_metadata",
    "sparse_mla_decode",
]


# --- ATOM's tuned constants ------------------------------------------------
#
# Deliberately hardcoded rather than exposed: they are a matched set, and the
# combination below is the one that has been measured end to end. Changing one
# in isolation (notably fast_mode) silently selects a different metadata kernel
# with different semantics -- see _METADATA_KERNEL_NOTE.
#
# _METADATA_KERNEL_NOTE: get_mla_metadata_v1 dispatches on (fast_mode,
# intra_batch_mode):
#     fast_mode=True                 -> v1_2_device, honors topk and page_size
#     fast_mode=False, intra=True    -> v1_0_device, IGNORES topk
#     fast_mode=False, intra=False   -> v1_1_device, honors topk
# Sparse attention must not use the path that ignores topk.
_FAST_MODE = True
_IS_CAUSAL = True
_KV_GRANULARITY = 16
_MAX_SPLIT_PER_BATCH = 16
_NHEAD_KV = 1
# KV is packed one token per page for sparse decode.
_PAGE_SIZE = 1
# Passing an explicit num_kv_splits also pins aiter to persistent mode: its
# non-persistent fallback cannot honor a top-k split scheme and would return
# wrong results, so it never downgrades a caller that sets this.
_NUM_KV_SPLITS = 16


def aiter_sparse_mla_out_dtype(q_dtype: torch.dtype) -> torch.dtype:
    """Output dtype for the decode call.

    On the a8w8 path Q is fp8 while the attention output is not: aiter's reduce
    kernels (``kn_mla_reduce_v1*``, ``csrc/kernels/mla/reduce.cu``) only
    instantiate ``out_t`` for bf16 and fp16 and hard-fail on anything else. ATOM
    sidesteps this by allocating its output in model dtype rather than Q dtype;
    mapping fp8 to bf16 here is the same thing expressed locally.
    """
    return torch.bfloat16 if q_dtype.itemsize == 1 else q_dtype


@dataclass
class AiterSparseMLADecodeMetadata:
    """Persistent work buffers plus the batch shape they were built for."""

    work_meta_data: torch.Tensor
    work_indptr: torch.Tensor
    work_info_set: torch.Tensor
    reduce_indptr: torch.Tensor
    reduce_final_map: torch.Tensor
    reduce_partial_map: torch.Tensor
    # All-ones, [capacity]. One token per page; see the module docstring.
    kv_last_page_lens: torch.Tensor

    capacity: int
    max_seqlen_q: int
    num_heads_padded: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype

    def matches(
        self,
        batch_size: int,
        max_seqlen_q: int,
        num_heads_padded: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
    ) -> bool:
        return (
            batch_size <= self.capacity
            and max_seqlen_q <= self.max_seqlen_q
            and num_heads_padded == self.num_heads_padded
            and q_dtype == self.q_dtype
            and kv_dtype == self.kv_dtype
        )

    def kernel_kwargs(self) -> dict:
        return {
            "work_meta_data": self.work_meta_data,
            "work_indptr": self.work_indptr,
            "work_info_set": self.work_info_set,
            "reduce_indptr": self.reduce_indptr,
            "reduce_final_map": self.reduce_final_map,
            "reduce_partial_map": self.reduce_partial_map,
        }


def alloc_sparse_mla_decode_metadata(
    batch_size: int,
    max_seqlen_q: int,
    num_heads_padded: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    device: torch.device,
) -> AiterSparseMLADecodeMetadata:
    """Allocate the persistent work buffers for up to ``batch_size`` sequences.

    Call this once per backend (and again only if the batch capacity or a dtype
    changes), never inside CUDA graph capture: an allocation there would be
    baked into the graph.
    """
    sizes = get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_q,
        num_heads_padded,
        q_dtype,
        kv_dtype,
        is_sparse=True,
        fast_mode=_FAST_MODE,
    )
    (
        work_meta_data,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = (torch.empty(size, dtype=dtype, device=device) for size, dtype in sizes)

    return AiterSparseMLADecodeMetadata(
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        kv_last_page_lens=torch.ones(batch_size, dtype=torch.int32, device=device),
        capacity=batch_size,
        max_seqlen_q=max_seqlen_q,
        num_heads_padded=num_heads_padded,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
    )


def build_sparse_mla_decode_metadata(
    metadata: AiterSparseMLADecodeMetadata,
    cu_seqlens_q: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    bs: int,
    max_seqlen_q: int,
) -> None:
    """Fill the work buffers in place. Call ONCE per forward batch.

    ``cu_seqlens_q`` and ``sparse_kv_indptr`` must already be ``[: bs + 1]``
    views; this function does not slice them, so that a caller inside CUDA graph
    capture keeps stable tensor addresses.
    """
    get_mla_metadata_v1(
        cu_seqlens_q,
        sparse_kv_indptr,
        metadata.kv_last_page_lens[:bs],
        metadata.num_heads_padded,
        _NHEAD_KV,
        _IS_CAUSAL,
        metadata.work_meta_data,
        metadata.work_info_set,
        metadata.work_indptr,
        metadata.reduce_indptr,
        metadata.reduce_final_map,
        metadata.reduce_partial_map,
        page_size=_PAGE_SIZE,
        kv_granularity=_KV_GRANULARITY,
        max_seqlen_qo=max_seqlen_q,
        uni_seqlen_qo=max_seqlen_q,
        fast_mode=_FAST_MODE,
        max_split_per_batch=_MAX_SPLIT_PER_BATCH,
        dtype_q=metadata.q_dtype,
        dtype_kv=metadata.kv_dtype,
    )


def sparse_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    o: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    metadata: AiterSparseMLADecodeMetadata,
    bs: int,
    max_seqlen_q: int,
    sm_scale: float,
    q_scale: Optional[torch.Tensor] = None,
    kv_scale: Optional[torch.Tensor] = None,
) -> None:
    """Run the persistent sparse MLA decode, writing into ``o``.

    Args:
        q: ``[num_tokens, num_heads_padded, kv_lora_rank + qk_rope_head_dim]``.
        kv_cache: flat MLA KV cache; viewed here as
            ``[-1, 1, 1, head_dim]`` (one token per page).
        o: ``[num_tokens, num_heads_padded, kv_lora_rank]``, dtype from
            :func:`aiter_sparse_mla_out_dtype`.
        cu_seqlens_q: ``[bs + 1]`` view.
        sparse_kv_indptr: ``[bs + 1]`` view over the top-k selection.
        sparse_kv_indices: the top-k index buffer (full capacity is fine; the
            kernel only reads what ``sparse_kv_indptr`` delimits).
        metadata: buffers already filled by
            :func:`build_sparse_mla_decode_metadata` for this batch.
        q_scale / kv_scale: required when Q is fp8 -- aiter's ``asm_mla.cu``
            rejects fp8 Q with either missing.
    """
    mla_decode_fwd(
        q,
        kv_cache.view(-1, _PAGE_SIZE, 1, q.shape[-1]),
        o,
        cu_seqlens_q,
        sparse_kv_indptr,
        sparse_kv_indices,
        metadata.kv_last_page_lens[:bs],
        max_seqlen_q,
        page_size=_PAGE_SIZE,
        nhead_kv=_NHEAD_KV,
        sm_scale=sm_scale,
        num_kv_splits=_NUM_KV_SPLITS,
        q_scale=q_scale,
        kv_scale=kv_scale,
        **metadata.kernel_kwargs(),
    )
