"""Stable-address staging buffers for NSA indexer metadata under PCG.

Inside the compiled / CUDA-graph-captured region the indexer cannot reach into
the Python `BaseIndexerMetadata` object (method dispatch + non-tensor attribute
walks are not torch.compile-traceable, and the metadata's tensor fields are
freshly allocated each step so their addresses are not stable across replays).

This module provides:
 - `PrefillNsaMetadataBuffers`: a dataclass of pre-allocated, stable-address
   device tensors sized to PCG upper bounds. `mark_static_address` is applied
   so Dynamo treats them as graph inputs.
 - `allocate_prefill_nsa_metadata_buffers(...)`: allocator (GPU, int32, zeros).
 - `stage_prefill_nsa_metadata_buffers(...)`: per-step host-side copy from the
   live metadata into the staging slots; padded entries are zeroed so stale
   data from a previous step never leaks into the current capture/replay.

Slots are added incrementally as each piece of `_get_topk_ragged` is migrated
into the compiled region (component C4 of the prefill PCG roadmap).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata


@dataclass
class PrefillNsaMetadataBuffers:
    """Pre-allocated, stable-address NSA indexer metadata staging buffers.

    All slots are sized to PCG upper bounds; the per-step usable region is
    `[:bs]` (or `[:num_tokens]` for per-token slots). Padded entries are zeroed
    in `stage_prefill_nsa_metadata_buffers` to avoid stale-data leaks.
    """

    # (max_bs,) int32 — mirror of `metadata.get_indexer_seq_len()`.
    indexer_seq_len: torch.Tensor

    # (max_num_tokens,) int32 — mirrors of `metadata.get_indexer_kvcache_range()`.
    ks: torch.Tensor
    ke: torch.Tensor

    # (max_num_tokens,) int32 — mirror of `metadata.get_seqlens_expanded()`.
    seqlens_expanded: torch.Tensor

    # (max_num_tokens,) int32 — mirror of `metadata.get_token_to_batch_idx()`.
    token_to_batch_idx: torch.Tensor

    # (max_bs+1,) int32 — mirror of `metadata.attn_metadata.cu_seqlens_q`.
    cu_seqlens_q: torch.Tensor

    # (max_num_tokens,) int32 — mirror of `metadata.attn_metadata.topk_indices_offset`.
    # Only populated under RAGGED topk_transform_method; left zero for PAGED.
    topk_indices_offset: torch.Tensor

    # (max_bs, max_blocks_64) int32 — mirror of `metadata.get_page_table_64()` /
    # `metadata.attn_metadata.real_page_table`. max_blocks_64 = ceil(max_seq_len/64)+1.
    page_table_64: torch.Tensor

    # (max_bs, max_seq_len) int32 — mirror of `metadata.attn_metadata.page_table_1`.
    page_table_1: torch.Tensor

    # `TopkTransformMethod` enum captured on the first `stage_*` call; pinned
    # thereafter (drift across captures triggers an assertion). Read inside
    # the compiled region as a Dynamo-specialized Python int.
    topk_transform_method: Optional[Any] = None


def _alloc_static(shape, device: torch.device) -> torch.Tensor:
    t = torch.zeros(shape, dtype=torch.int32, device=device)
    torch._dynamo.mark_static_address(t)
    return t


def allocate_prefill_nsa_metadata_buffers(
    max_bs: int,
    max_num_tokens: int,
    max_seq_len: int,
    device: torch.device,
) -> PrefillNsaMetadataBuffers:
    max_blocks_64 = (max_seq_len + 63) // 64 + 1
    return PrefillNsaMetadataBuffers(
        indexer_seq_len=_alloc_static((max_bs,), device),
        ks=_alloc_static((max_num_tokens,), device),
        ke=_alloc_static((max_num_tokens,), device),
        seqlens_expanded=_alloc_static((max_num_tokens,), device),
        token_to_batch_idx=_alloc_static((max_num_tokens,), device),
        cu_seqlens_q=_alloc_static((max_bs + 1,), device),
        topk_indices_offset=_alloc_static((max_num_tokens,), device),
        page_table_64=_alloc_static((max_bs, max_blocks_64), device),
        page_table_1=_alloc_static((max_bs, max_seq_len), device),
    )


def _assert_compatible_dtype(slot: torch.Tensor, src: torch.Tensor, name: str) -> None:
    """Reject mismatches that lose information; allow int<->int casts.

    All staging slots are int32 (sequence lengths and slot indices, well within
    int32 range). Real metadata may store these as int64 (`forward_batch.seq_lens`),
    so we permit int64->int32 via `tensor.copy_`'s built-in conversion. Float
    sources are rejected (would silently truncate)."""
    if slot.dtype.is_floating_point or src.dtype.is_floating_point:
        assert src.dtype == slot.dtype, (
            f"{name} dtype mismatch (float requires exact match): "
            f"src={src.dtype}, slot={slot.dtype}"
        )


def _stage_1d(slot: torch.Tensor, src: torch.Tensor, n: int, *, name: str) -> None:
    """Copy `src` (shape `(n,)`) into `slot[:n]`, zero `slot[n:]`."""
    _assert_compatible_dtype(slot, src, name)
    assert src.shape == (n,), (
        f"{name} shape mismatch: src={tuple(src.shape)}, expected ({n},)"
    )
    assert n <= slot.shape[0], (
        f"{name}: n={n} exceeds staging capacity {slot.shape[0]}"
    )
    slot[:n].copy_(src, non_blocking=True)
    if n < slot.shape[0]:
        slot[n:].zero_()


def _stage_2d(
    slot: torch.Tensor, src: torch.Tensor, n0: int, n1: int, *, name: str
) -> None:
    """Copy `src` (shape `(n0, n1)`) into `slot[:n0, :n1]`. Zero everything else."""
    _assert_compatible_dtype(slot, src, name)
    assert src.shape == (n0, n1), (
        f"{name} shape mismatch: src={tuple(src.shape)}, expected ({n0}, {n1})"
    )
    assert n0 <= slot.shape[0] and n1 <= slot.shape[1], (
        f"{name}: ({n0}, {n1}) exceeds staging capacity {tuple(slot.shape)}"
    )
    # Wipe the whole slot first to clear stale state outside the (n0, n1) tile.
    slot.zero_()
    slot[:n0, :n1].copy_(src, non_blocking=True)


def stage_prefill_nsa_metadata_buffers(
    buffers: PrefillNsaMetadataBuffers,
    metadata: BaseIndexerMetadata,
    bs: int,
    num_tokens: int,
) -> None:
    """Per-step copy of live metadata into stable-address staging slots."""
    _stage_1d(
        buffers.indexer_seq_len,
        metadata.get_indexer_seq_len(),
        bs,
        name="indexer_seq_len",
    )

    ks_src, ke_src = metadata.get_indexer_kvcache_range()
    _stage_1d(buffers.ks, ks_src, num_tokens, name="ks")
    _stage_1d(buffers.ke, ke_src, num_tokens, name="ke")

    _stage_1d(
        buffers.seqlens_expanded,
        metadata.get_seqlens_expanded(),
        num_tokens,
        name="seqlens_expanded",
    )
    _stage_1d(
        buffers.token_to_batch_idx,
        metadata.get_token_to_batch_idx(),
        num_tokens,
        name="token_to_batch_idx",
    )

    cu_seqlens_q_src = metadata.attn_metadata.cu_seqlens_q
    _stage_1d(
        buffers.cu_seqlens_q, cu_seqlens_q_src, bs + 1, name="cu_seqlens_q"
    )

    # topk_indices_offset is None for PAGED topk_transform_method; leave zeros.
    topk_offset_src = metadata.attn_metadata.topk_indices_offset
    if topk_offset_src is not None:
        _stage_1d(
            buffers.topk_indices_offset,
            topk_offset_src,
            num_tokens,
            name="topk_indices_offset",
        )
    else:
        buffers.topk_indices_offset.zero_()

    pt64_src = metadata.get_page_table_64()
    _stage_2d(
        buffers.page_table_64,
        pt64_src,
        pt64_src.shape[0],
        pt64_src.shape[1],
        name="page_table_64",
    )

    pt1_src = metadata.attn_metadata.page_table_1
    _stage_2d(
        buffers.page_table_1,
        pt1_src,
        pt1_src.shape[0],
        pt1_src.shape[1],
        name="page_table_1",
    )

    # Pin topk_transform_method on first stage; assert stability afterwards.
    method = getattr(metadata, "topk_transform_method", None)
    if buffers.topk_transform_method is None:
        buffers.topk_transform_method = method
    else:
        assert buffers.topk_transform_method == method, (
            f"topk_transform_method drifted: pinned={buffers.topk_transform_method}, "
            f"observed={method}. PCG requires a fixed topk method per capture."
        )
