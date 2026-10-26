"""Replay metadata of the MTP draft-decode step backends in one bounded launch.

The legacy replay precomputes the decode metadata into a scratch table as wide
as ``--context-length`` and copies its full width into every step backend
(``fused_metadata_copy_multi`` + ``fused_metadata_copy``), so its cost grows
with batch size times the context cap. Here one live-length-bounded kernel
writes every step backend's graph metadata straight from ``req_to_token``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.ops.attention.dsa_metadata_bounded import (
    MAX_MULTI_DSTS,
    fused_dsa_decode_metadata_multi,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa_backend import (
        DeepseekSparseAttnBackend,
        DSAMetadata,
    )


def draft_decode_metadata_writable(
    backends: Sequence[DeepseekSparseAttnBackend], bs: int
) -> bool:
    """Whether every step backend can take the one-launch replay for ``bs``.

    Static per captured ``bs`` (callers may cache it until the next capture):
    bounded metadata on, not FlashMLA, ``bs`` captured by every step backend,
    and graph tables that satisfy fused_dsa_decode_metadata_multi's contract.
    """
    if not backends:
        return False
    first = backends[0]
    if (
        not first.dsa_meta_bounded
        # FlashMLA's per-replay metadata comes from the precompute path.
        or first.dsa_decode_impl == "flashmla_kv"
    ):
        return False
    metas = [backend.decode_cuda_graph_metadata.get(bs) for backend in backends]
    if any(metadata is None for metadata in metas):
        return False
    return _tables_agree(metas, first.physical_page_size)


def _tables_agree(metas: Sequence[DSAMetadata], physical_page_size: int) -> bool:
    # The launch writes every destination with the first one's strides and
    # width, and needs real_page_table (page size > 1) or page_table_1.
    required = "real_page_table" if physical_page_size > 1 else "page_table_1"
    ref = metas[0]
    for metadata in metas:
        if getattr(metadata, required) is None:
            return False
        for name in ("page_table_1", "real_page_table"):
            a, b = getattr(ref, name), getattr(metadata, name)
            if (a is None) != (b is None):
                return False
            if a is not None and (a.shape != b.shape or a.stride() != b.stride()):
                return False
    return True


def write_draft_decode_metadata(
    *,
    backends: Sequence[DeepseekSparseAttnBackend],
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    bs: int,
    schedule_refresh_steps: Sequence[int],
) -> None:
    first = backends[0]
    metas = [backend.decode_cuda_graph_metadata[bs] for backend in backends]
    for start in range(0, len(metas), MAX_MULTI_DSTS):
        fused_dsa_decode_metadata_multi(
            seq_lens=seq_lens[:bs],
            req_pool_indices=req_pool_indices[:bs],
            req_to_token=first.req_to_token,
            dsts=[_decode_dst(m) for m in metas[start : start + MAX_MULTI_DSTS]],
            bs=bs,
            max_len=first._graph_page_table_width(metas[0]),
            dsa_index_topk=first.dsa_index_topk,
            physical_page_size=first.physical_page_size,
            program_budget=first.dsa_meta_program_budget,
            max_splits_per_row=first.dsa_meta_max_splits,
        )
    for step, (backend, metadata) in enumerate(zip(backends, metas, strict=True)):
        backend.set_dsa_prefill_impl(forward_batch=None)
        if step in schedule_refresh_steps:
            _refresh_decode_schedule(backend, metadata)
        backend.forward_metadata = metadata


def _decode_dst(metadata: DSAMetadata) -> tuple:
    return (
        metadata.cache_seqlens_int32,
        metadata.cu_seqlens_k,
        metadata.page_table_1,
        metadata.dsa_cache_seqlens_int32,
        metadata.dsa_cu_seqlens_k,
        metadata.real_page_table,
    )


def _refresh_decode_schedule(
    backend: DeepseekSparseAttnBackend, metadata: DSAMetadata
) -> None:
    # Same refresh as init_forward_metadata_replay_cuda_graph_from_precomputed
    # (DECODE): (N, 1) context lens, DeepGEMM schedule, top-k v2 plan.
    seqlens_32_2d = metadata.cache_seqlens_int32.contiguous().view(-1, 1)
    backend._refresh_paged_mqa_schedule_metadata(metadata, seqlens_32_2d)
    backend._refresh_topk_v2_plan(metadata)
    if metadata.paged_mqa_ctx_lens_2d is None:
        object.__setattr__(metadata, "paged_mqa_ctx_lens_2d", seqlens_32_2d)
    else:
        metadata.paged_mqa_ctx_lens_2d.copy_(seqlens_32_2d)
