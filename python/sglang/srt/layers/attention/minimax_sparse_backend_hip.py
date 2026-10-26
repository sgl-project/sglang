"""ROCm glue of minimax_sparse_backend: the branches MiniMaxSparseAttnBackend and
MiniMaxHybridAttnBackend take under is_hip, bound there as _hip. On ROCm the KV pool may
be the packed NVFP4 one, and target verify scores each request's drafts with the grouped
indexer."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import Nvfp4Cache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_gfx95_supported


def init_sparse_backend(backend, spec) -> None:
    """The ROCm state of MiniMaxSparseAttnBackend.__init__."""
    from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool

    backend.kv_is_nvfp4 = isinstance(backend.kv_pool, MiniMaxM3Nvfp4KVPool)
    # ROCm verify scores each draft row as a decode row; the grouped kernel scores max
    # only, and was measured on gfx950 only.
    backend._use_grouped_verify_indexer = (
        is_gfx95_supported()
        and spec.speculative_algorithm is not None
        and backend.score_type == "max"
    )


def nvfp4_layer_caches(
    pool, layer_id: int, k_cache, v_cache, idx_k_cache
) -> tuple[Nvfp4Cache, Nvfp4Cache, Nvfp4Cache]:
    """One layer's main K/V and index K payloads of the packed NVFP4 pool, paired with
    their block scales as the kernels take them."""
    return (
        Nvfp4Cache(
            payload=k_cache,
            scales=pool.get_main_k_scale_buffer(layer_id).view(torch.uint8),
        ),
        Nvfp4Cache(
            payload=v_cache,
            scales=pool.get_main_v_scale_buffer(layer_id).view(torch.uint8),
        ),
        Nvfp4Cache(
            payload=idx_k_cache,
            scales=pool.get_index_k_scale_buffer(layer_id).view(torch.uint8),
        ),
    )


def verify_num_draft(backend, forward_batch: ForwardBatch) -> Optional[int]:
    """Drafts per request for the grouped verify indexer, which minimax_sparse_decode
    runs only for K-only layers off the dense-main path."""
    # _init_rocm_linear_verify_metadata lays each request's drafts out as consecutive rows
    if (
        backend._use_grouped_verify_indexer
        and forward_batch.forward_mode.is_target_verify()
    ):
        return backend.speculative_num_draft_tokens
    return None


def dense_metadata_unused(sparse_backend) -> bool:
    # With every layer sparse (M3.1) the dense backend serves no layer, so its
    # per-forward metadata is dead work; validated on gfx950 only.
    return is_gfx95_supported() and not sparse_backend.dense_layer_ids
