# SPDX-License-Identifier: Apache-2.0
"""ROCm MiniMax sparse attention backend (packed NVFP4 KV + grouped verify + top-k reuse)."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import Nvfp4Cache
from sglang.srt.environ import envs
from sglang.srt.layers.attention.minimax_sparse.bf16_backend import (
    MiniMaxBF16SparseAttnBackend,
)
from sglang.srt.layers.moe.utils import is_tbo_enabled
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_gfx95_supported


def init_sparse_backend(backend, spec) -> None:
    """The ROCm state of MiniMaxROCmSparseAttnBackend.__init__."""
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


class MiniMaxROCmSparseAttnBackend(MiniMaxBF16SparseAttnBackend):
    """ROCm sparse attention backend extending the Triton pipeline with gfx950 optimizations."""

    def _init_platform_extensions(self, spec) -> None:
        init_sparse_backend(self, spec)
        if spec.speculative_algorithm is not None and spec.speculative_eagle_topk != 1:
            raise NotImplementedError(
                "MiniMax-M3 ROCm speculative attention requires a linear "
                "draft chain (--speculative-eagle-topk 1)."
            )

        self.index_topk_freq = (
            max(int(envs.SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ.get()), 1)
            if is_gfx95_supported() and not is_tbo_enabled()
            else 1
        )
        self.index_cache_enabled = self.index_topk_freq > 1
        # topk_index_reduce widens the last dim to idx_group_size * topk_blocks
        # (union of the group's selections), so the shared decode buffer must be
        # that wide. Head split mirrors MiniMaxM3 sparse attention's.
        self._idx_group_size = 1
        if self.index_cache_enabled:
            _num_idx_heads = max(
                self._sparse_cfg["sparse_num_index_heads"]
                // get_parallel().attn_tp_size,
                1,
            )
            self._idx_group_size = max(
                _num_idx_heads // self.kv_pool.main_pool.head_num, 1
            )
        # Persistent per-bs device buffer for decode top-k reuse. Allocated eagerly
        # outside CUDA-graph capture; the captured graph only copies into and reads
        # from a fixed address.
        self._decode_topk_buf: dict = {}
        self._topk_group_of_layer: dict[int, int] = {}
        self._topk_is_source: dict[int, bool] = {}
        for ordinal, lid in enumerate(
            lid for lid in self.sparse_layer_ids if lid in self.disable_value_layer_ids
        ):
            group = ordinal // self.index_topk_freq
            self._topk_group_of_layer[lid] = group
            self._topk_is_source[lid] = (ordinal % self.index_topk_freq) == 0
        self._topk_cache: dict = {}
        self._topk_cache_owner: Optional[ForwardBatch] = None

    def skips_unused_dense_metadata(self) -> bool:
        return dense_metadata_unused(self)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # New forward -> drop the per-forward index-cache top-k (prefill only).
        if self.index_cache_enabled:
            self._topk_cache = {}
            self._topk_cache_owner = None
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        # Decode top-k reuse: pre-allocate the per-bs persistent buffer so graph
        # capture never allocates. num_kv_heads == 1 at TP>=4 for M3.
        if self.index_cache_enabled and (
            forward_batch.forward_mode.is_decode_or_idle()
            or (self._linear_verify and forward_batch.forward_mode.is_target_verify())
        ):
            bs = forward_batch.seq_lens.shape[0]
            if forward_batch.forward_mode.is_target_verify():
                bs *= self.speculative_num_draft_tokens
            if bs > 0 and bs not in self._decode_topk_buf:
                _nkv = self.kv_pool.main_pool.head_num
                self._decode_topk_buf[bs] = torch.empty(
                    (_nkv, bs, self.topk_blocks * self._idx_group_size),
                    dtype=torch.int32,
                    device=forward_batch.seq_lens.device,
                )

    def _should_skip_extend_kv_store(
        self, forward_batch: ForwardBatch, layer_id: int, save_kv_cache: bool
    ) -> bool:
        # ROCm: save_kv_cache=False means the fused QK-norm/RoPE store already wrote K/V
        return (not save_kv_cache) or super()._should_skip_extend_kv_store(
            forward_batch, layer_id, save_kv_cache
        )

    def _get_layer_caches(self, layer_id: int, disable_value: bool):
        k_cache, v_cache, idx_k_cache, idx_v_cache = super()._get_layer_caches(
            layer_id, disable_value
        )
        if self.kv_is_nvfp4:
            k_cache, v_cache, idx_k_cache = nvfp4_layer_caches(
                self.kv_pool, layer_id, k_cache, v_cache, idx_k_cache
            )
        return k_cache, v_cache, idx_k_cache, idx_v_cache

    def _prepare_prefill_topk_reuse(
        self, forward_batch: ForwardBatch, layer_id: int, disable_value: bool
    ) -> tuple[Optional[torch.Tensor], bool, Optional[int]]:
        # Index cache: only for disable_value layers (idx_o is None,
        # so skipping the indexer has no output side effect). A group's source
        # layer computes + stores the reduced top-k; the other layers reuse it.
        if not (self.index_cache_enabled and disable_value):
            return None, False, None
        if self._topk_cache_owner is not forward_batch:
            self._topk_cache = {}
            self._topk_cache_owner = forward_batch
        group = self._topk_group_of_layer[layer_id]
        if self._topk_is_source[layer_id]:
            return None, True, group
        # Miss (e.g. source layer chunked differently) -> recompute safely.
        return self._topk_cache.get(group), False, group

    def _store_prefill_topk_reuse(
        self, group: Optional[int], reduced_topk_idx: torch.Tensor
    ) -> None:
        if group is not None:
            self._topk_cache[group] = reduced_topk_idx

    def _prepare_decode_topk_reuse(
        self,
        num_q_rows: int,
        layer_id: int,
        disable_value: bool,
        has_dense_attn_fn: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        _use_reuse = (
            self.index_cache_enabled and disable_value and not has_dense_attn_fn
        )
        if not _use_reuse:
            return None, None
        _topk_buf = self._decode_topk_buf.get(num_q_rows)
        if _topk_buf is None:
            return None, None
        if self._topk_is_source.get(layer_id, True):
            return None, _topk_buf
        return _topk_buf, None

    def _verify_num_draft(self, forward_batch: ForwardBatch) -> Optional[int]:
        return verify_num_draft(self, forward_batch)
