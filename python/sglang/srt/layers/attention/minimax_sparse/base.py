# SPDX-License-Identifier: Apache-2.0
"""Common base class and factory dispatch for MiniMax sparse attention backends."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.configs.model_config import (
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    get_minimax_sparse_score_type,
    uses_minimax_m3_nvfp4_native_stack,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_spec
from sglang.srt.server_args import m3_fp8_attn_gemm_enabled
from sglang.srt.utils import is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class MiniMaxSparseBaseBackend(AttentionBackend):
    """Shared base class for MiniMax sparse attention backends.

    Direct ``MiniMaxSparseBaseBackend(runner)`` / ``MiniMaxSparseAttnBackend(runner)``
    instantiation dispatches to the concrete platform/quantization subclass while
    preserving ``isinstance(..., MiniMaxSparseAttnBackend)`` across all subclasses.
    """

    _linear_verify: bool = False

    def __new__(cls, runner: Optional[ModelRunner] = None, *args, **kwargs):
        if cls is MiniMaxSparseBaseBackend and runner is not None:
            target_cls = resolve_minimax_sparse_backend_cls(runner)
            return super().__new__(target_cls)
        return super().__new__(cls)

    def __init__(self, runner: ModelRunner):
        assert isinstance(runner.token_to_kv_pool, MiniMaxSparseKVPool)
        self.is_npu = is_npu()
        self.is_hip = is_hip()
        self.kv_pool = runner.token_to_kv_pool
        self.hisparse_coordinator = runner.hisparse_coordinator
        self.token_to_kv_pool = runner.token_to_kv_pool  # alias for TboAttnBackend
        self.req_to_token_pool = runner.req_to_token_pool  # pool obj for TboAttnBackend
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.max_kv_len = self.req_to_token_pool.row_width
        self.page_size = self.kv_pool.page_size

        self.fp8_attn_gemm = m3_fp8_attn_gemm_enabled(
            resolving_view(runner.server_args)
        )
        if self.fp8_attn_gemm:
            assert self.kv_pool.main_pool.dtype == torch.float8_e4m3fn, (
                "fp8 attn-GEMM mode requires an fp8_e4m3fn main KV pool, got "
                f"{self.kv_pool.main_pool.dtype}"
            )

        hf_config = runner.model_config.hf_config
        sparse_cfg = get_minimax_sparse_attention_config(hf_config)
        self._sparse_cfg = sparse_cfg
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.dense_layer_ids, self.sparse_layer_ids = get_minimax_sparse_layer_ids(
            sparse_cfg
        )
        self.disable_value_layer_ids: set[int] = set(
            get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
        )
        self.score_type: str = get_minimax_sparse_score_type(sparse_cfg)

        # Plain Python int so it is safe inside CUDA graphs (no .item() at graph time).
        self._max_seqlen_q: int = 1
        self._max_seqlen_k: int = 1

        # (owning ForwardBatch, cu_seqlens, seq_lens, prefix_lens, cu_seqblocks_q,
        # max_seqblock_q, all_seqblock_q). The owner is part of the key because one
        # metadata init can be followed by more than one ForwardBatch reaching the
        # layers (two-batch overlap splits into two children with different
        # extend_seq_lens); a hit requires the SAME object, not just a live cache.
        self._prefill_seqblock_meta: Optional[tuple] = None
        self._linear_verify_meta: Optional[SimpleNamespace] = None

        self.block_size_q = 1
        self.block_size_k = sparse_cfg["sparse_block_size"]
        if "sparse_init_block" in sparse_cfg:
            self.init_blocks = sparse_cfg["sparse_init_block"]
        else:
            init_tokens = sparse_cfg["sparse_init_tokens"]
            self.init_blocks = (
                init_tokens + self.block_size_k - 1
            ) // self.block_size_k
        if "sparse_local_block" in sparse_cfg:
            self.local_blocks = sparse_cfg["sparse_local_block"]
        else:
            local_tokens = sparse_cfg["sparse_local_tokens"]
            self.local_blocks = (
                local_tokens + self.block_size_k - 1
            ) // self.block_size_k + 1
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]

        if self.hisparse_coordinator is not None:
            selected_tokens = self.topk_blocks * self.block_size_k
            assert selected_tokens <= self.hisparse_coordinator.device_buffer_size, (
                f"MiniMax M3 selects {selected_tokens} sparse-attention tokens, "
                "but the HiSparse device buffer holds only "
                f"{self.hisparse_coordinator.device_buffer_size}."
            )
            self._loc_mapping = (
                self.kv_pool.main_pool.full_to_hisparse_device_index_mapping
            )
        else:
            self._loc_mapping = None

        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.dense_backend: Optional[AttentionBackend] = None
        self.indexer_cp = None

    def skips_unused_dense_metadata(self) -> bool:
        return False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _hisparse_swap_in_blocks(
        self,
        forward_batch: ForwardBatch,
        topk_idx: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        assert topk_idx.size(0) == 1
        top_k_device_locs = self.hisparse_coordinator.swap_in_selected_blocks(
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            top_k_blocks=topk_idx[0],
            layer_id=layer_id,
            sparse_block_size=self.block_size_k,
        )
        return top_k_device_locs.unsqueeze(0)

    @staticmethod
    def _is_sparse_kv_cached_by_fusion(
        forward_batch: ForwardBatch, layer_id: int
    ) -> bool:
        layer_ids = forward_batch.minimax_m3_precached_sparse_layers
        return layer_ids is not None and layer_id in layer_ids

    def _init_linear_verify_metadata(self, forward_batch: ForwardBatch):
        ndt = self.speculative_num_draft_tokens
        # GPU seq_lens are the accepted prefix lengths. A linear EAGLE
        # chain exposes one more KV token to each successive query.
        offsets = torch.arange(
            1,
            ndt + 1,
            device=forward_batch.seq_lens.device,
            dtype=forward_batch.seq_lens.dtype,
        )
        self._linear_verify_meta = SimpleNamespace(
            seq_lens=(forward_batch.seq_lens[:, None] + offsets[None, :]).reshape(-1),
            req_pool_indices=forward_batch.req_pool_indices.repeat_interleave(ndt),
        )

    def _forward_linear_verify_funnel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        out_quant: str = "",
    ):
        meta = self._linear_verify_meta
        if meta is None or meta.seq_lens.numel() != q.shape[0]:
            raise RuntimeError("Missing MiniMax-M3 linear verify metadata")
        # Keep the dense backend's original per-request metadata intact.
        # Sparse decode accepts one causal query per row; cache stores
        # still use the original flattened out_cache_loc exactly once.
        verify_batch = copy.copy(forward_batch)
        verify_batch.seq_lens = meta.seq_lens
        verify_batch.req_pool_indices = meta.req_pool_indices
        return self.forward_decode(
            q,
            k,
            v,
            layer,
            verify_batch,
            save_kv_cache,
            idx_q=idx_q,
            idx_k=idx_k,
            idx_v=idx_v,
            out_quant=out_quant,
        )

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_batch.forward_mode.is_idle():
            idx_q = kwargs.get("idx_q")
            num_idx_heads = idx_q.shape[1]
            disable_value = layer.layer_id in self.disable_value_layer_ids
            idx_out: Optional[torch.Tensor] = (
                None
                if disable_value
                else q.new_zeros(q.shape[0], num_idx_heads * self.idx_head_dim)
            )
            out = q.new_zeros(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
            return idx_out, out
        return super().forward(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        out_quant: str = "",
    ):
        if self._linear_verify and forward_batch.forward_mode.is_target_verify():
            return self._forward_linear_verify_funnel(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                idx_q=idx_q,
                idx_k=idx_k,
                idx_v=idx_v,
                out_quant=out_quant,
            )
        raise NotImplementedError


def resolve_minimax_sparse_backend_cls(
    runner: ModelRunner,
) -> type[MiniMaxSparseBaseBackend]:
    """Select the concrete MiniMax sparse attention backend subclass for ``runner``."""
    if is_npu():
        from sglang.srt.layers.attention.minimax_sparse.npu_backend import (
            MiniMaxNPUSparseAttnBackend,
        )

        return MiniMaxNPUSparseAttnBackend

    if is_hip():
        from sglang.srt.layers.attention.minimax_sparse.rocm_backend import (
            MiniMaxROCmSparseAttnBackend,
        )

        return MiniMaxROCmSparseAttnBackend

    if uses_minimax_m3_nvfp4_native_stack(runner.model_config.quantization):
        from sglang.srt.layers.attention.minimax_sparse.msa_v1_backend import (
            MiniMaxMSAV1AttnBackend,
        )

        return MiniMaxMSAV1AttnBackend

    from sglang.srt.layers.attention.minimax_sparse.bf16_backend import (
        MiniMaxBF16SparseAttnBackend,
    )

    return MiniMaxBF16SparseAttnBackend


def create_minimax_sparse_backend(runner: ModelRunner) -> MiniMaxSparseBaseBackend:
    """Instantiate the concrete MiniMax sparse attention backend for ``runner``."""
    backend_cls = resolve_minimax_sparse_backend_cls(runner)
    return backend_cls(runner)


MiniMaxSparseAttnBackend = MiniMaxSparseBaseBackend
