# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU MiniMax sparse attention backend (BNSD Triton + native AscendC/FIA)."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.minimax_sparse.base import MiniMaxSparseBaseBackend
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Adaptive block_size_q thresholds (cut K-cache traffic; affects only the serial loop).
_BSQ_THRESHOLD_64 = 4096  # max_seqlen_k >= 4K  -> BSQ=64
_BSQ_THRESHOLD_32 = 1024  # max_seqlen_k >= 1K  -> BSQ=32
_BSQ_THRESHOLD_16 = 512  # max_seqlen_k >= 512 -> BSQ=16
# BSQ<=64 is UB-safe for the prefill indexer (Q tile up to 8KB at BSQ=64).


def _native_indexer_enabled() -> bool:
    # Native AscendC packed indexer switch (default off).
    return envs.SGLANG_MINIMAX_NPU_NATIVE_INDEXER.get()


def _native_attn_enabled() -> bool:
    # Native AscendC sparse MAIN-attention switch (default off).
    return envs.SGLANG_MINIMAX_NPU_NATIVE_ATTN.get()


def _kv_cache_to_bnsd(
    k_cache: torch.Tensor, v_cache: torch.Tensor, page_size: int
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Reshape NHD slot-major KV caches to BNSD [pages, page_size, heads, dim].

    Already-paged 4D inputs pass through unchanged.
    """
    if k_cache.dim() == 4:
        num_pages, _, num_kv_heads, head_dim = k_cache.shape
        return k_cache, v_cache, num_pages, num_kv_heads, head_dim
    num_pages = k_cache.shape[0] // page_size
    num_kv_heads = k_cache.shape[1]
    head_dim = k_cache.shape[2]
    return (
        k_cache.view(num_pages, page_size, num_kv_heads, head_dim),
        v_cache.view(num_pages, page_size, num_kv_heads, head_dim),
        num_pages,
        num_kv_heads,
        head_dim,
    )


def _idx_cache_to_bnsd(
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
    page_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    """Reshape NHD slot-major index caches to BNSD; already-paged 4D passes through."""
    if idx_k_cache.dim() == 4:
        return idx_k_cache, idx_v_cache, idx_k_cache.shape[2], idx_k_cache.shape[3]
    num_pages = idx_k_cache.shape[0] // page_size
    idx_kv_heads = idx_k_cache.shape[1]
    idx_dim = idx_k_cache.shape[2]
    idx_v_bnsd = (
        None
        if idx_v_cache is None
        else idx_v_cache.view(num_pages, page_size, idx_kv_heads, idx_dim)
    )
    return (
        idx_k_cache.view(num_pages, page_size, idx_kv_heads, idx_dim),
        idx_v_bnsd,
        idx_kv_heads,
        idx_dim,
    )


class MiniMaxNPUSparseAttnBackend(MiniMaxSparseBaseBackend):
    """Ascend NPU sparse attention backend."""

    def __init__(self, runner: ModelRunner):
        super().__init__(runner)
        self.use_msa = False
        self._use_msa_decode = False
        self._msa_owns_decode = False
        self.use_dense_sparse_decode = False
        self._linear_verify = False
        self.kv_is_nvfp4 = False
        self.index_topk_freq = 1
        self.index_cache_enabled = False

        # Per-forward cache for the native decode block table (rebuilt each forward).
        self._native_decode_bt: dict = {}
        # NPU: per-forward cached metadata for the triton paths (rebuilt each forward).
        self._prefill_meta: Optional[SimpleNamespace] = None
        self._extend_meta: Optional[SimpleNamespace] = None
        self._extend_meta_key: Optional[int] = None
        self._decode_seq_lens_i32_cg: dict[int, torch.Tensor] = {}
        self._verify_meta_cg: dict[tuple, SimpleNamespace] = {}

        # Prime the native sparse op probe before cuda-graph capture.
        from sgl_kernel_npu.operators.attention.flash_decode_bnsd_with_gqa_share_sparse import (
            _get_native_sparse_op,
        )

        self._native_sparse_ok = _get_native_sparse_op() is not None
        _decode_cuda_graph = not check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        )

        from sglang.srt.layers.attention.minimax_sparse_ops.indexer_cp import (
            make_indexer_cp,
        )

        self.indexer_cp = make_indexer_cp(self, runner, self._sparse_cfg)

        logger.info(
            f"[MiniMaxSparse] Backend initialized "
            f"(score_type={self.score_type!r}, "
            f"main_attn=triton, "
            f"index_topk_freq={self.index_topk_freq}, "
            f"msa_decode=False, "
            f"msa_owns_decode=False, "
            f"decode_cuda_graph={_decode_cuda_graph}, "
            f"fp8_attn_gemm={self.fp8_attn_gemm}, "
            f"hisparse={'enabled' if self._loc_mapping is not None else 'disabled'}, "
            f"npu_native_attn={'on' if (self._native_sparse_ok and _native_attn_enabled()) else 'off'}, "
            f"disable_value_layers={sorted(self.disable_value_layer_ids)})"
        )

    @staticmethod
    def _choose_decode_score_max_chunks(batch_size: int) -> int:
        """Score chunk count per graph bucket.

        bs=1 uses 16 chunks; larger buckets keep 32. Verify has its own tuning.
        """
        return 16 if int(batch_size) == 1 else 32

    @staticmethod
    def _choose_block_size_q(max_seqlen_k: int) -> int:
        """Pick block_size_q from max KV length (MINIMAX_NPU_PREFILL_BSQ overrides)."""
        _forced = os.environ.get("MINIMAX_NPU_PREFILL_BSQ")
        if _forced:
            try:
                _v = int(_forced)
                if _v > 0:
                    return _v
            except ValueError:
                pass
        if max_seqlen_k >= _BSQ_THRESHOLD_64:
            return 64
        if max_seqlen_k >= _BSQ_THRESHOLD_32:
            return 32
        if max_seqlen_k >= _BSQ_THRESHOLD_16:
            return 16
        return 1

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self._prefill_seqblock_meta = None
        # Invalidate cached prefill/extend metadata; rebuilt on first sparse layer.
        self._prefill_meta = None
        self._extend_meta = None
        self._extend_meta_key = None

        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is not None:
            self._max_seqlen_q = int(max(extend_lens))
        else:
            self._max_seqlen_q = 1

        if in_capture and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
        ):
            # Capture uses tiny dummy seq_lens; bound by full context so replay
            # (longer sequences) does not miss KV blocks.
            self._max_seqlen_k = self.max_kv_len
        else:
            self._max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())

        # ---- REPLAY-FRESH native verify block_table ----
        if (
            forward_batch.forward_mode.is_target_verify()
            and self.speculative_num_draft_tokens
        ):
            _ndt = self.speculative_num_draft_tokens
            _bs = forward_batch.seq_lens.shape[0]
            _key = (_bs, int(_ndt))
            _vmeta = self._verify_meta_cg.get(_key)
            if _vmeta is not None:
                _vmeta.per_query_req.copy_(
                    forward_batch.req_pool_indices.long().repeat_interleave(int(_ndt))
                )
                _prefix = (forward_batch.seq_lens.to(torch.long) - int(_ndt)).clamp(
                    min=0
                )
                _offs = torch.arange(
                    1,
                    int(_ndt) + 1,
                    device=forward_batch.seq_lens.device,
                    dtype=torch.long,
                )
                _vmeta.per_query_seq_lens.copy_(
                    (_prefix.unsqueeze(1) + _offs.unsqueeze(0))
                    .reshape(-1)
                    .to(torch.int32)
                )
                _mb = self.max_kv_len // self.page_size
                _bt_cols = (
                    torch.arange(
                        _mb, device=_vmeta.per_query_req.device, dtype=torch.long
                    )
                    * self.page_size
                ).clamp(max=self.max_kv_len - 1)
                _vmeta.native_bt = (
                    self.req_to_token[_vmeta.per_query_req][:, _bt_cols]
                    // self.page_size
                ).to(torch.int32)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        # Layer-invariant decode/verify metadata as captured ops (re-read at replay).
        fm = forward_batch.forward_mode
        if fm.is_target_verify():
            ndt = self.speculative_num_draft_tokens
            if ndt:
                prefix = (forward_batch.seq_lens.to(torch.long) - int(ndt)).clamp(min=0)
                offsets = torch.arange(
                    1,
                    int(ndt) + 1,
                    device=forward_batch.seq_lens.device,
                    dtype=torch.long,
                )
                per_query_seq_lens = (
                    (prefix.unsqueeze(1) + offsets.unsqueeze(0))
                    .reshape(-1)
                    .to(torch.int32)
                )
                per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
                    int(ndt)
                )
                # Captured block_table for the native verify op (re-runs at replay).
                _mb = self.max_kv_len // self.page_size
                _bt_cols = (
                    torch.arange(_mb, device=per_query_req.device, dtype=torch.long)
                    * self.page_size
                ).clamp(max=self.max_kv_len - 1)
                _native_bt = (
                    self.req_to_token[per_query_req][:, _bt_cols] // self.page_size
                ).to(torch.int32)
                self._verify_meta_cg[(forward_batch.seq_lens.shape[0], int(ndt))] = (
                    SimpleNamespace(
                        per_query_seq_lens=per_query_seq_lens,
                        per_query_req=per_query_req,
                        native_bt=_native_bt,
                    )
                )
        elif fm.is_decode_or_idle():
            self._decode_seq_lens_i32_cg[forward_batch.seq_lens.shape[0]] = (
                forward_batch.seq_lens.to(torch.int32)
            )

    def _merge_sparse_blocks(
        self,
        topk_blocks: torch.Tensor,
        query_positions: torch.Tensor,
        num_blocks: int,
    ) -> torch.Tensor:
        """Append forced init/local blocks to top-k block ids and deduplicate."""
        total = self.topk_blocks + self.init_blocks + self.local_blocks
        if self.init_blocks <= 0 and self.local_blocks <= 0:
            return topk_blocks

        block_size = self.block_size_k
        q_len = query_positions.shape[0]
        num_idx_heads = topk_blocks.shape[1]
        qcol = query_positions[:, None, None]

        if self.init_blocks == 0 and self.local_blocks == 1:
            local = (query_positions // block_size).clamp(
                min=0, max=max(num_blocks - 1, 0)
            )
            local = (
                local.to(topk_blocks.dtype)
                .view(q_len, 1, 1)
                .expand(-1, num_idx_heads, -1)
            )
            valid_topk = (topk_blocks >= 0) & (topk_blocks < num_blocks)
            valid_topk = valid_topk & (topk_blocks * block_size <= qcol)
            local_duplicate = ((topk_blocks == local) & valid_topk).any(
                dim=-1, keepdim=True
            )
            valid_local = (local >= 0) & (local < num_blocks)
            valid_local = valid_local & (local * block_size <= qcol) & ~local_duplicate
            return torch.cat(
                [
                    torch.where(
                        valid_topk, topk_blocks, torch.full_like(topk_blocks, -1)
                    ),
                    torch.where(valid_local, local, torch.full_like(local, -1)),
                ],
                dim=-1,
            )

        forced_parts = []
        if self.init_blocks > 0:
            forced_parts.append(
                torch.arange(
                    self.init_blocks,
                    device=topk_blocks.device,
                    dtype=topk_blocks.dtype,
                )
                .view(1, 1, -1)
                .expand(q_len, num_idx_heads, -1)
            )
        if self.local_blocks > 0:
            offsets = torch.arange(
                self.local_blocks,
                device=topk_blocks.device,
                dtype=query_positions.dtype,
            )
            block_ids = query_positions // block_size
            first = (block_ids - self.local_blocks + 1).clamp(min=0)
            forced_parts.append(
                (first[:, None] + offsets[None, :])
                .to(topk_blocks.dtype)
                .view(q_len, 1, -1)
                .expand(-1, num_idx_heads, -1)
            )

        forced = torch.cat(forced_parts, dim=-1)
        candidates = torch.cat([forced, topk_blocks], dim=-1)
        valid = (candidates >= 0) & (candidates < num_blocks)
        valid = valid & (candidates * block_size <= qcol)
        invalid_value = torch.full_like(candidates, num_blocks)
        sorted_candidates = torch.sort(
            torch.where(valid, candidates, invalid_value), dim=-1
        ).values
        sorted_valid = sorted_candidates < num_blocks
        previous = torch.cat(
            [
                torch.full_like(sorted_candidates[..., :1], -1),
                sorted_candidates[:, :, :-1],
            ],
            dim=-1,
        )
        keep = sorted_valid & (sorted_candidates != previous)
        ranks = torch.cumsum(keep.to(torch.int32), dim=-1) - 1
        output = torch.full(
            (q_len, num_idx_heads, total + 1),
            -1,
            dtype=topk_blocks.dtype,
            device=topk_blocks.device,
        )
        overflow_rank = torch.full_like(ranks, total)
        scatter_index = torch.where(keep & (ranks < total), ranks, overflow_rank).long()
        scatter_src = torch.where(keep, sorted_candidates, -1)
        output.scatter_(2, scatter_index, scatter_src)
        return output[:, :, :total]

    def _prepare_npu_triton_topk_idx(
        self,
        topk_idx: torch.Tensor,
        seq_lens: torch.Tensor,
        num_idx_heads: int,
        num_kv_heads: int,
        max_blocks: int,
    ) -> torch.Tensor:
        """Prepare NPU triton top-k ids in the GQA kernel layout."""
        if (
            self.init_blocks == 0
            and self.local_blocks == 1
            and num_idx_heads == num_kv_heads
            and topk_idx.shape[0] == num_kv_heads
            and topk_idx.dtype == torch.int32
            and topk_idx.is_contiguous()
            and seq_lens.is_contiguous()
        ):
            if topk_idx.shape[2] == self.topk_blocks + 1:
                return topk_idx
            from sgl_kernel_npu.operators.indexer import (
                append_local_block_to_topk_idx,
            )

            return append_local_block_to_topk_idx(
                topk_idx, seq_lens, self.block_size_k, max_blocks
            )

        if num_idx_heads > num_kv_heads:
            from sglang.kernels.ops.attention.minimax_sparse.common.index import (
                topk_index_reduce,
            )

            idx_group_size = num_idx_heads // num_kv_heads
            topk_idx = topk_index_reduce(
                topk_idx.view(num_kv_heads, idx_group_size, -1, self.topk_blocks),
                dim=1,
            )

        topk_2d = topk_idx.permute(1, 0, 2).contiguous()
        query_positions = (seq_lens.to(torch.long) - 1).clamp(min=0)
        topk_merged = self._merge_sparse_blocks(topk_2d, query_positions, max_blocks)
        return topk_merged.permute(1, 0, 2).contiguous()

    def _build_native_block_table(
        self, req_indices: torch.Tensor, max_blocks: int, device
    ) -> torch.Tensor:
        """Logical->physical page table for the native sparse main op."""
        blk_cols = (
            torch.arange(max_blocks, device=device, dtype=torch.long) * self.page_size
        ).clamp(max=self.max_kv_len - 1)
        return (self.req_to_token[req_indices][:, blk_cols] // self.page_size).to(
            torch.int32
        )

    def _forward_npu_triton_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ):
        """NPU decode via the ported triton kernels (BNSD paged)."""
        from sgl_kernel_npu.operators.attention import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )
        from sgl_kernel_npu.operators.indexer import (
            flash_decode_bnsd_with_topk_idx,
        )

        page_size = self.page_size
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]

        k_bnsd, v_bnsd, num_pages, num_kv_heads, head_dim = _kv_cache_to_bnsd(
            k_cache, v_cache, page_size
        )
        idx_k_bnsd, idx_v_bnsd, idx_kv_heads, idx_dim = _idx_cache_to_bnsd(
            idx_k_cache, idx_v_cache, page_size
        )
        del idx_kv_heads

        bs = forward_batch.seq_lens.shape[0]
        seq_lens = self._decode_seq_lens_i32_cg.get(bs)
        if seq_lens is None:
            seq_lens = forward_batch.seq_lens.to(torch.int32)
        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        disable_index_value = idx_v_cache is None

        _native_main_kwargs = None
        if self._native_sparse_ok and _native_attn_enabled():
            try:
                _fb_id = id(forward_batch)
                _bt = self._native_decode_bt.get(_fb_id)
                if _bt is None or _bt.shape[0] != q.shape[0]:
                    _bt = self._build_native_block_table(
                        forward_batch.req_pool_indices.long(), max_blocks, q.device
                    )
                    self._native_decode_bt = {_fb_id: _bt}
                _native_main_kwargs = {"block_table": _bt}
            except Exception:
                _native_main_kwargs = None

        if disable_index_value:
            page_source_kwargs = dict(
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=False,
            )
        else:
            req_idx = forward_batch.req_pool_indices.long()
            max_cols = self.max_kv_len
            blk_cols = (
                torch.arange(max_blocks, device=q.device, dtype=torch.long) * page_size
            ).clamp(max=max_cols - 1)
            token_slots = self.req_to_token[req_idx][:, blk_cols]
            page_source_kwargs = dict(
                block_table=(token_slots // page_size).to(torch.int32)
            )

        idx_o, topk_idx = flash_decode_bnsd_with_topk_idx(
            q=idx_q,
            sink=None,
            k_cache_bnsd=idx_k_bnsd,
            v_cache_bnsd=idx_v_bnsd,
            **page_source_kwargs,
            seq_lens=seq_lens,
            max_seqlen=max_seqlen,
            block_size=page_size,
            topk=self.topk_blocks,
            init_blocks=0,
            local_blocks=0,
            sm_scale=idx_dim**-0.5,
            score_type=self.score_type,
            disable_index_value=disable_index_value,
            runtime_fill_only=True,
            score_max_chunks=self._choose_decode_score_max_chunks(bs),
            fused_append_local=True,
            use_native=_native_indexer_enabled(),
        )

        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx, seq_lens, num_idx_heads, num_kv_heads, max_blocks
        )

        _main_kwargs = (
            {**page_source_kwargs, **_native_main_kwargs}
            if _native_main_kwargs is not None
            else page_source_kwargs
        )
        o = flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_bnsd,
            v_cache_bnsd=v_bnsd,
            **_main_kwargs,
            seq_lens=seq_lens,
            block_size=page_size,
            topk_idx=topk_idx,
            sm_scale=head_dim**-0.5,
            use_native=_native_attn_enabled(),
        )

        return idx_o, o

    def _forward_npu_triton_verify(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        prefix_lens: torch.Tensor,
    ):
        """Capture-safe sparse attention for TARGET_VERIFY."""
        del prefix_lens
        from sgl_kernel_npu.operators.attention import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )
        from sgl_kernel_npu.operators.indexer import (
            flash_decode_bnsd_with_topk_idx,
        )

        page_size = self.page_size
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]
        num_tokens = q.shape[0]
        bs = forward_batch.seq_lens.shape[0]
        ndt = num_tokens // max(bs, 1)

        k_bnsd, v_bnsd, num_pages, num_kv_heads, head_dim = _kv_cache_to_bnsd(
            k_cache, v_cache, page_size
        )
        idx_k_bnsd, idx_v_bnsd, idx_kv_heads, idx_dim = _idx_cache_to_bnsd(
            idx_k_cache, idx_v_cache, page_size
        )

        vmeta = self._verify_meta_cg.get((bs, ndt))
        if vmeta is None:
            prefix = (forward_batch.seq_lens.to(torch.long) - int(ndt)).clamp(min=0)
            offsets = torch.arange(1, int(ndt) + 1, device=q.device, dtype=torch.long)
            per_query_seq_lens = (
                (prefix.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1).to(torch.int32)
            )
            per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
                int(ndt)
            )
        else:
            per_query_seq_lens = vmeta.per_query_seq_lens
            per_query_req = vmeta.per_query_req

        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(per_query_seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        disable_index_value = idx_v_cache is None

        _native_main_kwargs = None
        if self._native_sparse_ok and _native_attn_enabled():
            try:
                _bt = (
                    vmeta.native_bt
                    if (
                        vmeta is not None
                        and getattr(vmeta, "native_bt", None) is not None
                    )
                    else None
                )
                if _bt is None:
                    _bt = self._build_native_block_table(
                        per_query_req.long(), max_blocks, q.device
                    )
                _native_main_kwargs = {"block_table": _bt}
            except Exception:
                _native_main_kwargs = None

        if disable_index_value:
            page_source_kwargs = dict(
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=per_query_req,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=True,
            )
        else:
            max_cols = self.max_kv_len
            blk_cols = (
                torch.arange(max_blocks, device=q.device, dtype=torch.long) * page_size
            ).clamp(max=max_cols - 1)
            token_slots = self.req_to_token[per_query_req][:, blk_cols]
            block_table = (token_slots // page_size).to(torch.int32)
            block_table = block_table.clamp(min=0, max=num_pages - 1)
            page_source_kwargs = dict(block_table=block_table)

        pack_verify = (
            disable_index_value and int(ndt) > 1 and num_idx_heads == idx_kv_heads
        )
        if pack_verify:
            idx_q_score = idx_q.reshape(bs, ndt * num_idx_heads, idx_dim)
            if num_idx_heads == 1:
                score_seq_lens = per_query_seq_lens
            else:
                score_seq_lens = (
                    per_query_seq_lens.view(bs, ndt, 1)
                    .expand(bs, ndt, num_idx_heads)
                    .reshape(-1)
                )
            score_page_source_kwargs = dict(
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=True,
            )
        else:
            idx_q_score = idx_q
            score_seq_lens = per_query_seq_lens
            score_page_source_kwargs = page_source_kwargs

        idx_o, topk_idx = flash_decode_bnsd_with_topk_idx(
            q=idx_q_score,
            sink=None,
            k_cache_bnsd=idx_k_bnsd,
            v_cache_bnsd=idx_v_bnsd,
            **score_page_source_kwargs,
            seq_lens=score_seq_lens,
            max_seqlen=max_seqlen,
            block_size=page_size,
            topk=self.topk_blocks,
            init_blocks=0,
            local_blocks=0,
            sm_scale=idx_dim**-0.5,
            score_type=self.score_type,
            disable_index_value=disable_index_value,
            packed_seq_lens=pack_verify,
            score_blocks_per_chunk=8 if pack_verify else 16,
            score_max_chunks=64 if pack_verify else 32,
            runtime_fill_only=pack_verify,
            runtime_score_short_max_blocks=256 if pack_verify else 0,
            runtime_score_short_chunks=16 if pack_verify else 0,
            fused_append_local=True,
            use_native=_native_indexer_enabled(),
        )
        if pack_verify:
            k_last = topk_idx.shape[-1]
            topk_idx = (
                topk_idx.view(ndt, num_idx_heads, bs, k_last)
                .permute(1, 2, 0, 3)
                .reshape(num_idx_heads, bs * ndt, k_last)
                .contiguous()
            )

        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx,
            per_query_seq_lens,
            num_idx_heads,
            num_kv_heads,
            max_blocks,
        )

        _vmain_kwargs = (
            {**page_source_kwargs, **_native_main_kwargs}
            if _native_main_kwargs is not None
            else page_source_kwargs
        )
        o = flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_bnsd,
            v_cache_bnsd=v_bnsd,
            **_vmain_kwargs,
            seq_lens=per_query_seq_lens,
            block_size=page_size,
            topk_idx=topk_idx,
            sm_scale=head_dim**-0.5,
            use_native=_native_attn_enabled(),
        )
        return idx_o, o

    def _build_prefill_meta(
        self,
        forward_batch: ForwardBatch,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        device,
        page_size: int,
        num_pages: int,
        total_q: int,
    ) -> SimpleNamespace:
        """Build layer-invariant prefill metadata once per forward."""
        del num_pages
        seq_lens_l = seq_lens.to(device=device, dtype=torch.long)
        prefix_lens_l = prefix_lens.to(device=device, dtype=torch.long)
        cu_q = cu_seqlens.to(device=device, dtype=torch.long)
        extend_lens = (seq_lens_l - prefix_lens_l).clamp(min=0)
        per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
            extend_lens
        )
        per_query_prefix = prefix_lens_l.repeat_interleave(extend_lens)
        per_query_within = torch.arange(
            total_q, device=device, dtype=torch.long
        ) - cu_q[:-1].repeat_interleave(extend_lens)
        per_query_seq_lens = (per_query_prefix + per_query_within + 1).to(torch.int32)

        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(per_query_seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        block_size_q = self._choose_block_size_q(max_seqlen)

        from sgl_kernel_npu.operators.indexer.indexer_utils import (
            build_qblock_mappings,
        )

        qblock_mappings = build_qblock_mappings(
            cu_seqlens,
            seq_lens,
            self.req_to_token,
            forward_batch.req_pool_indices,
            block_size_q,
            page_size,
            max_blocks,
            device,
        )

        topk1 = self.topk_blocks + 1
        fia_block_table_ws = torch.empty(
            (total_q, topk1), dtype=torch.int32, device=device
        )
        fia_actual_kvlen_ws = torch.empty((total_q,), dtype=torch.int32, device=device)

        return SimpleNamespace(
            per_query_req=per_query_req,
            per_query_req_i32=per_query_req.to(torch.int32),
            per_query_seq_lens=per_query_seq_lens,
            max_seqlen=max_seqlen,
            max_blocks=max_blocks,
            block_size_q=block_size_q,
            qblock_mappings=qblock_mappings,
            fia_block_table_ws=fia_block_table_ws,
            fia_actual_kvlen_ws=fia_actual_kvlen_ws,
        )

    def _forward_npu_triton_prefill(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        main_num_warps: int = 4,
        main_num_stages: int = 2,
        main_blocks_per_step: int = 1,
    ):
        """NPU block-sparse PREFILL via the ported triton decode kernels."""
        from sgl_kernel_npu.operators.attention import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )

        page_size = self.page_size
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]
        total_q = q.shape[0]

        k_bnsd, v_bnsd, num_pages, num_kv_heads, head_dim = _kv_cache_to_bnsd(
            k_cache, v_cache, page_size
        )
        idx_k_bnsd, idx_v_bnsd, idx_kv_heads, idx_dim = _idx_cache_to_bnsd(
            idx_k_cache, idx_v_cache, page_size
        )
        del idx_kv_heads

        meta = self._prefill_meta
        if meta is None:
            meta = self._build_prefill_meta(
                forward_batch,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                q.device,
                page_size,
                num_pages,
                total_q,
            )
            self._prefill_meta = meta
        per_query_seq_lens = meta.per_query_seq_lens
        max_blocks = meta.max_blocks
        block_size_q = meta.block_size_q
        per_query_req = meta.per_query_req

        disable_index_value = idx_v_cache is None

        from sgl_kernel_npu.operators.indexer import (
            flash_prefill_bnsd_indexer,
        )
        from sgl_kernel_npu.operators.indexer import (
            flash_prefill_bnsd_with_topk_idx as _flash_prefill_score_topk,
        )

        if disable_index_value:
            idx_o = None
            topk_idx = _flash_prefill_score_topk(
                idx_q,
                idx_k_bnsd,
                cu_seqlens,
                seq_lens,
                self.req_to_token,
                forward_batch.req_pool_indices,
                block_size_q,
                page_size,
                self.topk_blocks,
                idx_dim**-0.5,
                self.score_type,
                qblock_mappings=meta.qblock_mappings,
                per_query_seq_lens=per_query_seq_lens,
            )
        else:
            idx_o, topk_idx = flash_prefill_bnsd_indexer(
                idx_q,
                idx_k_bnsd,
                idx_v_bnsd,
                cu_seqlens,
                seq_lens,
                self.req_to_token,
                forward_batch.req_pool_indices,
                block_size_q,
                page_size,
                self.topk_blocks,
                idx_dim**-0.5,
                self.score_type,
                qblock_mappings=meta.qblock_mappings,
                per_query_seq_lens=per_query_seq_lens,
            )

        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx,
            per_query_seq_lens,
            num_idx_heads,
            num_kv_heads,
            max_blocks,
        )

        main_bps = int(
            os.environ.get(
                "SGLANG_MINIMAX_NPU_PREFILL_MAIN_BPS", str(main_blocks_per_step)
            )
        )
        main_ns = main_num_stages if main_bps == 1 else min(main_num_stages, 1)

        def _decode_main():
            return flash_decode_bnsd_with_gqa_share_sparse(
                q=q,
                sink=None,
                k_cache_bnsd=k_bnsd,
                v_cache_bnsd=v_bnsd,
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=per_query_req,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=True,
                seq_lens=per_query_seq_lens,
                block_size=page_size,
                topk_idx=topk_idx,
                sm_scale=head_dim**-0.5,
                topk_blocks_per_step=main_bps,
                num_warps=main_num_warps,
                num_stages=main_ns,
            )

        def _fia_main():
            from sgl_kernel_npu.operators.attention import (
                flash_prefill_bnsd_blockq_sparse_fia,
            )

            return flash_prefill_bnsd_blockq_sparse_fia(
                q=q,
                k_cache_bnsd=k_bnsd,
                v_cache_bnsd=v_bnsd,
                topk_idx=topk_idx,
                seq_lens=per_query_seq_lens,
                per_query_req=meta.per_query_req_i32,
                req_to_token=self.req_to_token,
                block_size=page_size,
                sm_scale=head_dim**-0.5,
                num_pages=num_pages,
                max_num_blocks=max_blocks,
                block_table_out=meta.fia_block_table_ws,
                actual_kvlen_out=meta.fia_actual_kvlen_ws,
            )

        use_fia = envs.SGLANG_MINIMAX_NPU_PREFILL_FIA.get() and num_kv_heads == 1
        o = _fia_main() if use_fia else _decode_main()

        return idx_o, o

    def _resolve_extend_meta(self, forward_batch: ForwardBatch, q: torch.Tensor):
        """Return (cu_seqlens, seq_lens, prefix_lens); NPU caches per-forward casts."""
        if forward_batch.extend_seq_lens is None:
            _bs = forward_batch.seq_lens.shape[0]
            _ndt = self.speculative_num_draft_tokens or (q.shape[0] // max(_bs, 1))
            forward_batch.extend_seq_lens = torch.full(
                (_bs,),
                int(_ndt),
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            forward_batch.extend_seq_lens_cpu = [int(_ndt)] * _bs
            if forward_batch.extend_prefix_lens is None:
                forward_batch.extend_prefix_lens = (
                    forward_batch.seq_lens.to(torch.int32) - int(_ndt)
                ).clamp(min=0)

        if self._extend_meta_key == id(forward_batch) and self._extend_meta is not None:
            m = self._extend_meta
            return m.cu_seqlens, m.seq_lens, m.prefix_lens

        cu_seqlens = torch.cat(
            [
                torch.zeros(
                    1, dtype=torch.int32, device=forward_batch.extend_seq_lens.device
                ),
                forward_batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
            ]
        )
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
        else:
            prefix_lens = torch.zeros_like(seq_lens)

        self._extend_meta = SimpleNamespace(
            cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens
        )
        self._extend_meta_key = id(forward_batch)
        return cu_seqlens, seq_lens, prefix_lens

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        out_quant: str = "",
    ):
        del save_kv_cache, out_quant
        disable_value = layer.layer_id in self.disable_value_layer_ids
        if not self._is_sparse_kv_cached_by_fusion(forward_batch, layer.layer_id):
            self.kv_pool.set_fused_kv_index_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                idx_k,
                None if disable_value else idx_v,
                layer.k_scale_float,
                layer.v_scale_float,
                layer.idx_k_scale_float,
                layer.idx_v_scale_float,
            )
        k_cache, v_cache = self.kv_pool.layer_router.read(layer.layer_id).kv()
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        cached = self._prefill_seqblock_meta
        if cached is None or cached[0] is not forward_batch:
            cu_seqlens, seq_lens, prefix_lens = self._resolve_extend_meta(
                forward_batch, q
            )
            cached = (
                forward_batch,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                None,
                None,
                None,
            )
            self._prefill_seqblock_meta = cached
        _, cu_seqlens, seq_lens, prefix_lens, _, _, _ = cached

        # DP attention pads q beyond real tokens; trim (CPU list avoids a sync).
        if forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
        else:
            actual_num_tokens = int(cu_seqlens[-1].item())
        original_num_tokens = q.shape[0]
        if actual_num_tokens < original_num_tokens:
            q = q[:actual_num_tokens]
            idx_q = idx_q[:actual_num_tokens]

        if forward_batch.forward_mode.is_target_verify():
            idx_o, o = self._forward_npu_triton_verify(
                q,
                k_cache,
                v_cache,
                idx_q,
                idx_k_cache,
                idx_v_cache,
                forward_batch,
                prefix_lens,
            )
        else:
            idx_o, o = self._forward_npu_triton_prefill(
                q,
                k_cache,
                v_cache,
                idx_q,
                idx_k_cache,
                idx_v_cache,
                forward_batch,
                cu_seqlens,
                seq_lens,
                prefix_lens,
            )

        if actual_num_tokens < original_num_tokens:
            pad_len = original_num_tokens - actual_num_tokens
            o = torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)
            if idx_o is not None:
                idx_o = torch.cat(
                    [idx_o, idx_o.new_zeros(pad_len, *idx_o.shape[1:])], dim=0
                )

        if not isinstance(o, tuple):
            o = o.reshape(original_num_tokens, -1).contiguous()
        return (
            (
                None
                if idx_o is None
                else idx_o.reshape(original_num_tokens, -1).contiguous()
            ),
            o,
        )

    def forward_decode(
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
        **kwargs,
    ):
        del save_kv_cache, out_quant
        assert len(kwargs) == 0
        disable_value = layer.layer_id in self.disable_value_layer_ids
        if not self._is_sparse_kv_cached_by_fusion(forward_batch, layer.layer_id):
            self.kv_pool.set_fused_kv_index_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                idx_k,
                None if disable_value else idx_v,
                layer.k_scale_float,
                layer.v_scale_float,
                layer.idx_k_scale_float,
                layer.idx_v_scale_float,
            )
        k_cache, v_cache = self.kv_pool.layer_router.read(layer.layer_id).kv()
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        idx_o, o = self._forward_npu_triton_decode(
            q,
            k_cache,
            v_cache,
            idx_q,
            idx_k_cache,
            idx_v_cache,
            forward_batch,
        )
        if not isinstance(o, tuple):
            o = o.reshape(q.shape[0], -1).contiguous()
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o,
        )
