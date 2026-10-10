"""QSA indexer for Qwen4-Exp checkpoints."""

from __future__ import annotations

import os

import torch
from sglang.srt.layers.attention.qsa.cache_sharding import (
    QSACacheShardingRuntime,
    assert_qsa_cache_sharding_runtime_match,
    get_qsa_cache_sharding_runtime,
)
from sglang.srt.layers.attention.qsa.distributed_topk import (
    gather_and_merge_qsa_topk_candidates,
)
from sglang.srt.layers.attention.qsa.kernel import (
    average_pool_qsa_keys,
    expand_qsa_block_indices,
    qsa_fast_topk,
)
from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
    build_rope_position_matrix,
)
from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_decode, qsa_mqa_prefill
from sglang.srt.layers.cp.utils import (
    ContextParallelStrategyKind,
    cp_materialize_global_token_order,
    cp_shard_hidden_states,
    cp_shard_position_ids,
    get_cp_strategy,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.mem_cache.qsa_kv_pool import assert_qsa_indices_in_bounds

# Cap on the fp32 [query_rows, compressed_keys] prefill logits workspace;
# top-k is per row, so tiling rows does not change the selection.
_QSA_PREFILL_LOGITS_BUDGET_BYTES = 128 * 1024 * 1024


def _qsa_prefill_row_chunk_size(
    rows: int,
    keys: int,
    heads: int,
    *,
    collective_keys: int | None = None,
    collective_world_size: int = 1,
) -> int:
    if rows <= 0 or keys <= 0:
        return max(rows, 1)
    if collective_keys is not None:
        if collective_world_size <= 0:
            raise ValueError("collective_world_size must be positive")
        # Every rank must issue the same sequence of candidate collectives.
        # Rank-local sharded cache lengths can differ, so derive the workspace
        # schedule from the shared global key count instead.
        keys = -(collective_keys // -collective_world_size)
    block_q = max(1, 128 // heads)
    bytes_per_row = keys * torch.float32.itemsize
    max_padded_rows = max(block_q, _QSA_PREFILL_LOGITS_BUDGET_BYTES // bytes_per_row)
    max_padded_rows = max(block_q, max_padded_rows // block_q * block_q)
    return min(rows, max_padded_rows)


class QSAIndexer(MultiPlatformOp):
    """Config-driven fused-QK, weight-free sparse-attention indexer."""

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
        rotary_emb=None,
        distributed_topk_group=None,
        cache_sharding_runtime: QSACacheShardingRuntime | None = None,
    ) -> None:
        self._validate_config(config)
        super().__init__()
        resolved_runtime = get_qsa_cache_sharding_runtime()
        if cache_sharding_runtime is None:
            cache_sharding_runtime = resolved_runtime
        assert_qsa_cache_sharding_runtime_match(
            resolved_runtime,
            cache_sharding_runtime,
            component="QSA indexer",
        )
        expected_group = (
            cache_sharding_runtime.group if cache_sharding_runtime.enabled else None
        )
        if (
            distributed_topk_group is not None
            and distributed_topk_group is not expected_group
        ):
            raise ValueError(
                "indexer QSA cache-sharding collective group does not match "
                "the runtime contract"
            )
        self.cache_sharding_runtime = cache_sharding_runtime
        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.block_topk = self.token_topk // self.compress_ratio
        if rotary_emb is None:
            raise ValueError("QSAIndexer must reuse its Qwen4-Exp attention RoPE")
        self.rotary_emb = rotary_emb
        if not 0 < self.rotary_emb.rotary_dim <= self.index_head_dim:
            raise ValueError(
                "Qwen4-Exp attention RoPE rotary_dim must fit the QSA index head: "
                f"{self.rotary_emb.rotary_dim=} {self.index_head_dim=}"
            )
        self.index_qk_proj = ReplicatedLinear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self._rope_axis_map_cache = None
        self.distributed_topk_group = expected_group

    @staticmethod
    def _validate_config(config) -> None:
        names = (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
        missing = [name for name in names if getattr(config, name, None) is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}")
        values = {name: int(getattr(config, name)) for name in names}
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"QSA config values must be positive: {values}")
        if values["indexer_compress_ratio"] < 2:
            # DP token-padding rows carry logical length 1, which must never
            # reach a compression boundary; ratio >= 2 guarantees that.
            raise ValueError(
                "QSA requires indexer_compress_ratio >= 2, got "
                f"{values['indexer_compress_ratio']}"
            )
        if values["indexer_kv_heads"] != 1:
            raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
        if values["indexer_budget"] % values["indexer_compress_ratio"] != 0:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio"
            )
        block_topk = values["indexer_budget"] // values["indexer_compress_ratio"]
        if block_topk not in (512, 2048):
            raise ValueError(
                "fast_topk_v2 requires indexer_budget / indexer_compress_ratio "
                f"to be 512 or 2048, got {block_topk}"
            )

    def _use_fused_prep(self, tensor: torch.Tensor) -> bool:
        """Whether the fused indexer-prep kernels support this configuration."""
        return (
            tensor.is_cuda
            and tensor.dtype in (torch.bfloat16, torch.float16)
            and self.index_head_dim in (64, 128, 256)
            and self.rotary_emb.rotary_dim % 2 == 0
            and not getattr(self.rotary_emb, "mrope_interleaved_glm", False)
            and len(getattr(self.rotary_emb, "mrope_section", None) or ()) in (0, 3)
            and getattr(self.rotary_emb, "cos_sin_cache", None) is not None
            and self.rotary_emb.cos_sin_cache.is_cuda
            and self.rotary_emb.cos_sin_cache.dtype == torch.float32
        )

    def _rope_axis_map(self, device) -> torch.Tensor:
        """axis_map[i] is the MRoPE position axis whose cos/sin rotary pair i reads."""
        cache = self._rope_axis_map_cache
        if cache is not None and cache.device == device:
            return cache
        half = self.rotary_emb.rotary_dim // 2
        section = getattr(self.rotary_emb, "mrope_section", None) or None
        axis_map = torch.zeros(half, dtype=torch.int32)
        if section is not None:
            s0, s1, s2 = (int(v) for v in section)
            if getattr(self.rotary_emb, "mrope_interleaved", False):
                pair = torch.arange(half, dtype=torch.int32)
                axis_map[((pair % 3) == 1) & (pair < s1 * 3)] = 1
                axis_map[((pair % 3) == 2) & (pair < s2 * 3)] = 2
            else:
                axis_map[s0 : s0 + s1] = 1
                axis_map[s0 + s1 :] = 2
        self._rope_axis_map_cache = axis_map.to(device)
        return self._rope_axis_map_cache

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        pool=None,
        cache_loc: torch.Tensor | None = None,
        q_heads_padded: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        qk, _ = self.index_qk_proj(hidden_states)
        token_k = qk[:, self.index_n_heads * self.index_head_dim :].reshape(
            -1, self.index_kv_heads, self.index_head_dim
        )
        if (
            pool is not None
            and cache_loc is not None
            and qk.shape[0] > 0
            and self._use_fused_prep(qk)
        ):
            from sglang.kernels.ops.attention.qsa_indexer import (
                qsa_index_q_norm_rope_store,
            )

            if not get_is_capture_mode() and hasattr(
                self.rotary_emb, "_ensure_cos_sin_cache_length"
            ):
                self.rotary_emb._ensure_cos_sin_cache_length(
                    int(positions.max().item())
                )
            key_state_buffer = pool.get_qsa_key_state_buffer(self.layer_id)
            fused_state_slots = cache_loc[: qk.shape[0]].long()
            assert_qsa_indices_in_bounds(
                fused_state_slots,
                key_state_buffer.shape[0],
                valid_mask=torch.ones_like(fused_state_slots, dtype=torch.bool),
                label="QSA pending-state write locations",
            )
            q = qsa_index_q_norm_rope_store(
                qk,
                positions.long(),
                self.rotary_emb.cos_sin_cache,
                self._rope_axis_map(qk.device),
                self.q_layernorm.weight.data,
                fused_state_slots,
                key_state_buffer.view(key_state_buffer.shape[0], -1),
                pool.qsa_rope_position_buffer,
                self.index_n_heads,
                self.rotary_emb.rotary_dim,
                self.q_layernorm.variance_epsilon,
                self.rotary_emb.is_neox_style,
                q_heads_padded=q_heads_padded,
                out_dtype=pool.qsa_compressed_dtype,
            )
            return q, token_k, True
        q_raw = qk[:, : self.index_n_heads * self.index_head_dim]
        q = self.q_layernorm(q_raw.reshape(-1, self.index_head_dim)).reshape(
            -1, self.index_n_heads, self.index_head_dim
        )
        q = self.apply_rope(positions, q)
        if pool is not None:
            q = q.to(pool.qsa_compressed_dtype)
        return q, token_k, False

    def normalize_compressed_keys(
        self, compressed_keys: torch.Tensor, block_positions: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.k_layernorm(
            compressed_keys.reshape(-1, self.index_head_dim)
        ).reshape(-1, self.index_kv_heads, self.index_head_dim)
        return self.apply_rope(block_positions, normalized)

    def _use_fused_compress(self, pool) -> bool:
        return getattr(
            pool, "qsa_rope_position_buffer", None
        ) is not None and self._use_fused_prep(
            pool.get_qsa_key_state_buffer(self.layer_id)
        )

    def _fused_compress_store(
        self,
        pool,
        group_locs: torch.Tensor,
        owner_mask: torch.Tensor,
        local_write_locs: torch.Tensor | None = None,
        source_keys: torch.Tensor | None = None,
        source_rope: torch.Tensor | None = None,
    ) -> None:
        """Fused mean -> gemma norm -> MRoPE -> compressed-cache store;
        a None source_keys/source_rope reads the members from the pending ring."""
        from sglang.kernels.ops.attention.qsa_indexer import (
            qsa_index_k_compress_store,
        )

        if source_keys is None:
            source_keys = pool.get_qsa_key_state_buffer(self.layer_id)
        if source_rope is None:
            source_rope = pool.qsa_rope_position_buffer
        if local_write_locs is None:
            local_write_locs = owner_mask
            owner_mask = torch.ones_like(local_write_locs, dtype=torch.bool)
        compressed_buffer = pool.get_qsa_compressed_k_buffer(self.layer_id)
        owner_rows = owner_mask.to(torch.bool)
        assert_qsa_indices_in_bounds(
            group_locs,
            source_keys.shape[0],
            valid_mask=owner_rows.unsqueeze(1).expand_as(group_locs),
            label="QSA compressed source read locations",
        )
        assert_qsa_indices_in_bounds(
            local_write_locs,
            compressed_buffer.shape[0],
            valid_mask=owner_rows,
            label="QSA compressed destination write locations",
        )
        qsa_index_k_compress_store(
            source_keys.reshape(source_keys.shape[0], -1)
            .contiguous()
            .to(pool.index_state_dtype),
            group_locs.to(torch.int32),
            source_rope,
            self.rotary_emb.cos_sin_cache,
            self._rope_axis_map(source_keys.device),
            self.k_layernorm.weight.data,
            owner_mask.to(torch.int32),
            local_write_locs.to(torch.int32),
            compressed_buffer.view(compressed_buffer.shape[0], -1),
            self.compress_ratio,
            self.rotary_emb.rotary_dim,
            self.k_layernorm.variance_epsilon,
            self.rotary_emb.is_neox_style,
        )

    def _pending_ring_slots(
        self, metadata, logical_positions: torch.Tensor, is_extend: bool
    ) -> torch.Tensor:
        return build_pending_ring_slots(
            token_to_batch_idx=metadata.token_to_batch_idx,
            req_pool_indices=metadata.req_pool_indices,
            sequence_lengths=metadata.sequence_lengths,
            logical_positions=logical_positions,
            compress_ratio=self.compress_ratio,
            is_extend=is_extend,
        )

    def _group_ring_slots(
        self, metadata, group_end_positions: torch.Tensor, sequence_ids: torch.Tensor
    ) -> torch.Tensor:
        return build_group_ring_slots(
            req_pool_indices=metadata.req_pool_indices,
            group_end_positions=group_end_positions,
            sequence_ids=sequence_ids,
            compress_ratio=self.compress_ratio,
        )

    def update_key_state_and_compress(
        self,
        token_k: torch.Tensor,
        logical_positions: torch.Tensor,
        rope_positions: torch.Tensor,
        metadata,
        state_slots: torch.Tensor | None = None,
        state_stored: bool = False,
    ) -> None:
        """Store the pending-group ring and compress each completed group."""

        pool = metadata.token_to_kv_pool
        is_extend = metadata.compress_member_rows is not None
        if not state_stored:
            if state_slots is None:
                state_slots = self._pending_ring_slots(
                    metadata, logical_positions, is_extend
                )
            pool.set_qsa_key_state_buffer(
                self.layer_id, state_slots[: token_k.shape[0]], token_k
            )
            pool.set_qsa_rope_position_buffer(
                state_slots[: token_k.shape[0]], rope_positions
            )

        if metadata.is_cuda_graph:
            self._compress_decode_cuda_graph(metadata)
            return

        if metadata.write_locs is None:
            raise RuntimeError(
                "QSA metadata is missing the precomputed write plan; the "
                "sparse-attention backend derives it from the batch lengths"
            )
        if metadata.write_locs.numel() == 0:
            return
        group_end_positions = metadata.compress_group_positions.long()
        owner_mask = metadata.write_owner_mask
        compressed_locs = metadata.local_write_locs
        if owner_mask is None or compressed_locs is None:
            raise RuntimeError(
                "QSA eager compression metadata requires owner mask and local "
                "write locations"
            )
        owner_mask = owner_mask.bool()
        if is_extend:
            # Extend chunks are group-aligned; each planned group lies in this forward,
            # so read its members from the packed chunk tensors.
            member_rows = metadata.compress_member_rows.long()
            group_locs = member_rows[:, None] + torch.arange(
                self.compress_ratio, device=member_rows.device, dtype=torch.long
            )
            source_keys = token_k
            group_locs = group_locs.clamp_max(source_keys.shape[0] - 1)
            source_rope = metadata.extend_rope_matrix
            if source_rope is None:
                source_rope = build_rope_position_matrix(
                    rope_positions, token_k.shape[0]
                )
        else:
            # Paged eager rows (speculative fallback) complete at most one
            # group each; its members are exactly the pending ring window.
            group_locs = metadata.compress_group_ring_locs
            if group_locs is None:
                group_locs = self._group_ring_slots(
                    metadata,
                    group_end_positions,
                    metadata.compress_sequence_ids.long(),
                )
            source_keys = pool.get_qsa_key_state_buffer(self.layer_id)
            source_rope = pool.qsa_rope_position_buffer
        if self._use_fused_compress(pool):
            self._fused_compress_store(
                pool,
                group_locs,
                owner_mask,
                compressed_locs,
                source_keys=source_keys,
                source_rope=source_rope,
            )
            return
        owned_rows = torch.nonzero(owner_mask, as_tuple=False).flatten()
        if owned_rows.numel() == 0:
            return
        owned_group_locs = group_locs.index_select(0, owned_rows)
        key_groups = source_keys[owned_group_locs]
        pooled = average_pool_qsa_keys(key_groups)
        compressed_rope_positions = self._rope_from_matrix(
            source_rope[owned_group_locs[:, 0]]
        )
        normalized = self.normalize_compressed_keys(pooled, compressed_rope_positions)
        buffer = pool.get_qsa_compressed_k_buffer(self.layer_id)
        buffer[compressed_locs.index_select(0, owned_rows).long()] = normalized.to(
            buffer.dtype
        )

    def _compress_decode_cuda_graph(self, metadata) -> None:
        """Fixed-shape graph-replay compression; non-boundary rows write slot 0."""

        if metadata.graph_write_locs is None or metadata.graph_ring_group_locs is None:
            raise RuntimeError("QSA CUDA graph compression metadata is incomplete")
        pool = metadata.token_to_kv_pool
        group_locs = metadata.graph_ring_group_locs
        if self._use_fused_compress(pool):
            self._fused_compress_store(
                pool,
                group_locs,
                (
                    metadata.write_owner_mask
                    if metadata.write_owner_mask is not None
                    else metadata.graph_write_locs != 0
                ),
                metadata.graph_write_locs,
            )
            return
        key_groups = pool.get_qsa_key_state_buffer(self.layer_id)[group_locs]
        compressed = average_pool_qsa_keys(key_groups)
        compressed_rope_positions = self._rope_from_matrix(
            pool.qsa_rope_position_buffer[group_locs[:, 0]]
        )
        compressed = self.normalize_compressed_keys(
            compressed, compressed_rope_positions
        )
        pool.set_qsa_compressed_k_buffer(
            self.layer_id, metadata.graph_write_locs, compressed.contiguous()
        )

    def _rope_from_matrix(self, positions: torch.Tensor) -> torch.Tensor:
        """[n, 3] slot coordinates -> the layout apply_rope expects."""
        positions = positions.transpose(0, 1)
        if not getattr(self.rotary_emb, "mrope_section", None):
            return positions[0]
        return positions

    def apply_rope(self, positions: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        positions = positions.long()
        num_positions = (
            positions.shape[-1] if positions.ndim == 2 else positions.numel()
        )
        if num_positions != tensor.shape[0]:
            raise ValueError("QSA RoPE positions must match the token dimension")
        if not get_is_capture_mode() and hasattr(
            self.rotary_emb, "_ensure_cos_sin_cache_length"
        ):
            self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))

        # position_cos/position_sin repeat cos/sin to the full rotary width;
        # apply_rotary_emb consumes one half.
        self.rotary_emb.get_cos_sin_with_position(positions)
        rotary_dim = self.rotary_emb.rotary_dim
        half_rotary_dim = rotary_dim // 2
        cos = self.rotary_emb.position_cos.reshape(num_positions, -1)[
            :, :half_rotary_dim
        ]
        sin = self.rotary_emb.position_sin.reshape(num_positions, -1)[
            :, :half_rotary_dim
        ]
        rotated = apply_rotary_emb(
            tensor[..., :rotary_dim],
            cos,
            sin,
            self.rotary_emb.is_neox_style,
        )
        return torch.cat([rotated, tensor[..., rotary_dim:]], dim=-1)

    def _maybe_merge_distributed_topk(
        self,
        logits: torch.Tensor | None,
        local_indices: torch.Tensor,
        row_starts: torch.Tensor,
        logical_block_positions: torch.Tensor | None = None,
        static_buffers=None,
    ) -> torch.Tensor:
        """Convert local candidates to exact global top-k when explicitly enabled."""

        group = getattr(self, "distributed_topk_group", None)
        if group is None:
            return local_indices
        if (
            static_buffers is not None
            and getattr(static_buffers, "candidate_symm_state", None) is not None
            and logits is not None
            and logits.shape[1] > 0
            and os.environ.get("SGLANG_QSA_FUSED_CANDIDATE_PUBLISH", "1") != "0"
        ):
            from sglang.srt.layers.attention.qsa import distributed_topk

            _, merged_indices = (
                distributed_topk.fused_publish_and_merge_qsa_topk_candidates(
                    logits,
                    local_indices,
                    row_starts,
                    logical_block_positions,
                    group=group,
                    topk=self.block_topk,
                    static_buffers=static_buffers,
                    # Interleaved compressed-page ownership makes ids disjoint.
                    deduplicate=False,
                )
            )
            return merged_indices
        valid = local_indices >= 0
        mapped_logical_positions = None
        if logical_block_positions is not None:
            if logical_block_positions.shape[1] == 0:
                valid = torch.zeros_like(valid)
            else:
                within_local_row = local_indices < logical_block_positions.shape[1]
                safe_local_indices = torch.where(
                    valid & within_local_row,
                    local_indices,
                    torch.zeros_like(local_indices),
                ).long()
                mapped_logical_positions = logical_block_positions.gather(
                    1, safe_local_indices
                )
                valid &= within_local_row & (mapped_logical_positions >= 0)
        if logits is None or logits.shape[1] == 0:
            local_scores = torch.full(
                local_indices.shape,
                float("-inf"),
                dtype=torch.float32,
                device=local_indices.device,
            )
        else:
            absolute_indices = (
                local_indices.clamp_min(0).long()
                + row_starts.long().unsqueeze(1)
            )
            within_logits = absolute_indices < logits.shape[1]
            valid &= within_logits
            safe_absolute_indices = torch.where(
                valid, absolute_indices, torch.zeros_like(absolute_indices)
            )
            local_scores = logits.gather(1, safe_absolute_indices).masked_fill(
                ~valid, float("-inf")
            )
        if logical_block_positions is None:
            global_indices = local_indices.to(torch.int64) * int(
                group.world_size
            ) + int(group.rank_in_group)
        elif logical_block_positions.shape[1] == 0:
            global_indices = torch.full_like(local_indices, -1, dtype=torch.int64)
        else:
            global_indices = mapped_logical_positions
        global_indices = torch.where(
            valid, global_indices, torch.full_like(global_indices, -1)
        ).to(torch.int32)
        graph_kwargs = (
            {"static_buffers": static_buffers} if static_buffers is not None else {}
        )
        _, merged_indices = gather_and_merge_qsa_topk_candidates(
            local_scores,
            global_indices,
            group=group,
            topk=self.block_topk,
            # Interleaved compressed-page ownership partitions logical block
            # positions across QSA cache shards, so gathered ids are disjoint.
            deduplicate=False,
            **graph_kwargs,
        )
        return merged_indices

    def select_prefill_tokens(
        self,
        q: torch.Tensor,
        compressed_keys: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths_for_rows: torch.Tensor,
        logical_block_positions: torch.Tensor | None = None,
        logical_block_sequence_ids: torch.Tensor | None = None,
        static_buffers=None,
        collective_keys: int | None = None,
        collective_world_size: int = 1,
    ) -> torch.Tensor:
        rows = q.shape[0]
        output = torch.empty(
            (rows, self.token_topk + self.compress_ratio - 1),
            dtype=torch.int32,
            device=q.device,
        )
        if rows == 0:
            return output

        row_chunk_size = _qsa_prefill_row_chunk_size(
            rows,
            compressed_keys.shape[0],
            q.shape[1],
            collective_keys=collective_keys,
            collective_world_size=collective_world_size,
        )
        for row_start in range(0, rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, rows)
            chunk_slice = slice(row_start, row_end)
            if compressed_keys.shape[0] == 0:
                block_indices = torch.full(
                    (row_end - row_start, self.block_topk),
                    -1,
                    dtype=torch.int32,
                    device=q.device,
                )
                logits = None
            else:
                logits = qsa_mqa_prefill(
                    q[chunk_slice],
                    compressed_keys,
                    row_starts[chunk_slice],
                    row_ends[chunk_slice],
                )
                block_indices = qsa_fast_topk(
                    logits,
                    row_starts[chunk_slice],
                    row_ends[chunk_slice],
                    topk=self.block_topk,
                )
            chunk_logical_block_positions = None
            if logical_block_positions is not None:
                chunk_logical_block_positions = (
                    logical_block_positions[chunk_slice]
                    if logical_block_sequence_ids is None
                    else logical_block_positions.index_select(
                        0, logical_block_sequence_ids[chunk_slice].long()
                    )
                )
            block_indices = QSAIndexer._maybe_merge_distributed_topk(
                self,
                logits,
                block_indices,
                row_starts[chunk_slice],
                chunk_logical_block_positions,
                static_buffers,
            )
            selected = expand_qsa_block_indices(
                block_indices,
                query_positions[chunk_slice],
                sequence_lengths_for_rows[chunk_slice],
                compress_ratio=self.compress_ratio,
                token_topk=self.token_topk,
            )
            output[chunk_slice].copy_(selected)
            del logits, block_indices, selected
        return output

    def select_decode_tokens(
        self,
        q: torch.Tensor,
        compressed_cache: torch.Tensor,
        compressed_page_table: torch.Tensor,
        compressed_lengths: torch.Tensor,
        max_model_len: int,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        defer_expansion: bool = False,
        logical_block_positions: torch.Tensor | None = None,
        static_buffers=None,
    ) -> torch.Tensor:
        row_starts = torch.zeros_like(compressed_lengths, dtype=torch.int32)
        with torch.profiler.record_function("qsa.local_scan"):
            logits = qsa_mqa_decode(
                q,
                compressed_cache,
                compressed_page_table,
                compressed_lengths,
                max_model_len,
            )
        if logits.is_cuda and self.block_topk == 512:
            # Decode rows start at zero, so compressed lengths double as row lengths;
            # skip the generic zero-fill + subtract.
            from sglang.kernels.ops.attention.fast_topk import fast_topk

            block_indices = fast_topk(
                logits,
                compressed_lengths.to(torch.int32),
                topk=self.block_topk,
                row_starts=None,
            )
        else:
            block_indices = qsa_fast_topk(
                logits, row_starts, compressed_lengths, topk=self.block_topk
            )
        block_indices = QSAIndexer._maybe_merge_distributed_topk(
            self,
            logits,
            block_indices,
            row_starts,
            logical_block_positions,
            static_buffers,
        )
        if defer_expansion:
            return block_indices
        selected = expand_qsa_block_indices(
            block_indices,
            query_positions,
            sequence_lengths,
            compress_ratio=self.compress_ratio,
            token_topk=self.token_topk,
        )
        return selected

    def forward_cuda_cp(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        global_rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill CP: score only this rank's zigzag query rows.

        ``hidden_states``/``positions`` are the local (padded) rows and their
        RoPE positions. Index q/k are projected for the local rows only; the raw
        keys are all-gathered into global token order so the pending-group ring
        store and the block compression run on the batch's global write plan
        exactly as without CP (every rank ends up with the complete compressed-K
        cache). Selection then scores the local rows against all compressed
        blocks of their request, position-driven, so the output rows line up
        with the local query rows the sparse attention consumes."""
        strategy = get_cp_strategy()
        if strategy is None or strategy.kind != ContextParallelStrategyKind.ZIGZAG:
            strategy_name = strategy.name if strategy is not None else "none"
            raise NotImplementedError(
                "QSA prefill CP only supports the zigzag strategy; "
                f"got {strategy_name}."
            )
        meta = forward_batch.attn_cp_metadata
        num_local = int(meta.total_q_prev_tokens + meta.total_q_next_tokens)
        global_logical = getattr(forward_batch, "positions", None)
        global_logical = (
            global_logical[0] if global_logical.ndim == 2 else global_logical
        ).flatten()
        num_global = indexer_metadata.get_token_to_batch_idx().numel()
        global_logical = global_logical[:num_global]
        local_logical = cp_shard_position_ids(global_logical, forward_batch)[:num_local]
        local_sequence_ids = cp_shard_hidden_states(
            indexer_metadata.get_token_to_batch_idx(), forward_batch
        )[:num_local]
        hidden_states = hidden_states[:num_local]
        positions = (
            positions[:, :num_local] if positions.ndim == 2 else positions[:num_local]
        )
        # Use the compressed-cache dtype for Q. Leaving cache_loc unset defers
        # the ring store until raw local K has been gathered into global order.
        q, token_k, _ = self.project_qk(
            hidden_states, positions, pool=indexer_metadata.token_to_kv_pool
        )
        # Raw keys of all new tokens in global packed order: the ring store
        # and compression use the global write plan unchanged. Cached prefix
        # groups already reside in the compressed-K pool on every rank.
        token_k_local = token_k.reshape(num_local, -1).contiguous()
        token_k_full = cp_materialize_global_token_order(
            token_k_local, forward_batch
        ).reshape(-1, self.index_kv_heads, self.index_head_dim)
        global_rope = (
            global_rope_positions[:, :num_global]
            if global_rope_positions.ndim == 2
            else global_rope_positions[:num_global]
        )
        self.update_key_state_and_compress(
            token_k_full,
            global_logical,
            global_rope,
            indexer_metadata,
            state_slots=indexer_metadata.pending_ring_slots,
            state_stored=False,
        )
        compressed_keys, row_starts, row_ends, sequence_lengths = (
            indexer_metadata.get_prefill_mqa_inputs(
                self.layer_id, local_logical, query_sequence_ids=local_sequence_ids
            )
        )
        row_sequence_lengths = sequence_lengths.index_select(
            0, local_sequence_ids.long()
        )
        return self.select_prefill_tokens(
            q,
            compressed_keys,
            row_starts,
            row_ends,
            local_logical,
            row_sequence_lengths,
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        cp_global_rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Portable orchestration shared by different platforms.
        Fast paths are gated per platforms inside kernel calls.
        """
        forward_mode = forward_batch.forward_mode
        is_target_verify = getattr(forward_mode, "is_target_verify", lambda: False)()
        is_draft_extend = getattr(forward_mode, "is_draft_extend_v2", lambda: False)()
        if cp_global_rope_positions is not None and not (
            forward_mode.is_decode() or is_target_verify or is_draft_extend
        ):
            return self.forward_cuda_cp(
                hidden_states,
                positions,
                forward_batch,
                indexer_metadata,
                cp_global_rope_positions,
            )
        if forward_mode.is_decode() or is_target_verify or is_draft_extend:
            # EAGLE/MTP may advance the model's RoPE coordinate independently
            # from the physical paged-KV position.  Compression and sparse
            # selection must use the latter; otherwise a draft step can index
            # token_slot_table[:, seq_len] one past the valid range.
            logical_positions = indexer_metadata.decode_logical_positions
            if logical_positions is None:
                logical_positions = indexer_metadata.get_seqlens_expanded() - 1
        else:
            logical_positions = getattr(forward_batch, "positions", None)
            if logical_positions is None:
                logical_positions = positions[0] if positions.ndim == 2 else positions
            logical_positions = logical_positions.flatten()
        # DP MAX_LEN padding adds token rows without assigning them to a
        # request. token_to_batch_idx is the source of truth for semantic rows.
        num_valid_tokens = indexer_metadata.get_token_to_batch_idx().numel()
        if logical_positions.numel() < num_valid_tokens:
            raise ValueError(
                "QSA logical positions are shorter than the request mapping: "
                f"positions={logical_positions.numel()}, mapping={num_valid_tokens}"
            )
        if hidden_states.shape[0] < num_valid_tokens:
            raise ValueError(
                "QSA hidden states are shorter than the request mapping: "
                f"hidden={hidden_states.shape[0]}, mapping={num_valid_tokens}"
            )
        position_tokens = (
            positions.shape[-1] if positions.ndim == 2 else positions.numel()
        )
        if position_tokens < num_valid_tokens:
            raise ValueError(
                "QSA RoPE positions are shorter than the request mapping: "
                f"positions={position_tokens}, mapping={num_valid_tokens}"
            )

        logical_positions = logical_positions[:num_valid_tokens]
        hidden_states = hidden_states[:num_valid_tokens]
        positions = (
            positions[:, :num_valid_tokens]
            if positions.ndim == 2
            else positions[:num_valid_tokens]
        )
        state_slots = indexer_metadata.pending_ring_slots
        if state_slots is None:
            state_slots = self._pending_ring_slots(
                indexer_metadata,
                logical_positions,
                indexer_metadata.compress_member_rows is not None,
            )
        q, token_k, state_stored = self.project_qk(
            hidden_states,
            positions,
            pool=indexer_metadata.token_to_kv_pool,
            cache_loc=state_slots,
            q_heads_padded=(
                # The tilelang decode MQA kernel needs query heads in multiples of 8.
                ((self.index_n_heads + 7) // 8) * 8
                if (forward_mode.is_decode() or is_target_verify or is_draft_extend)
                else None
            ),
        )
        self.update_key_state_and_compress(
            token_k,
            logical_positions,
            positions,
            indexer_metadata,
            state_slots=state_slots,
            state_stored=state_stored,
        )
        static_buffers = (
            getattr(
                indexer_metadata.token_to_kv_pool,
                "qsa_cache_sharding_buffers",
                None,
            )
            if indexer_metadata.is_cuda_graph
            else None
        )
        if static_buffers is not None:
            static_buffers = static_buffers.for_layer(self.layer_id)
        if forward_mode.is_decode() or is_target_verify or is_draft_extend:
            compressed_cache, page_table, compressed_lengths, max_model_len = (
                indexer_metadata.get_decode_mqa_inputs(self.layer_id)
            )
            return self.select_decode_tokens(
                q,
                compressed_cache,
                page_table,
                compressed_lengths,
                max_model_len,
                logical_positions,
                indexer_metadata.get_seqlens_int32(),
                defer_expansion=q.is_cuda and indexer_metadata.defer_block_expansion,
                logical_block_positions=getattr(
                    indexer_metadata, "decode_block_positions", None
                ),
                static_buffers=static_buffers,
            )

        compressed_keys, row_starts, row_ends, sequence_lengths = (
            indexer_metadata.get_prefill_mqa_inputs(self.layer_id, logical_positions)
        )
        query_sequence_ids = indexer_metadata.get_token_to_batch_idx()
        prefill_kwargs = {}
        block_positions = getattr(indexer_metadata, "prefill_block_positions", None)
        if block_positions is not None:
            prefill_kwargs["logical_block_positions"] = block_positions
            prefill_kwargs["logical_block_sequence_ids"] = query_sequence_ids
        group = getattr(self, "distributed_topk_group", None)
        if group is not None:
            prefill_kwargs["collective_keys"] = int(
                torch.div(
                    indexer_metadata.sequence_lengths,
                    self.compress_ratio,
                    rounding_mode="floor",
                )
                .sum()
                .item()
            )
            prefill_kwargs["collective_world_size"] = int(group.world_size)
        row_sequence_lengths = sequence_lengths.index_select(
            0, query_sequence_ids.long()
        )
        return self.select_prefill_tokens(
            q,
            compressed_keys,
            row_starts,
            row_ends,
            logical_positions,
            row_sequence_lengths,
            **prefill_kwargs,
            static_buffers=static_buffers,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        cp_global_rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            hidden_states,
            positions,
            forward_batch,
            indexer_metadata,
            cp_global_rope_positions,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
    ) -> torch.Tensor:
        return self._forward_impl(
            hidden_states, positions, forward_batch, indexer_metadata
        )


__all__ = [
    "QSAIndexer",
]
