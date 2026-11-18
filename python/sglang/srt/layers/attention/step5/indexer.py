# SPDX-License-Identifier: Apache-2.0
"""Step5 index projections, compression and block selection."""

import torch
from torch import nn

from sglang.kernels.ops.attention.step5.indexer import (
    step5_indexer_decode_logits,
    step5_indexer_norm_rope,
    step5_pack_decode_blocks,
    step5_topk_blocks,
    step5_topk_blocks_deepselect,
)
from sglang.srt.layers.attention.step5.compressor import compress_index_keys
from sglang.srt.layers.attention.step5.metadata import (
    Step5IndexerMetadata,
    Step5SelectionWorkspace,
    allocate_selection_workspace,
)
from sglang.srt.layers.attention.step5.norm import Step5RMSNorm
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    ReplicatedParallelGroup,
)
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils.common import add_prefix


class Step5Indexer(nn.Module):
    """Provider-sharded Q/W, replicated K/Z, compression and block selection."""

    def __init__(
        self,
        *,
        layer_id: int,
        hidden_size: int,
        index_head_dim: int,
        num_index_heads: int,
        index_tp_size: int,
        tp_rank: int,
        tp_size: int,
        rotary_dim: int,
        rope_theta: float,
        rope_scaling: dict,
        max_position_embeddings: int,
        eps: float,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.index_head_dim = index_head_dim
        self.provider_rank = tp_rank // (tp_size // index_tp_size)
        self.num_heads = num_index_heads // index_tp_size
        self.weight_scale = self.num_heads**-0.5
        self.sparse_indexer_q = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_index_heads * index_head_dim,
            bias=False,
            gather_output=False,
            parallel_group=ReplicatedParallelGroup("attn_tp", tp_size // index_tp_size),
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("sparse_indexer_q", prefix),
        )
        self.sparse_indexer_k = ReplicatedLinear(
            input_size=hidden_size,
            output_size=index_head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("sparse_indexer_k", prefix),
        )
        self.sparse_indexer_z = ReplicatedLinear(
            input_size=hidden_size,
            output_size=index_head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("sparse_indexer_z", prefix),
        )
        self.sparse_indexer_w = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_index_heads,
            bias=False,
            gather_output=False,
            parallel_group=ReplicatedParallelGroup("attn_tp", tp_size // index_tp_size),
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("sparse_indexer_w", prefix),
        )
        self.rotary_emb = get_rope(
            head_size=index_head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        self.eps = eps
        self.sparse_indexer_q_norm = Step5RMSNorm(
            index_head_dim, eps=eps, weight_dtype=torch.bfloat16
        )
        self.sparse_indexer_k_norm = LayerNorm(
            index_head_dim, eps=eps, dtype=torch.bfloat16
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: Step5IndexerMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = hidden_states.shape[0]
        index_q, _ = self.sparse_indexer_q(hidden_states)
        index_key, _ = self.sparse_indexer_k(hidden_states)
        compression_logits, _ = self.sparse_indexer_z(hidden_states)
        index_weights, _ = self.sparse_indexer_w(hidden_states)
        index_weights = index_weights.float() * self.weight_scale
        index_q = index_q.view(tokens, self.num_heads, self.index_head_dim)
        index_key = index_key.view(tokens, 1, self.index_head_dim)
        compression_logits = compression_logits.view(tokens, 1, self.index_head_dim)
        index_q, index_key = step5_indexer_norm_rope(
            index_q,
            index_key,
            q_weight=self.sparse_indexer_q_norm.weight,
            k_weight=self.sparse_indexer_k_norm.weight,
            k_bias=self.sparse_indexer_k_norm.bias,
            cache=self.rotary_emb.cos_sin_cache_as(torch.bfloat16),
            positions=positions,
            q_out=torch.empty(
                index_q.shape, dtype=index_q.dtype, device=index_q.device
            ),
            k_out=torch.empty_like(index_key),
            eps=self.eps,
            zero_centered=True,
        )
        state = metadata.layer_states[self.layer_id]
        # Read the old ring before publishing this forward's pending inputs.
        compress_index_keys(
            batch=metadata,
            compressed_index_keys=state.compressed_index_keys,
            index_key_ring=state.index_key_ring,
            compression_logits_ring=state.compression_logits_ring,
            positions=state.index_positions,
            index_key=index_key,
            compression_logits=compression_logits,
            assert_success=True,
            enabled=metadata.compression_enabled,
            success=metadata.compression_success,
        )
        return _topk_blocks(
            index_q=index_q,
            index_weights=index_weights,
            compressed_index_keys=state.compressed_index_keys,
            req_to_token=metadata.req_to_token,
            row_to_req=metadata.query_request_slots,
            positions=metadata.query_positions,
            score_width=metadata.score_width,
            workspace=metadata.selection_workspace,
        )


def _topk_blocks(
    *,
    index_q: torch.Tensor,
    index_weights: torch.Tensor,
    compressed_index_keys: torch.Tensor,
    req_to_token: torch.Tensor,
    row_to_req: torch.Tensor,
    positions: torch.Tensor,
    score_width: int,
    workspace: Step5SelectionWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed block IDs, counts and lengths without owning request state.

    Index-key compression precedes this call. Every row sees the same completed
    compressed_index_keys and causal position as the untiled path. Slicing only query rows
    keeps request IDs, logical blocks and subsequent prefill groups unchanged.
    Scratch is reused on the caller's stream; Graph loops depend on shapes only.
    """
    if workspace is None:
        # Standalone kernel callers may allocate their own one-call workspace.
        workspace = allocate_selection_workspace(
            rows=index_q.shape[0],
            score_width=score_width,
            device=index_q.device,
            row_to_req=row_to_req,
            positions=positions,
        )
    packed, counts, lengths = workspace.packed, workspace.counts, workspace.lengths
    indexer_logits, ids = workspace.indexer_logits, workspace.ids
    for start, end in workspace.tiles:
        tile_scores = indexer_logits[: end - start]
        step5_indexer_decode_logits(
            index_q[start:end],
            index_weights[start:end],
            compressed_index_keys=compressed_index_keys,
            req_to_token=req_to_token,
            row_to_req=row_to_req[start:end],
            positions=positions[start:end],
            out=tile_scores,
            clear_future=True,
        )
        if workspace.use_deepselect:
            step5_topk_blocks_deepselect(
                tile_scores,
                positions[start:end],
                row_to_req[start:end],
                packed[start:end],
                counts[start:end],
                req_to_token=req_to_token,
                causal_lengths=lengths[start:end],
                history_lengths=workspace.topk_lengths[start:end],
                topk_ids=workspace.topk_ids[: end - start],
                block_indices=ids[: end - start],
                history_ready=workspace.topk_lengths_ready,
            )
        else:
            tile_ids = ids[: end - start]
            step5_topk_blocks(
                tile_scores,
                positions=positions[start:end],
                row_to_req=row_to_req[start:end],
                block_indices=tile_ids,
                counts=counts[start:end],
            )
            step5_pack_decode_blocks(
                tile_ids,
                counts[start:end],
                req_to_token=req_to_token,
                row_to_req=row_to_req[start:end],
                positions=positions[start:end],
                packed_blocks=packed[start:end],
                causal_lengths=lengths[start:end],
            )
    return packed, counts, lengths
