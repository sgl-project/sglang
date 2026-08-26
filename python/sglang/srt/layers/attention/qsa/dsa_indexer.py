"""Tokenwise (per-token) QSA indexer for Qwen3Next-DSA models.

Ported from qsa_0511's ``nsa/qwen_indexer.py`` (QwenIndexer) onto this tree's
shared QSA pieces: ``QSAIndexerMetadata`` row semantics, the
``QwenSparseAttnBackend`` sparse-attention kernels and the ``qsa_fast_topk``
primitive.  A tokenwise profile has ``compress_ratio = 1`` and
``block_topk = budget = 2048``; it never consumes the compressed-only MQA
inputs (``get_prefill_mqa_inputs``/``get_decode_mqa_inputs``).

Only the BF16 torch reference compute is ported.  The qsa_0511 FP8
(deep_gemm) and TileLang MQA fast paths are deliberately out of scope here:
requesting them fails loudly instead of silently degrading.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.config import (
    QSA_VARIANT_TOKENWISE,
    parse_qsa_profile,
)
from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
from sglang.srt.layers.attention.qsa.qsa_indexer import _qsa_prefill_row_chunk_size
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import MultiPlatformOp

logger = logging.getLogger(__name__)


def torch_dsa_weighted_mqa_logits(
    q: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    """Lightning-Index scoring reference: ReLU dot-product + per-head weight.

    Args:
        q: ``[rows, n_heads, head_dim]``
        w: ``[rows, n_heads]`` per-head scalar weights
        k: ``[keys, 1, head_dim]`` for shared packed keys, or
            ``[rows, keys, 1, head_dim]`` for per-row keys (paged modes).

    Returns:
        ``[rows, keys]`` float32 logits, divided by ``score_scale``.
    """

    if k.ndim == 4:
        if k.shape[2] != 1 or k.shape[0] != q.shape[0]:
            raise ValueError(
                "tokenwise MQA requires per-row k [rows, keys, 1, head_dim], "
                f"got {k.shape}"
            )
        scores = torch.relu(torch.einsum("mhd,mkhd->mkh", q.float(), k.float()))
    else:
        if k.ndim != 3 or k.shape[1] != 1:
            raise ValueError(
                f"tokenwise MQA requires k [keys, 1, head_dim], got {k.shape}"
            )
        scores = torch.relu(torch.einsum("mhd,khd->mkh", q.float(), k.float()))
    return (scores * w.float().unsqueeze(1)).sum(dim=-1) / score_scale


class QwenDSAIndexer(MultiPlatformOp):
    """Tokenwise Lightning Indexer producing ``[rows, 2048]`` logical indices.

    Input and output contracts match the compressed ``QSAIndexer``:
    ``forward_cuda(hidden_states, positions, forward_batch, indexer_metadata)``
    returns per-query-row logical token indices consumed as ``topk_indices``
    by ``QwenSparseAttnBackend``.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
        page_size: int = 64,
        max_model_len=None,
    ) -> None:
        super().__init__()
        profile = parse_qsa_profile(config)
        if profile is None or profile.variant != QSA_VARIANT_TOKENWISE:
            raise ValueError(
                "QwenDSAIndexer requires a tokenwise QSA config (index_topk/), "
                f"got profile={profile}"
            )
        if page_size != 64:
            # The paged index-K layout and every fast path assume 64-token
            # pages, matching qsa_0511.
            raise ValueError(
                f"tokenwise QSA requires page_size = 64, got {page_size}"
            )
        if envs.SGLANG_QWEN_DSA_USE_FP8_INDEXER.get():
            raise NotImplementedError(
                "The FP8 tokenwise indexer path (qsa_0511 deep_gemm) is not "
                "ported to this tree; unset SGLANG_QWEN_DSA_USE_FP8_INDEXER "
                "to use the BF16 reference path"
            )
        self.qsa_profile = profile
        self.layer_id = int(layer_id)
        self.index_n_heads = profile.n_heads
        self.index_kv_heads = profile.kv_heads
        self.index_head_dim = profile.head_dim
        self.token_topk = profile.budget
        self.score_scale = float(profile.head_dim) ** 0.5
        self.page_size = page_size
        self.max_model_len = max_model_len

        # Fused Q/K/W projection.  Output layout:
        #   q_raw: [M, index_n_heads * index_head_dim]
        #   k_raw: [M, index_kv_heads * index_head_dim]
        #   w:     [M, index_n_heads]  per-head scalar weight
        self.index_q_dim = self.index_n_heads * self.index_head_dim
        self.index_k_dim = self.index_kv_heads * self.index_head_dim
        self.index_w_dim = self.index_n_heads
        self.index_qkw_proj = ReplicatedLinear(
            config.hidden_size,
            self.index_q_dim + self.index_k_dim + self.index_w_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qkw_proj" if prefix else "index_qkw_proj",
        )
        self.index_q_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.index_k_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

        # Independent RoPE for indexer Q/K (qsa_0511): the main attention RoPE
        # is shaped for the main head_dim, so the indexer keeps its own
        # indexer-shaped instance and never touches another model's RoPE
        # internals.  The effective rotary width follows the main attention's.
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = getattr(config, "rope_parameters", None)
        rope_theta = getattr(config, "rope_theta", 10000)
        if isinstance(rope_scaling, dict) and "rope_theta" in rope_scaling:
            rope_theta = rope_scaling["rope_theta"]
        main_head_dim = getattr(config, "head_dim", None)
        if main_head_dim is None:
            main_head_dim = getattr(config, "hidden_size") // getattr(
                config, "num_attention_heads"
            )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
        if partial_rotary_factor is None and isinstance(rope_scaling, dict):
            partial_rotary_factor = rope_scaling.get("partial_rotary_factor")
        if partial_rotary_factor is None:
            partial_rotary_factor = 1.0
        indexer_rotary_dim = min(
            self.index_head_dim, int(main_head_dim * float(partial_rotary_factor))
        )
        if indexer_rotary_dim <= 0 or indexer_rotary_dim % 2 != 0:
            raise ValueError(
                "tokenwise QSA indexer requires a positive even rotary dim, got "
                f"{indexer_rotary_dim=} from {main_head_dim=} and "
                f"{partial_rotary_factor=}"
            )
        self.rotary_emb = get_rope_wrapper(
            head_size=self.index_head_dim,
            rotary_dim=indexer_rotary_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=rope_theta,
            rope_scaling=rope_scaling if isinstance(rope_scaling, dict) else None,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

    def project_qkw(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ):
        """Fused Q/K/W projection, per-head RMS norm and indexer RoPE."""

        qkw, _ = self.index_qkw_proj(hidden_states)
        q_raw, k_raw, w = torch.split(
            qkw, [self.index_q_dim, self.index_k_dim, self.index_w_dim], dim=-1
        )
        q = self.index_q_layernorm(q_raw.reshape(-1, self.index_head_dim)).reshape(
            -1, self.index_n_heads, self.index_head_dim
        )
        k = self.index_k_layernorm(k_raw.reshape(-1, self.index_head_dim)).reshape(
            -1, self.index_kv_heads, self.index_head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, w, k

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
    ) -> torch.Tensor:
        forward_mode = forward_batch.forward_mode
        is_target_verify = getattr(
            forward_mode, "is_target_verify", lambda: False
        )()
        is_draft_extend = getattr(
            forward_mode, "is_draft_extend", lambda **_: False
        )(include_v2=True)
        is_paged = (
            forward_mode.is_decode() or is_target_verify or is_draft_extend
        )
        if is_paged:
            # See the compressed QSAIndexer: speculative/decode rows derive
            # their physical causal length from the paged metadata, not from
            # the model's RoPE coordinate.
            logical_positions = indexer_metadata.get_seqlens_expanded() - 1
        else:
            logical_positions = getattr(forward_batch, "positions", None)
            if logical_positions is None:
                logical_positions = (
                    positions[0] if positions.ndim == 2 else positions
                )
            logical_positions = logical_positions.flatten()

        # DP padding adds token rows without assigning them to a request;
        # token_to_batch_idx is the source of truth for semantic rows, exactly
        # like the compressed indexer.
        num_valid_tokens = indexer_metadata.get_token_to_batch_idx().numel()
        if logical_positions.numel() < num_valid_tokens:
            raise ValueError(
                "tokenwise QSA logical positions are shorter than the request "
                f"mapping: positions={logical_positions.numel()}, "
                f"mapping={num_valid_tokens}"
            )
        if hidden_states.shape[0] < num_valid_tokens:
            raise ValueError(
                "tokenwise QSA hidden states are shorter than the request "
                f"mapping: hidden={hidden_states.shape[0]}, "
                f"mapping={num_valid_tokens}"
            )
        position_tokens = (
            positions.shape[-1] if positions.ndim == 2 else positions.numel()
        )
        if position_tokens < num_valid_tokens:
            raise ValueError(
                "tokenwise QSA RoPE positions are shorter than the request "
                f"mapping: positions={position_tokens}, "
                f"mapping={num_valid_tokens}"
            )

        logical_positions = logical_positions[:num_valid_tokens]
        hidden_states = hidden_states[:num_valid_tokens]
        positions = (
            positions[:, :num_valid_tokens]
            if positions.ndim == 2
            else positions[:num_valid_tokens]
        )
        if num_valid_tokens == 0:
            return torch.empty(
                (0, self.token_topk),
                dtype=torch.int32,
                device=hidden_states.device,
            )

        q, w, k = self.project_qkw(hidden_states, positions)

        pool = indexer_metadata.token_to_kv_pool
        out_cache_loc = getattr(indexer_metadata, "out_cache_loc", None)
        if out_cache_loc is None:
            out_cache_loc = forward_batch.out_cache_loc
        pool.set_dsa_index_k_buffer(
            self.layer_id, out_cache_loc[:num_valid_tokens], k
        )

        if is_paged:
            return self._select_paged(q, w, indexer_metadata)
        return self._select_prefill(q, w, logical_positions, indexer_metadata)

    def _select_paged(
        self,
        q: torch.Tensor,
        w: torch.Tensor,
        indexer_metadata,
    ) -> torch.Tensor:
        """Per-query-row top-k over ``[0, row_len)`` for paged modes."""

        pool = indexer_metadata.token_to_kv_pool
        index_k = pool.get_dsa_index_k_buffer(self.layer_id)
        sequence_lengths = indexer_metadata.sequence_lengths.to(torch.int32)
        table = indexer_metadata.token_slot_table
        rows, max_len = table.shape
        if rows != indexer_metadata.token_to_batch_idx.numel():
            raise ValueError(
                "tokenwise QSA paged modes need one slot-table row per query "
                f"row: table_rows={rows}, "
                f"mapping={indexer_metadata.token_to_batch_idx.numel()}"
            )
        output = torch.full(
            (rows, self.token_topk), -1, dtype=torch.int32, device=q.device
        )
        row_chunk = _qsa_prefill_row_chunk_size(
            rows, max_len, self.index_n_heads
        )
        table_long = table.long()
        for row_start in range(0, rows, row_chunk):
            row_end = min(row_start + row_chunk, rows)
            # Table columns at/after each row's length hold stale slots; the
            # gathers stay in range and fast_topk masks them out by length.
            k_chunk = index_k.index_select(0, table_long[row_start:row_end].reshape(-1))
            k_chunk = k_chunk.reshape(row_end - row_start, max_len, 1, -1)
            logits = torch_dsa_weighted_mqa_logits(
                q[row_start:row_end],
                w[row_start:row_end],
                k_chunk,
                self.score_scale,
            )
            lengths = sequence_lengths[row_start:row_end]
            selected = qsa_fast_topk(
                logits,
                torch.zeros_like(lengths),
                lengths.clamp(min=0, max=max_len),
                topk=self.token_topk,
            )
            output[row_start:row_end].copy_(selected)
        return output

    def _select_prefill(
        self,
        q: torch.Tensor,
        w: torch.Tensor,
        logical_positions: torch.Tensor,
        indexer_metadata,
    ) -> torch.Tensor:
        """Packed per-sequence top-k with causal windows for extend modes."""

        pool = indexer_metadata.token_to_kv_pool
        index_k = pool.get_dsa_index_k_buffer(self.layer_id)
        sequence_lengths = indexer_metadata.sequence_lengths.to(torch.int32)
        table = indexer_metadata.token_slot_table
        query_sequence_ids = indexer_metadata.token_to_batch_idx.long()
        row_ends_all = (logical_positions.to(torch.int32) + 1).clamp(
            min=0, max=table.shape[1]
        )
        rows = q.shape[0]
        output = torch.full(
            (rows, self.token_topk), -1, dtype=torch.int32, device=q.device
        )
        for sequence_id in range(sequence_lengths.numel()):
            seq_len = int(sequence_lengths[sequence_id].item())
            row_mask = query_sequence_ids == sequence_id
            if seq_len <= 0 or not bool(row_mask.any()):
                continue
            row_indices = row_mask.nonzero(as_tuple=True)[0]
            slots = table[sequence_id, :seq_len].long()
            k_seq = index_k.index_select(0, slots)
            row_chunk = _qsa_prefill_row_chunk_size(
                row_indices.numel(), seq_len, self.index_n_heads
            )
            for chunk_start in range(0, row_indices.numel(), row_chunk):
                chunk_rows = row_indices[chunk_start : chunk_start + row_chunk]
                row_ends = row_ends_all.index_select(0, chunk_rows)
                logits = torch_dsa_weighted_mqa_logits(
                    q.index_select(0, chunk_rows),
                    w.index_select(0, chunk_rows),
                    k_seq,
                    self.score_scale,
                )
                selected = qsa_fast_topk(
                    logits,
                    torch.zeros_like(row_ends),
                    row_ends,
                    topk=self.token_topk,
                )
                # Tensor indexing returns a copy on read; use index_put style
                # assignment or the selection would never reach `output`.
                output[chunk_rows] = selected
        return output


__all__ = [
    "QwenDSAIndexer",
    "torch_dsa_weighted_mqa_logits",
]
