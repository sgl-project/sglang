"""Hunyuan-4-Flash.

Hy4-Flash keeps Hy4's iHC residual streams, gated MLA and learnable sinks, and
adds:

* hybrid attention: sliding-window layers (64 heads, window 512) between DSA
  layers; both run sparse MLA, the window layers over their causal window;
* the coarse-only IndexPool indexer on the DSA layers: one pooled FP8 key per
  group of ``compress_ratio`` tokens, top groups plus the query's own group;
* Engram n-gram embeddings added to the residual streams of a few layers;
* latent routed experts and sandwich norms after every attention and FFN block.
"""

import logging
from copy import copy
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import triton
from torch import nn

from sglang.kernels.ops.attention.dsa import hy4_flash_indexpool as indexpool_kernels
from sglang.kernels.ops.embeddings.hy4_flash_engram import (
    hy4_flash_engram_combine,
    hy4_flash_engram_lookup,
)
from sglang.kernels.ops.layernorm.hy4_ihc import (
    fused_hy4_ihc_post,
    hy4_ihc_post_pre_norm,
    ihc_fuses_out_norm,
)
from sglang.srt.configs.hy_v4_flash import (
    SLIDING_ATTENTION,
    HYV4FlashConfig,
    HYV4FlashTextConfig,
)
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.engram import EngramHasher
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.layers.layer_boundary.adapters.attention import AttentionInputs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    prepare_mm_inputs,
)
from sglang.srt.managers.req_types import (
    MM_PAD_SHIFT_VALUE,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, MoEGate
from sglang.srt.models.hunyuan_v4 import (
    HYV4Attention,
    HYV4HCHeadLayer,
    HYV4HCLayer,
)
from sglang.srt.models.hunyuan_v4_vl import HYV4VisionTransformer
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import BumpAllocator

logger = logging.getLogger(__name__)

INDEX_DIM = 128
_DIRECT_WEIGHT_UPDATE_ERROR = (
    "load_format='direct' does not refresh the Hy4-Flash fused IndexPool and "
    "Engram weights; update weights with the default load format."
)


def _maybe_all_reduce(x: torch.Tensor) -> torch.Tensor:
    if get_parallel().tp_size > 1:
        return tensor_model_parallel_all_reduce(x)
    return x


class _FlashMRotaryEmbedding(nn.Module):
    def __init__(self, rope, mrope):
        super().__init__()
        self.rope = rope
        self.mrope = mrope

    @property
    def cos_sin_cache(self):
        return self.rope.cos_sin_cache

    @property
    def is_neox_style(self):
        return self.rope.is_neox_style

    def forward(self, positions, query, key):
        if positions.ndim == 1:
            return self.rope(positions, query, key)
        query_out, key_out = self.mrope(positions, query.flatten(1), key.flatten(1))
        return query_out.view_as(query), key_out.view_as(key)


def _build_mrope(rope_dim, max_position, theta, section, interleaved, neox_style):
    return get_rope(
        rope_dim,
        rotary_dim=rope_dim,
        max_position=max_position,
        base=theta,
        is_neox_style=neox_style,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": list(section),
            "mrope_interleaved": interleaved,
        },
    )


class _MRopePositionStore:
    def __init__(self, section):
        self.section = tuple(section)
        self.buffer: Optional[torch.Tensor] = None

    def write(self, positions: torch.Tensor, out_cache_loc: torch.Tensor):
        if self.buffer is None:
            pool = get_token_to_kv_pool()
            slots = pool.size + 2 * pool.page_size + 1
            self.buffer = torch.zeros(
                (slots, 3), dtype=torch.int32, device=positions.device
            )
        rows = min(out_cache_loc.shape[0], positions.shape[-1])
        if positions.ndim == 2:
            xyz = positions[:, :rows].t()
        else:
            xyz = positions[:rows, None].expand(-1, 3)
        slots = layout_clamp(out_cache_loc[:rows].to(torch.int64), self.buffer.shape[0])
        self.buffer[slots] = xyz.to(torch.int32)


# ---------------------------------------------------------------------------
# Sparse index selection shared by the window and IndexPool indexers.
# ---------------------------------------------------------------------------


class _RowLayout:
    """Per query row: owning batch entry and logical position."""

    def __init__(self, forward_batch: ForwardBatch, metadata, num_rows: int):
        mode = forward_batch.forward_mode
        device = forward_batch.seq_lens.device
        if mode.is_decode_or_idle():
            lengths = metadata.get_seqlens_int32()[:num_rows].to(torch.int64)
            row_to_batch = torch.arange(num_rows, device=device)
        else:
            lengths = metadata.get_seqlens_expanded()[:num_rows].to(torch.int64)
            row_to_batch = torch.repeat_interleave(
                torch.arange(forward_batch.batch_size, device=device),
                forward_batch.extend_seq_lens.to(torch.int64),
                output_size=num_rows,
            )
        self.lengths = lengths
        self.positions = (lengths - 1).contiguous()
        self.seq_lens_cpu = forward_batch.seq_lens_cpu
        self.row_to_batch = row_to_batch
        self.metadata = metadata
        self.num_rows = num_rows

    @property
    def max_tokens(self) -> int:
        """Token columns the page tables can address for every row."""
        table = self.metadata.get_page_table_1()
        if table is not None:
            return table.shape[1]
        return self.metadata.get_page_table_64().shape[1] * 64

    def slots(self, logical: torch.Tensor, rows: Optional[torch.Tensor] = None):
        """Physical KV slots of logical token positions (int64, clamped in range).

        Decode CUDA graphs drop the page_size=1 table, so fall back to the
        64-token page table."""
        if rows is None:
            rows = self.row_to_batch
        rows = rows.reshape(rows.shape + (1,) * (logical.ndim - rows.ndim))
        safe = logical.clamp_min(0)
        table = self.metadata.get_page_table_1()
        if table is not None:
            return table[rows, safe.clamp_max(table.shape[1] - 1)].to(torch.int64)
        pages = self.metadata.get_page_table_64()
        page = (safe // 64).clamp_max(pages.shape[1] - 1)
        return pages[rows, page].to(torch.int64) * 64 + safe % 64


def _row_layout(forward_batch: ForwardBatch, metadata, num_rows: int) -> _RowLayout:
    """The row layout is the same for every layer of a forward pass; the model
    resets `forward_batch.hy4f_layouts` at the start of each pass."""
    cache = forward_batch.hy4f_layouts
    key = (id(metadata.attn_metadata), num_rows)
    layout = cache.get(key)
    if layout is None:
        layout = cache[key] = _RowLayout(forward_batch, metadata, num_rows)
    return layout


def _ragged_offsets(layout: "_RowLayout") -> Optional[torch.Tensor]:
    if layout.metadata.topk_transform_method != TopkTransformMethod.RAGGED:
        return None
    offsets = layout.metadata.attn_metadata.topk_indices_offset[: layout.num_rows]
    return offsets.to(torch.int64).contiguous()


class FlashWindowIndexer(nn.Module):
    """Selects the causal window of a sliding-window layer."""

    def __init__(self, window: int, index_topk: int):
        super().__init__()
        self.window = window
        self.index_topk = index_topk

    def forward(
        self, x, q_lora, positions, forward_batch, layer_id, return_indices=True
    ):
        num_rows = q_lora.shape[0]
        if forward_batch.forward_mode.is_idle() or num_rows == 0:
            return q_lora.new_full((num_rows, self.index_topk), -1, dtype=torch.int32)
        metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
        layout = _row_layout(forward_batch, metadata, num_rows)
        return indexpool_kernels.window_indices(
            layout.positions.contiguous(),
            metadata.get_page_table_64(),
            layout.row_to_batch.contiguous(),
            _ragged_offsets(layout),
            self.window,
            self.index_topk,
        )


class FlashIndexPool(nn.Module):
    """Coarse-only gated IndexPool.

    Per DSA layer it keeps a raw per-token key (bf16) and stores each completed
    group's pooled FP8 key in the DSA index-key cache at the slot of the group's
    first token.
    """

    # Bytes of fp32 logits one scoring chunk may hold.
    _LOGITS_BUDGET = 512 << 20

    def __init__(
        self,
        config: HYV4FlashTextConfig,
        layer_id: int,
        prefix: str,
        rope_store: Optional[_MRopePositionStore] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.rope_store = rope_store
        self.mrope_section = rope_store.section if rope_store is not None else None
        self.ratio = config.compressed_indexer_compress_ratio
        if self.ratio != 4:
            raise ValueError("The Hy4-Flash IndexPool kernels assume 4-token groups")
        self.group_budget = config.compressed_indexer_group_topk - 1
        self.index_topk = config.index_topk
        self.num_heads = config.compressed_indexer_num_attention_heads
        self.eps = config.rms_norm_eps
        theta = config.rope_for_layer_type("deepseek_sparse_attention")["rope_theta"]
        rope_dim = config.qk_rope_head_dim
        self.register_buffer(
            "inv_freq",
            theta ** (-torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim),
            persistent=False,
        )
        self.query_projection = ReplicatedLinear(
            config.q_lora_rank,
            self.num_heads * INDEX_DIM,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.query_projection",
        )
        self.head_weight_projection = ReplicatedLinear(
            config.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.head_weight_projection",
        )
        self.token_key_projection = ReplicatedLinear(
            config.hidden_size,
            INDEX_DIM,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.token_key_projection",
        )
        # A plain bf16 nn.RMSNorm, as in the reference, so the raw keys round the
        # same way.
        self.token_key_norm = nn.RMSNorm(INDEX_DIM, eps=self.eps, dtype=torch.bfloat16)
        self.key_value_projection = nn.Linear(
            INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32
        )
        self.key_gating = nn.Linear(
            INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32
        )
        self.key_norm = nn.Parameter(torch.ones(INDEX_DIM, dtype=torch.float32))
        self.key_norm.weight_loader = self._copy_loader
        self.intra_group_ape = nn.Parameter(
            torch.empty(self.ratio, INDEX_DIM, dtype=torch.float32)
        )
        self.intra_group_ape.weight_loader = self._copy_loader
        for linear in (self.key_value_projection, self.key_gating):
            linear.weight.weight_loader = self._copy_loader
        self._raw_keys: Optional[torch.Tensor] = None
        self._weights_t = None
        self._key_head_weight = None
        self._direct_weight_update_error = _DIRECT_WEIGHT_UPDATE_ERROR

    @staticmethod
    def _copy_loader(param, loaded_weight):
        param.data.copy_(loaded_weight.to(param.dtype))

    def _raw_key_buffer(self, device) -> torch.Tensor:
        if self._raw_keys is None:
            pool = get_token_to_kv_pool()
            slots = pool.size + 2 * pool.page_size + 1
            self._raw_keys = torch.zeros(
                (slots, INDEX_DIM), dtype=torch.bfloat16, device=device
            )
        return self._raw_keys

    def refresh_derived_weights(self):
        if self._weights_t is not None:
            self._weights_t[0].copy_(self.key_value_projection.weight.t())
            self._weights_t[1].copy_(self.key_gating.weight.t())
        if self._key_head_weight is not None:
            torch.cat(
                [self.token_key_projection.weight, self.head_weight_projection.weight],
                out=self._key_head_weight,
            )

    def _transposed_weights(self):
        if self._weights_t is None:
            self._weights_t = (
                self.key_value_projection.weight.t().contiguous(),
                self.key_gating.weight.t().contiguous(),
            )
        return self._weights_t

    def _project_key_and_head_weights(self, x):
        """Raw key projection [R, 128] and head weights [R, H] from one GEMM."""
        if self._key_head_weight is None:
            self._key_head_weight = torch.cat(
                [
                    self.token_key_projection.weight,
                    self.head_weight_projection.weight,
                ]
            ).contiguous()
        out = F.linear(x, self._key_head_weight)
        return out[:, :INDEX_DIM], out[:, INDEX_DIM:]

    def _index_buffer(self):
        pool = get_token_to_kv_pool()
        return pool.get_index_k_with_scale_buffer(
            layer_id=self.layer_id
        ), pool.index_page_size

    def _pool_groups(self, members, first_positions, write_slots):
        raw_buf = self._raw_keys
        buf, page_size = self._index_buffer()
        indexpool_kernels.indexpool_pool_write(
            raw_buf,
            layout_clamp(members, raw_buf.shape[0]),
            first_positions,
            write_slots,
            *self._transposed_weights(),
            self.intra_group_ape,
            self.key_norm,
            self.eps,
            self.inv_freq,
            buf,
            page_size,
            mrope_section=self.mrope_section,
        )

    def _decode_update_cache(self, raw_key_proj, forward_batch, layout: "_RowLayout"):
        """Decode: one kernel normalizes and stores the raw keys and pools the
        groups this step completes."""
        raw_buf = self._raw_key_buffer(raw_key_proj.device)
        buf, page_size = self._index_buffer()
        rows = raw_key_proj.shape[0]
        indexpool_kernels.indexpool_decode_update(
            raw_key_proj,
            self.token_key_norm.weight,
            self.token_key_norm.eps,
            forward_batch.out_cache_loc[:rows],
            layout.metadata.get_seqlens_int32()[:rows],
            layout.metadata.get_page_table_64(),
            raw_buf,
            *self._transposed_weights(),
            self.intra_group_ape,
            self.key_norm,
            self.eps,
            self.inv_freq,
            buf,
            page_size,
            rope_xyz=self.rope_store.buffer if self.rope_store is not None else None,
            mrope_section=self.mrope_section,
        )

    def _update_cache(self, keys, forward_batch, layout: "_RowLayout"):
        raw_buf = self._raw_key_buffer(keys.device)
        out_loc = forward_batch.out_cache_loc[: keys.shape[0]].to(torch.int64)
        raw_buf[layout_clamp(out_loc, raw_buf.shape[0])] = keys
        ratio = self.ratio
        device = keys.device
        ends = forward_batch.seq_lens.to(torch.int64)
        starts = ends - forward_batch.extend_seq_lens.to(torch.int64)
        first_group = starts // ratio
        counts = (ends // ratio - first_group).clamp_min(0)
        seq_cpu, ext_cpu = forward_batch.seq_lens_cpu, forward_batch.extend_seq_lens_cpu
        if seq_cpu is not None and ext_cpu is not None:
            # Size the work from host copies: a device sync here would stall the
            # launch queue once per DSA layer.
            total = sum(
                max(e // ratio - (e - n) // ratio, 0)
                for e, n in zip(seq_cpu.tolist(), list(ext_cpu))
            )
        else:
            total = int(counts.sum().item())
        if total == 0:
            return
        batch = torch.repeat_interleave(
            torch.arange(counts.shape[0], device=device), counts, output_size=total
        )
        prefix = torch.cumsum(counts, 0) - counts
        group = first_group[batch] + torch.arange(total, device=device) - prefix[batch]
        members = group[:, None] * ratio + torch.arange(ratio, device=device)
        member_slots = layout.slots(members, batch)
        first_positions = members[:, 0]
        if self.rope_store is not None:
            first_positions = self.rope_store.buffer[member_slots[:, 0]]
        self._pool_groups(member_slots, first_positions, member_slots[:, 0])

    def _select(self, layout: "_RowLayout", q_fp8, weights, decode: bool):
        """Top groups strictly before each query's group, expanded to tokens, plus
        the query's own group up to the query, packed at the front."""
        positions = layout.positions
        if decode:
            num_groups = layout.max_tokens // self.ratio
        else:
            seq_cpu = layout.seq_lens_cpu
            if seq_cpu is not None and len(seq_cpu):
                num_groups = max((int(seq_cpu.max()) - 1) // self.ratio, 1)
            else:
                num_groups = max(int(positions.max().item()) // self.ratio, 1)
        # flashinfer's top-k needs aligned rows; padded columns score -inf.
        num_groups = triton.cdiv(num_groups, 128) * 128
        pages = layout.metadata.get_page_table_64()
        row_to_batch = layout.row_to_batch.contiguous()
        buf, page_size = self._index_buffer()
        count = min(self.group_budget, num_groups)
        rows = positions.shape[0]
        chunk = max(1, self._LOGITS_BUDGET // (4 * num_groups)) if not decode else rows

        def score(begin, end):
            logits = indexpool_kernels.indexpool_logits(
                q_fp8[begin:end],
                weights[begin:end],
                pages,
                row_to_batch[begin:end],
                positions[begin:end],
                buf,
                page_size,
                num_groups,
            )
            return _topk(logits, count)

        if chunk >= rows:
            values, picked = score(0, rows)
        else:
            picked = torch.empty(
                (rows, count), dtype=torch.int64, device=positions.device
            )
            values = torch.empty(
                (rows, count), dtype=torch.float32, device=positions.device
            )
            for begin in range(0, rows, chunk):
                end = min(rows, begin + chunk)
                values[begin:end], picked[begin:end] = score(begin, end)
        return indexpool_kernels.indexpool_expand(
            picked,
            values,
            positions,
            pages,
            row_to_batch,
            _ragged_offsets(layout),
            self.index_topk,
        )

    def _query_rope_positions(self, positions, layout: "_RowLayout"):
        if self.rope_store is None:
            return layout.positions
        rows = layout.num_rows
        if positions.ndim == 2:
            return positions[:, :rows].t()
        return positions[:rows]

    def forward(
        self, x, q_lora, positions, forward_batch, layer_id, return_indices=True
    ):
        num_rows = q_lora.shape[0]
        if forward_batch.forward_mode.is_idle() or num_rows == 0:
            return q_lora.new_full((num_rows, self.index_topk), -1, dtype=torch.int32)
        metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
        layout = _row_layout(forward_batch, metadata, num_rows)
        raw_key_proj, head_weights = self._project_key_and_head_weights(x)
        if forward_batch.forward_mode.is_decode():
            self._decode_update_cache(raw_key_proj, forward_batch, layout)
        else:
            keys = self.token_key_norm(raw_key_proj)
            self._update_cache(keys, forward_batch, layout)
        q_fp8, weights = indexpool_kernels.indexpool_query(
            self.query_projection(q_lora)[0],
            self._query_rope_positions(positions, layout),
            head_weights,
            self.inv_freq,
            self.num_heads,
            mrope_section=self.mrope_section,
        )
        return self._select(
            layout, q_fp8, weights, decode=forward_batch.forward_mode.is_decode()
        )


def _topk(logits: torch.Tensor, k: int):
    """Unordered top-k (values, indices); -inf values mark non-candidates."""
    if logits.shape[1] <= k:
        idx = torch.arange(logits.shape[1], device=logits.device)
        return logits, idx.expand(logits.shape[0], -1).contiguous()
    import flashinfer

    return flashinfer.top_k(logits, k)


def layout_clamp(slots: torch.Tensor, size: int) -> torch.Tensor:
    """Page-table entries past a row's length can be stale or unset."""
    return slots.clamp(0, size - 1)


# ---------------------------------------------------------------------------
# Engram
# ---------------------------------------------------------------------------


def _engram_head_range(sizes, tp, rank):
    """Contiguous hash heads per TP rank, balanced by table rows."""
    if not 1 <= tp <= len(sizes):
        raise ValueError("Engram needs at least one hash head per TP rank")
    prefix = np.cumsum([0, *sizes]).tolist()
    boundaries = [0]
    for r in range(1, tp):
        target = prefix[-1] * r / tp
        candidates = range(boundaries[-1] + 1, len(sizes) - (tp - r) + 1)
        boundaries.append(min(candidates, key=lambda i: abs(prefix[i] - target)))
    boundaries.append(len(sizes))
    return boundaries[rank], boundaries[rank + 1]


class FlashEngramHasher(EngramHasher):
    """N-gram hash ids [T, n_engram_layers, n_hash_cols] for a forward batch.

    Subclasses EngramHasher so the runner sizes its per-request history and the
    scheduler stages the predecessors of each extend (prefix hits, slot reuse).
    The token map comes from the checkpoint's ``lookup_table`` instead of the
    tokenizer, and the primes from ``engram_layer_head_vocab_sizes``.
    """

    def __init__(self, config: HYV4FlashTextConfig):
        nn.Module.__init__(self)
        self.layer_ids = tuple(config.engram_layer_ids)
        self.max_ngram_size = config.engram_max_ngram_size
        self.vocab_size = config.vocab_size
        self.seed = config.engram_seed
        self.pad_token_id = config.engram_pad_id
        groups = config.engram_layer_head_vocab_sizes
        primes = [groups.get(str(layer), groups.get(layer)) for layer in self.layer_ids]
        flat = [[p for order in group for p in order] for group in primes]
        offsets = np.array(
            [np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64
        )
        self.register_buffer(
            "primes", torch.tensor(primes, dtype=torch.int64), persistent=False
        )
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer(
            "token_map",
            torch.arange(self.vocab_size, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            torch.ones(len(self.layer_ids), self.max_ngram_size, dtype=torch.int64),
            persistent=False,
        )
        self.pad_id = 0
        self.history: Optional[torch.Tensor] = None
        self.pad_row = 0
        self.visual_token_ids: tuple[int, ...] = ()
        self._cpu_hash_params = None
        self._loaded = False

    def load_lookup(self, value: torch.Tensor) -> None:
        if self._loaded:
            return
        lookup = value.detach().cpu().to(torch.int64).numpy()
        compressed_vocab = int(lookup.max()) + 1
        token_map = lookup.tolist()
        token_map.extend(range(len(token_map), self.vocab_size))
        bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
        multipliers = np.stack(
            [
                np.random.default_rng(self.seed + 10007 * layer).integers(
                    0, bound, size=self.max_ngram_size, dtype=np.int64
                )
                * 2
                + 1
                for layer in self.layer_ids
            ]
        )
        self.token_map.copy_(torch.tensor(token_map, dtype=torch.int64))
        self.multipliers.copy_(torch.tensor(multipliers))
        self.pad_id = token_map[self.pad_token_id]
        self._loaded = True

    def _ensure_history(self, device):
        if self.history is None:
            from sglang.srt.model_executor.forward_context import get_req_to_token_pool

            self.init_history(get_req_to_token_pool().size + 1, device)
        return self.history

    def _restore_visual_ids(self, tokens, mm_pad_ids):
        pads = tokens >= MM_PAD_SHIFT_VALUE
        restored = torch.full_like(tokens, self.visual_token_ids[0])
        if mm_pad_ids is not None:
            pad_values, token_ids = mm_pad_ids
            match = tokens[..., None] == pad_values
            first = match.int().argmax(-1)
            restored = torch.where(match.any(-1), token_ids[first], restored)
        return torch.where(pads, restored, tokens)

    def forward(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        mm_pad_ids: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        n = self.max_ngram_size
        device = input_ids.device
        history = self._ensure_history(device)
        num_tokens = input_ids.shape[0]
        mode = forward_batch.forward_mode
        req = forward_batch.req_pool_indices.to(torch.int64)
        ids = input_ids.to(torch.int64)
        if mode.is_decode_or_idle():
            # history row holds the n - 1 previous tokens, oldest first.
            past = history[req[:num_tokens]].to(torch.int64)
            tokens = torch.cat([ids[:, None], past.flip(-1)], -1)
            positions = forward_batch.positions[:num_tokens].to(torch.int64)
            row = torch.arange(num_tokens, device=device)
            commit_rows, commit_last = req[:num_tokens], row
        else:
            token_index = torch.arange(num_tokens, device=device)
            lens = forward_batch.extend_seq_lens.to(torch.int64)
            starts = torch.cumsum(lens, 0) - lens
            row = torch.repeat_interleave(
                torch.arange(lens.shape[0], device=device),
                lens,
                output_size=num_tokens,
            )
            offset = token_index - starts[row]
            staged = forward_batch.engram_history
            past = staged if staged is not None else history[req]
            shifts = torch.arange(n, device=device)
            in_batch = shifts[None] <= offset[:, None]
            from_batch = ids[(token_index[:, None] - shifts).clamp_min(0)]
            hist_col = (n - 1 - (shifts[None] - offset[:, None])).clamp(0, n - 2)
            from_hist = past.to(torch.int64)[row].gather(1, hist_col)
            tokens = torch.where(in_batch, from_batch, from_hist)
            positions = forward_batch.positions[:num_tokens].to(torch.int64)
            commit_rows = req
            commit_last = (starts + lens - 1).clamp_min(0)
        if self.visual_token_ids and not mode.is_decode_or_idle():
            tokens = self._restore_visual_ids(tokens, mm_pad_ids)
        shifts = torch.arange(n, device=device)
        blocked = positions[:, None] < shifts
        compressed = torch.where(blocked, self.pad_id, self.token_map[tokens])
        products = compressed[:, None, :] * self.multipliers[None]
        rolling, hashes = products[..., 0], []
        for i in range(1, n):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self.primes[None, :, i - 1])
        hash_ids = torch.cat(hashes, -1) + self.offsets[None]
        no_hash = input_ids == self.pad_token_id
        if self.visual_token_ids:
            for token_id in self.visual_token_ids:
                no_hash |= input_ids == token_id
        hash_ids = hash_ids.masked_fill(no_hash[:, None, None], -1)
        if num_tokens:
            last = tokens[commit_last]
            # out_cache_loc 0 marks padded rows that must not commit.
            out_loc = forward_batch.out_cache_loc
            if out_loc is not None:
                commit_rows = torch.where(
                    out_loc[commit_last] == 0, self.pad_row, commit_rows
                )
            history[commit_rows] = last[:, : n - 1].flip(-1).to(history.dtype)
        return hash_ids


class FlashEngramEmbedding(nn.Module):
    def __init__(self, sizes, dim: int, fp8: bool):
        super().__init__()
        tp, rank = get_parallel().tp_size, get_parallel().tp_rank
        self.head_start, self.head_end = _engram_head_range(sizes, tp, rank)
        self.vocab_start = sum(sizes[: self.head_start])
        self.vocab_end = sum(sizes[: self.head_end])
        self.num_embeddings = sum(sizes)
        self.dim = dim
        self.rows = self.vocab_end - self.vocab_start
        # The table dtype (bf16, or fp8 with per-row bf16 scales) is taken from
        # the checkpoint when it loads.
        self.weight = nn.Parameter(
            torch.empty(0, dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(0, dtype=torch.bfloat16), requires_grad=False
        )

    def load(self, leaf: str, value: torch.Tensor) -> None:
        if value.shape[0] != self.num_embeddings:
            raise ValueError(f"Invalid Engram table rows: {tuple(value.shape)}")
        param = getattr(self, leaf)
        param.data = torch.empty(
            (self.rows, *value.shape[1:]), dtype=value.dtype, device=param.device
        )
        param.data.copy_(value[self.vocab_start : self.vocab_end])

    def forward(self, hashes: torch.Tensor) -> torch.Tensor:
        local = hashes[:, self.head_start : self.head_end].to(torch.int64)
        if self.weight.numel() == 0:
            # Dummy weights never materialize the table.
            return hashes.new_zeros((*local.shape, self.dim), dtype=torch.float32)
        rows = (local - self.vocab_start).clamp(0, self.rows - 1)
        out = self.weight[rows].float()
        if self.weight_scale_inv.numel():
            out = out * self.weight_scale_inv[rows].float()[..., None]
        return out


class FlashEngram(nn.Module):
    def __init__(self, config: HYV4FlashTextConfig, layer_number: int, prefix: str):
        super().__init__()
        groups = config.engram_layer_head_vocab_sizes
        group = groups.get(str(layer_number), groups.get(layer_number))
        sizes = [p for order in group for p in order]
        heads = config.engram_n_head_per_ngram
        self.head_dim = config.engram_n_embed_per_ngram // heads
        self.hidden_size, self.hc_mult = config.hidden_size, config.hc_mult
        self.multi_head_embedding = FlashEngramEmbedding(
            sizes, self.head_dim, config.engram_fp8_embedding
        )
        local = (
            self.multi_head_embedding.head_end - self.multi_head_embedding.head_start
        )
        self.local_heads = local
        self.total_width = len(sizes) * self.head_dim
        self.start = self.multi_head_embedding.head_start * self.head_dim
        self.stop = self.multi_head_embedding.head_end * self.head_dim
        self.rank = get_parallel().tp_rank
        self.lookup_gamma = nn.Parameter(
            torch.ones(local, self.head_dim, dtype=torch.bfloat16)
        )
        self.value_proj = nn.Linear(
            local * self.head_dim, self.hidden_size, dtype=torch.bfloat16
        )
        self.key_projs = nn.ModuleList(
            nn.Linear(local * self.head_dim, self.hidden_size, dtype=torch.bfloat16)
            for _ in range(self.hc_mult)
        )
        self.norm_key = nn.Parameter(
            torch.ones(self.hc_mult, self.hidden_size, dtype=torch.bfloat16)
        )
        self.norm_query = nn.Parameter(
            torch.ones(self.hc_mult, self.hidden_size, dtype=torch.bfloat16)
        )
        self.unified_pre_gamma = nn.Parameter(
            torch.ones(self.hidden_size, dtype=torch.bfloat16)
        )
        self.unified_per_hc_scales = nn.Parameter(
            torch.zeros(self.hc_mult, self.hidden_size, dtype=torch.bfloat16)
        )
        for projection in [self.value_proj, *self.key_projs]:
            projection.weight.weight_loader = self._load_projection
            projection.bias.weight_loader = self._load_bias
        self._fused_kv = None
        self._direct_weight_update_error = _DIRECT_WEIGHT_UPDATE_ERROR

    def refresh_derived_weights(self):
        if self._fused_kv is not None:
            weight, bias = self._fused_kv
            torch.cat(
                [p.weight for p in self.key_projs] + [self.value_proj.weight],
                out=weight,
            )
            torch.cat(
                [p.bias for p in self.key_projs] + [self.value_proj.bias], out=bias
            )

    def _load_projection(self, param, value):
        if value.shape != (self.hidden_size, self.total_width):
            raise ValueError("Engram projection weight shape mismatch")
        param.data.copy_(value[:, self.start : self.stop])

    def _load_bias(self, param, value):
        param.data.copy_(value if self.rank == 0 else torch.zeros_like(value))

    def load_leaf(self, leaf: str, value: torch.Tensor) -> bool:
        """Checkpoint leaves stored per head or per hc channel."""
        parts = leaf.split(".")
        if parts[0] == "lookup_per_head_norms":
            head = int(parts[1]) - self.multi_head_embedding.head_start
            if 0 <= head < self.local_heads:
                self.lookup_gamma.data[head].copy_(value)
            return True
        if parts[0] in ("norm_key", "norm_query"):
            getattr(self, parts[0]).data[int(parts[1])].copy_(value)
            return True
        if parts[0] == "unified_per_hc_scales":
            self.unified_per_hc_scales.data[int(parts[1])].copy_(value)
            return True
        if parts[0] == "unified_pre_gamma":
            self.unified_pre_gamma.data.copy_(value)
            return True
        return False

    def forward(self, hidden: torch.Tensor, hashes: torch.Tensor) -> torch.Tensor:
        """hidden [T, hc_mult, dim]; hashes [T, n_hash_cols] of this layer."""
        if hidden.shape[0] == 0:
            return hidden
        table = self.multi_head_embedding
        if table.weight.numel() and hidden.is_cuda:
            x = hy4_flash_engram_lookup(
                hashes,
                table.head_start,
                self.local_heads,
                table.vocab_start,
                table.weight,
                table.weight_scale_inv,
                self.lookup_gamma,
            )
        else:
            x = self._lookup_eager(hashes, hidden.dtype)
        # The 4 key projections and the value projection share their input: one
        # GEMM and one all-reduce for all of them.
        if self._fused_kv is None:
            self._fused_kv = (
                torch.cat(
                    [p.weight for p in self.key_projs] + [self.value_proj.weight]
                ),
                torch.cat([p.bias for p in self.key_projs] + [self.value_proj.bias]),
            )
        kv = _maybe_all_reduce(F.linear(x, *self._fused_kv))
        if hidden.is_cuda:
            return hy4_flash_engram_combine(
                kv,
                hidden,
                hashes,
                self.norm_key,
                self.norm_query,
                self.unified_pre_gamma,
                self.unified_per_hc_scales,
            )
        return self._combine_eager(kv, hidden, hashes)

    def _lookup_eager(self, hashes, dtype):
        rows = self.multi_head_embedding(hashes)
        eps = torch.finfo(torch.float32).eps
        table = self.multi_head_embedding
        local = hashes[:, table.head_start : table.head_end]
        x = rows.masked_fill(local.unsqueeze(-1) < 0, 0)
        x = (
            x
            * (x.square().mean(-1, keepdim=True) + eps).rsqrt()
            * self.lookup_gamma.float()
        ).to(dtype)
        return x.flatten(1)

    def _combine_eager(self, kv, hidden, hashes):
        eps = torch.finfo(torch.float32).eps
        bf16_eps = torch.finfo(torch.bfloat16).eps
        keys = kv.view(-1, self.hc_mult + 1, self.hidden_size)
        gates = []
        for channel in range(self.hc_mult):
            key = keys[:, channel]
            gate = (
                F.rms_norm(key, (self.hidden_size,), self.norm_key[channel], bf16_eps)
                * F.rms_norm(
                    hidden[:, channel],
                    (self.hidden_size,),
                    self.norm_query[channel],
                    bf16_eps,
                )
            ).sum(-1)
            gate = gate * self.hidden_size**-0.5
            gates.append((gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid())
        gate = torch.stack(gates, dim=1)
        value = keys[:, self.hc_mult].float()
        signal = (
            self.unified_pre_gamma.float()
            * value
            * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
        )
        result = (
            (gate.float().unsqueeze(-1) + self.unified_per_hc_scales.float())
            * signal.unsqueeze(1)
        ).to(hidden.dtype)
        return hidden + result.masked_fill((hashes < 0).all(-1)[:, None, None], 0)


# ---------------------------------------------------------------------------
# Attention, FFN and the decoder layer
# ---------------------------------------------------------------------------


class HYV4FlashAttention(HYV4Attention):
    def __init__(
        self,
        config: HYV4FlashTextConfig,
        layer_id,
        quant_config,
        prefix,
        alt_stream,
        rope_store: Optional[_MRopePositionStore] = None,
    ):
        self.is_window_layer = config.layer_types[layer_id] == SLIDING_ATTENTION
        layer_config = copy(config)
        layer_config.rope_parameters = dict(
            config.rope_for_layer_type(config.layer_types[layer_id])
        )
        if self.is_window_layer:
            layer_config.num_attention_heads = config.swa_num_attention_heads
            layer_config.qk_nope_head_dim = config.swa_qk_nope_head_dim
            layer_config.qk_rope_head_dim = config.swa_qk_rope_head_dim
            layer_config.v_head_dim = config.swa_v_head_dim
        super().__init__(layer_config, layer_id, quant_config, prefix, alt_stream)
        if rope_store is not None:
            self.rotary_emb = _FlashMRotaryEmbedding(
                self.rotary_emb,
                _build_mrope(
                    rope_dim=layer_config.qk_rope_head_dim,
                    max_position=config.max_position_embeddings,
                    theta=layer_config.rope_parameters["rope_theta"],
                    section=rope_store.section,
                    interleaved=config.mrope_interleaved,
                    neox_style=self.rotary_emb.is_neox_style,
                ),
            )
        # The sandwich norm needs the full sum before the post-projection norm.
        self.o_proj.reduce_results = True
        if self.is_window_layer:
            # The window holds at most `sliding_window` keys; a selection that
            # narrow lets the sparse kernel stop scanning there.
            width = min(config.index_topk, triton.cdiv(config.sliding_window, 64) * 64)
            self.indexer = FlashWindowIndexer(config.sliding_window, width)
        else:
            self.indexer = FlashIndexPool(
                config, layer_id, f"{prefix}.indexer", rope_store=rope_store
            )
        self.post_proj_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            and (self.is_window_layer or not config.dsa_skip_attn_post_norm)
            else None
        )

    def should_run_indexer(self, prev_topk_indices=None) -> bool:
        return True

    @property
    def out_norm(self):
        return self.post_proj_norm

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        apply_out_norm=True,
        **kwargs,
    ):
        out = super().forward(positions, hidden_states, forward_batch, zero_allocator)
        if isinstance(out, tuple):
            out = out[0]
        if apply_out_norm and self.post_proj_norm is not None and out.shape[0] > 0:
            out = self.post_proj_norm(out)
        return out


class HYV4FlashDenseMLP(DeepseekV2MLP):
    def __init__(self, config, quant_config, prefix):
        super().__init__(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=prefix,
            reduce_results=True,
        )
        self.post_fc2_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    @property
    def out_norm(self):
        return self.post_fc2_norm

    def forward(self, x, forward_batch=None, apply_out_norm=True):
        out = super().forward(x)
        if apply_out_norm and self.post_fc2_norm is not None and out.shape[0] > 0:
            out = self.post_fc2_norm(out)
        return out


class HYV4FlashMoE(nn.Module):
    """Router and shared expert on the full hidden; routed experts on a latent."""

    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.layer_id = layer_id
        self.routed_scaling_factor = config.routed_scaling_factor
        latent = config.routed_expert_hidden_size or config.hidden_size
        self.latent = latent
        # The checkpoint router is bf16: a bf16 x bf16 -> fp32 GEMM gives the
        # reference's fp32 logits up to summation order, without upcasting x.
        gate_config = copy(config)
        gate_config.router_fp32 = False
        self.gate = MoEGate(
            config=gate_config, quant_config=None, prefix=f"{prefix}.gate"
        )
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts,
            num_fused_shared_experts=0,
            top_k=config.num_experts_per_tok,
            hidden_size=latent,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            swiglu_limit=config.swiglu_limit,
            prefix=f"{prefix}.experts",
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
        )
        self.topk.bind_runner_backend(
            self.experts.runner_backend,
            routing_method_type=self.experts.routing_method_type,
        )
        self.routed_latent_down = None
        self.routed_latent_up = None
        if latent != config.hidden_size:
            self.routed_latent_down = ReplicatedLinear(
                config.hidden_size,
                latent,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_latent_down",
            )
            self.routed_latent_up = ReplicatedLinear(
                latent,
                config.hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_latent_up",
            )
        self.shared_experts = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            quant_config=None,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )
        self.combine_output_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    @property
    def out_norm(self):
        return self.combine_output_norm

    def forward(self, hidden_states, forward_batch=None, apply_out_norm=True):
        if hidden_states.shape[0] == 0:
            return hidden_states
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        routed_input = hidden_states
        if self.routed_latent_down is not None:
            routed_input = self.routed_latent_down(hidden_states)[0]
        routed = self.experts(routed_input, topk_output)
        # The replicated up projection is linear, so project each rank's partial
        # latent sum and reduce it together with the shared expert's partial
        # output: one all-reduce instead of two.
        if self.routed_latent_up is not None:
            routed = self.routed_latent_up(routed)[0]
        out = _maybe_all_reduce(routed + self.shared_experts(hidden_states))
        if apply_out_norm and self.combine_output_norm is not None:
            out = self.combine_output_norm(out)
        return out


class HYV4FlashDecoderLayer(nn.Module):
    def __init__(
        self, config, layer_id, quant_config, prefix, alt_stream, rope_store=None
    ):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = HYV4FlashAttention(
            config,
            layer_id,
            quant_config,
            f"{prefix}.self_attn",
            alt_stream,
            rope_store=rope_store,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.mlp_layer_types[layer_id] == "dense":
            self.mlp = HYV4FlashDenseMLP(config, None, f"{prefix}.mlp")
        else:
            self.mlp = HYV4FlashMoE(config, layer_id, quant_config, f"{prefix}.mlp")
        self.hc_attn_layer = HYV4HCLayer(config, f"{prefix}.hc_attn_layer")
        self.hc_mlp_layer = HYV4HCLayer(config, f"{prefix}.hc_mlp_layer")
        self.engram_layer: Optional[FlashEngram] = None

    @staticmethod
    def _ihc_pre(hc_layer, norm, residual, output=None, post=None, out_norm=None):
        """iHC pre and the block's input RMSNorm, with the previous block's
        output RMSNorm (`out_norm`) and iHC post folded in when given."""
        p = hc_layer.hc_pre
        return hy4_ihc_post_pre_norm(
            output,
            residual,
            post,
            p.hc_fn.weight,
            p.hc_scale,
            p.hc_base,
            p.magnitude,
            p.rms_norm_eps,
            p.hc_eps,
            norm.weight,
            norm.variance_epsilon,
            out_norm.weight if out_norm is not None else None,
            out_norm.variance_epsilon if out_norm is not None else 0.0,
        )

    @staticmethod
    def _finish_post(residual, pending):
        output, post, out_norm = pending
        if out_norm is not None:
            output = out_norm(output)
        return fused_hy4_ihc_post(output, residual, post)

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        engram_hashes=None,
        pending=None,
    ):
        """`pending` is the previous layer's (mlp output, post gates, deferred out
        norm); its iHC post is fused into this layer's iHC pre when possible."""
        defer_norm = ihc_fuses_out_norm(hidden_states.shape[0])
        if pending is not None and self.engram_layer is not None:
            hidden_states = self._finish_post(hidden_states, pending)
            pending = None
        if pending is None:
            residual = self.hc_attn_layer.prepare_input(hidden_states)
            if self.engram_layer is not None:
                residual = self.engram_layer(residual, engram_hashes)
            residual, x, post = self._ihc_pre(
                self.hc_attn_layer, self.input_layernorm, residual
            )
        else:
            residual, x, post = self._ihc_pre(
                self.hc_attn_layer, self.input_layernorm, hidden_states, *pending
            )
        context = get_attn_tp_context()
        context.set_attn_inputs(
            AttentionInputs(x, forward_batch, self.self_attn.prepare_qkv_latent)
        )
        try:
            x = self.self_attn(
                positions,
                x,
                forward_batch,
                zero_allocator,
                apply_out_norm=not defer_norm,
            )
        finally:
            context.clear_attn_inputs()
        residual, x, post = self._ihc_pre(
            self.hc_mlp_layer,
            self.post_attention_layernorm,
            residual,
            x,
            post,
            self.self_attn.out_norm if defer_norm else None,
        )
        x = self.mlp(x, forward_batch, apply_out_norm=not defer_norm)
        return residual, (x, post, self.mlp.out_norm if defer_norm else None)


class HYV4FlashModel(nn.Module):
    def __init__(
        self,
        config: HYV4FlashTextConfig,
        quant_config=None,
        prefix="",
        visual_token_ids: Optional[tuple[int, int]] = None,
    ):
        super().__init__()
        if get_parallel().pp_group.world_size != 1:
            raise ValueError("Hy4-Flash pipeline parallelism is not supported")
        self.config = config
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
            **get_embedding_tp_kwargs(),
        )
        self.alt_stream = None
        self.rope_store = (
            _MRopePositionStore(config.mrope_section)
            if visual_token_ids is not None
            else None
        )
        self.layers = nn.ModuleList(
            HYV4FlashDecoderLayer(
                config,
                i,
                quant_config,
                f"{prefix}.layers.{i}",
                self.alt_stream,
                rope_store=self.rope_store,
            )
            for i in range(config.num_hidden_layers)
        )
        self.engram_hasher = None
        self.engram_index = {}
        if config.engram_layer_ids:
            self.engram_hasher = FlashEngramHasher(config)
            if visual_token_ids is not None:
                self.engram_hasher.visual_token_ids = visual_token_ids
            for index, layer_number in enumerate(config.engram_layer_ids):
                layer = self.layers[layer_number - 1]
                layer.engram_layer = FlashEngram(
                    config,
                    layer_number,
                    f"{prefix}.layers.{layer_number - 1}.engram_layer",
                )
                self.engram_index[layer_number - 1] = index
        self.hc_head = HYV4HCHeadLayer(config, f"{prefix}.hc_head")
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, input_ids, positions, forward_batch, input_embeds=None, mm_pad_ids=None
    ):
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if self.rope_store is not None and not forward_batch.forward_mode.is_idle():
            self.rope_store.write(positions, forward_batch.out_cache_loc)
        hashes = None
        if self.engram_hasher is not None:
            hashes = self.engram_hasher(input_ids, forward_batch, mm_pad_ids)
        forward_batch.hy4f_layouts = {}
        zero_allocator = BumpAllocator(
            buffer_size=2 * len(self.layers),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        pending = None
        for i, layer in enumerate(self.layers):
            layer_hashes = None
            if i in self.engram_index:
                layer_hashes = hashes[:, self.engram_index[i]]
            hidden_states, pending = layer(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                layer_hashes,
                pending,
            )
        hidden_states = HYV4FlashDecoderLayer._finish_post(hidden_states, pending)
        return self.hc_head(hidden_states, self.norm)


_RENAMED_WEIGHTS = (
    (".mlp.routed_expert_down_proj.", ".mlp.routed_latent_down."),
    (".mlp.routed_expert_up_proj.", ".mlp.routed_latent_up."),
    (".indexer.key_gate_projection.", ".indexer.key_gating."),
)


class HYV4FlashForConditionalGeneration(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        return "Hy4-Flash shared experts run on the full hidden, routed on a latent."

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        text_config = getattr(config, "text_config", config)
        self.config = text_config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.num_fused_shared_experts = 0
        self.visual = None
        self.visual_token_ids = None
        if (
            isinstance(config, HYV4FlashConfig)
            and config.vision_config is not None
            and not config.language_model_only
        ):
            if text_config.mrope_section is None:
                raise ValueError("Hy4-Flash vision serving requires mrope_section")
            self.visual = HYV4VisionTransformer(
                config.vision_config, quant_config=None, prefix="visual"
            )
            self.visual_token_ids = (config.image_token_id, config.video_token_id)
        self.is_mrope_enabled = self.visual is not None
        self.model = HYV4FlashModel(
            text_config,
            quant_config,
            "model",
            visual_token_ids=self.visual_token_ids,
        )
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=None,
            prefix="lm_head",
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(text_config)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        mm_pad_ids = None
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions
            if forward_batch.forward_mode.is_decode():
                positions = positions[0]
        if self.is_mrope_enabled and input_embeds is None:
            mm_pad_ids = self._mm_pad_ids(forward_batch, input_ids.device)
            input_embeds = prepare_mm_inputs(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self,
                kwargs={},
                multimodal_model=self,
            )
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, mm_pad_ids=mm_pad_ids
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def _mm_pad_ids(self, forward_batch, device):
        if forward_batch.forward_mode.is_decode() or not (
            forward_batch.contains_mm_inputs()
        ):
            return None
        image_id, video_id = self.visual_token_ids
        token_of_pad = {}
        for mm_input in forward_batch.mm_inputs:
            if mm_input is None:
                continue
            for item in mm_input.mm_items:
                token_id = video_id if item.modality == Modality.VIDEO else image_id
                for pad in item.padding_values():
                    token_of_pad.setdefault(pad, token_id)
        return (
            torch.tensor(list(token_of_pad), dtype=torch.int64, device=device),
            torch.tensor(list(token_of_pad.values()), dtype=torch.int64, device=device),
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def pad_input_ids(self, input_ids, mm_inputs: MultimodalInputs):
        if mm_inputs and mm_inputs.mm_items and self.visual is None:
            raise RuntimeError(
                "Hy4-Flash was loaded language_model_only; multimodal inputs are "
                "not supported."
            )
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._get_visual_feature(items, "image_grid_thw")

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._get_visual_feature(items, "video_grid_thw")

    def _get_visual_feature(self, items, grid_key: str) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        grid_thw = torch.cat([getattr(item, grid_key) for item in items], dim=0)
        return torch.cat(self.visual(pixel_values, grid_thw.tolist()), dim=0)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        model = self.model
        vision_weights = []

        def mapped_weights():
            for name, loaded_weight in weights:
                if name.startswith("vit."):
                    if self.visual is not None:
                        vision_weights.append(
                            (name.removeprefix("vit."), loaded_weight)
                        )
                    continue
                if name.startswith(("model.dfly_layers.", "model.mtp_layers.")):
                    continue
                if ".engram_layer." in name:
                    self._load_engram(name, loaded_weight)
                    continue
                name = name.replace(".compressed_indexer.", ".indexer.")
                # The DeepSeek loader rewrites any "gate_proj"/"up_proj" substring
                # as a stacked gate_up_proj shard; keep these names clear of it.
                for old, new in _RENAMED_WEIGHTS:
                    name = name.replace(old, new)
                if name.endswith(".indexer.key_norm.weight"):
                    name = name.removesuffix(".weight")
                if name.endswith((".hc_fn", ".hc_head_fn")):
                    name += ".weight"
                if name.endswith(
                    (".experts.gate_up_proj_scale_inv", ".experts.down_proj_scale_inv")
                ):
                    name = name.removesuffix("_inv")
                elif name.endswith(".weight_scale"):
                    name += "_inv"
                yield name, loaded_weight

        self.do_load_weights(mapped_weights())
        for module in model.modules():
            if isinstance(module, (FlashIndexPool, FlashEngram)):
                module.refresh_derived_weights()
        if self.visual is not None:
            self.visual.load_weights(vision_weights)
        if model.engram_hasher is not None and not model.engram_hasher._loaded:
            raise ValueError("Hy4-Flash checkpoint is missing the Engram lookup_table")

    def _load_engram(self, name: str, value: torch.Tensor):
        prefix, leaf = name.split(".engram_layer.", 1)
        layer_id = int(prefix.rsplit(".", 1)[1])
        engram = self.model.layers[layer_id].engram_layer
        if engram is None:
            raise ValueError(f"Undeclared Engram checkpoint weight: {name}")
        if leaf == "lookup_table":
            self.model.engram_hasher.load_lookup(value)
            return
        if leaf.startswith("multi_head_embedding.embedding."):
            engram.multi_head_embedding.load(leaf.rsplit(".", 1)[1], value)
            return
        if engram.load_leaf(leaf, value):
            return
        param = dict(engram.named_parameters())[leaf]
        loader = getattr(param, "weight_loader", None)
        if loader is None:
            param.data.copy_(value)
        else:
            loader(param, value)


EntryClass = [HYV4FlashForConditionalGeneration]
