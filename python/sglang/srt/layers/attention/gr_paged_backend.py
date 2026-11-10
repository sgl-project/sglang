"""Eager GR attention over paged context and a group-owned beam arena.

The _prefill_attention/_decode_attention methods are the operation boundary for
a future kernel. Their defaults use the torch_*_attention functions below.
Context plus selected ancestry is one softmax domain; scratch is bounded by a
query block for prefill and one beam for decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, MHATokenToKVPool

if TYPE_CHECKING:
    from sglang.srt.beam_search.paged_kv import GRPagedForwardMetadata


@dataclass(frozen=True)
class _PrefillMetadata:
    context_pages: torch.Tensor
    context_len: int
    prefix_len: int


def _context_locations(pages, length, page_size):
    positions = torch.arange(length, device=pages.device, dtype=torch.int64)
    return pages[positions // page_size].long() * page_size + positions % page_size


def _torch_attention(q, k, v, scale, *, query_offset=None, query_block_size=32):
    """FP32 QK/softmax/PV with local-head GQA and model-dtype output.

    Shapes are tokens x local heads x head dimension. Processing one KV head's
    GQA group at a time keeps GEMM geometry independent of TP size.
    query_offset supplies absolute causal positions for prefix extension;
    None means all supplied keys are legal (ancestry-filtered beam decode).
    """
    q_heads, kv_heads = q.shape[1], k.shape[1]
    if q_heads % kv_heads or k.shape[1] != v.shape[1]:
        raise ValueError("gr_paged requires integral local Q/KV head grouping")
    groups = q_heads // kv_heads
    result = q.new_empty((q.shape[0], q_heads, v.shape[-1]))
    key_positions = torch.arange(k.shape[0], device=q.device)
    for head in range(kv_heads):
        key = k[:, head].float().transpose(0, 1)
        value = v[:, head].float()
        head_slice = slice(head * groups, (head + 1) * groups)
        for start in range(0, q.shape[0], query_block_size):
            end = min(start + query_block_size, q.shape[0])
            query = q[start:end, head_slice].float().transpose(0, 1)
            logits = torch.matmul(query, key) * scale
            if query_offset is not None:
                query_positions = torch.arange(
                    query_offset + start, query_offset + end, device=q.device
                )
                logits.masked_fill_(
                    key_positions[None, :] > query_positions[:, None], -torch.inf
                )
            output = torch.matmul(torch.softmax(logits, dim=-1), value)
            result[start:end, head_slice] = output.transpose(0, 1).to(q.dtype)
    return result


def torch_prefill_attention(
    q, key_pool, value_pool, context_pages, context_len, page_size, prefix_len, scale
):
    """Read paged context in logical order and apply offset causality."""
    locations = _context_locations(context_pages, context_len, page_size)
    return _torch_attention(
        q, key_pool[locations], value_pool[locations], scale, query_offset=prefix_len
    )


def torch_decode_attention(
    q, key_pool, value_pool, metadata: GRPagedForwardMetadata, scale
):
    """Read context + ancestry, including this step's just-written token.

    A native replacement can consume the same context/beam pages and
    topk_indices directly; this POC gathers exactly those legal keys.
    """
    context_locations = _context_locations(
        metadata.context_pages, metadata.context_len, metadata.page_size
    )
    context_k = key_pool[context_locations].float()
    context_v = value_pool[context_locations].float()
    beam_locations = metadata.beam_locations()
    result = q.new_empty((q.shape[0], q.shape[1], value_pool.shape[-1]))
    for beam in range(metadata.beam_width):
        key = torch.cat((context_k, key_pool[beam_locations[beam]].float()), dim=0)
        value = torch.cat((context_v, value_pool[beam_locations[beam]].float()), dim=0)
        result[beam : beam + 1] = _torch_attention(
            q[beam : beam + 1], key, value, scale
        )
    return result


def validate_gr_paged_configuration(config):
    """One startup guard for the explicitly bounded eager GR POC."""
    required = {
        "enable_gr": True,
        "attention_backend": "gr_paged",
        "disable_overlap_schedule": True,
        "chunked_prefill_size": -1,
        "cuda_graph_backend_decode": "disabled",
        "cuda_graph_backend_prefill": "disabled",
    }
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError(f"gr_paged requires {name}={expected!r}")
    if config.get("tp_size", 1) not in (1, 2, 4):
        raise ValueError("gr_paged currently supports TP=1, 2 or 4")
    if not isinstance(config.get("page_size"), int) or config["page_size"] <= 0:
        raise ValueError("gr_paged requires a positive integer page_size")
    for name in ("pp_size", "dp_size", "attn_dp_size", "attn_cp_size", "dcp_size"):
        if config.get(name, 1) != 1:
            raise ValueError(f"gr_paged requires {name}=1")
    for name in (
        "quantization",
        "speculative_algorithm",
        "enable_torch_compile",
        "enable_lora",
        "enable_layernorm_sp",
        "enable_hierarchical_cache",
        "enable_unified_memory",
        "enable_deterministic_inference",
        "enable_page_major_kv_layout",
        "enable_session_radix_cache",
        "enable_unified_cache_external_linker",
        "enable_prefill_cp",
        "enable_mixed_chunk",
        "enable_priority_scheduling",
    ):
        if config.get(name):
            raise ValueError(f"gr_paged does not support {name}")
    for name in ("prefill_attention_backend", "decode_attention_backend"):
        if config.get(name) not in (None, "gr_paged"):
            raise ValueError(f"gr_paged requires {name}=gr_paged")


class GRPagedAttentionBackend(AttentionBackend):
    """Ordinary SGLang backend; owns compute scratch, never KV allocations."""

    def __init__(self, runner):
        super().__init__()
        config = runner.server_args.resolved_dict()
        validate_gr_paged_configuration(config)
        self.req_to_token_pool = runner.req_to_token_pool
        self.token_to_kv_pool = runner.token_to_kv_pool
        self.page_size = runner.page_size
        self.data_type = runner.dtype
        self.kv_cache_dtype = runner.kv_cache_dtype
        if self.data_type != torch.bfloat16 or self.kv_cache_dtype != torch.bfloat16:
            raise ValueError("gr_paged currently requires BF16 model and KV")
        if (
            type(self.token_to_kv_pool) is not MHATokenToKVPool
            or self.token_to_kv_pool.kv_cache_layout != "nhd"
        ):
            raise ValueError("gr_paged requires the plain static NHD MHA KV pool")
        if config.get("tp_size", 1) > runner.model_config.get_num_kv_heads(1):
            raise ValueError("gr_paged requires head-sharded KV (TP <= KV heads)")
        if self.token_to_kv_pool.page_size != self.page_size:
            raise ValueError("gr_paged backend and KV pool page sizes must agree")
        torch.backends.cuda.matmul.allow_tf32 = False

    def init_forward_metadata(self, batch):
        if batch.forward_mode.is_decode():
            meta = batch.gr_paged_metadata
            if meta is None or meta.page_size != self.page_size:
                raise ValueError("gr_paged decode requires matching GR page metadata")
            self.forward_metadata = meta
        elif batch.forward_mode.is_extend() and not batch.forward_mode.is_mixed():
            if len(batch.extend_seq_lens_cpu) != 1:
                raise ValueError("gr_paged supports one admitted GR group per batch")
            prefix = batch.extend_prefix_lens_cpu[0]
            context_len = prefix + batch.extend_seq_lens_cpu[0]
            if context_len < 1 or batch.extend_seq_lens_cpu[0] < 1:
                raise ValueError("gr_paged requires a nonempty prefill suffix")
            context_locs = self.req_to_token_pool.req_to_token[
                batch.req_pool_indices[0], :context_len
            ]
            pages = (context_locs[:: self.page_size] // self.page_size).clone()
            self.forward_metadata = _PrefillMetadata(pages, context_len, prefix)
        elif not batch.forward_mode.is_idle():
            raise ValueError("gr_paged supports eager prefill and decode only")

    def _write_kv(self, k, v, layer, batch, save_kv_cache):
        if layer.is_cross_attention or layer.logit_cap or layer.sliding_window_size > 0:
            raise ValueError("gr_paged requires ordinary cached self-attention Q/K/V")
        # Native Qwen can fuse RoPE with the write to batch.out_cache_loc.
        # In that case both prefill and decode read the already populated pool.
        if save_kv_cache:
            if k is None or v is None:
                raise ValueError("gr_paged KV writes require both K and V")
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc.for_layer(batch, layer), k, v
            )

    def forward_extend(self, q, k, v, layer, batch, save_kv_cache=True, **kwargs):
        self._write_kv(k, v, layer, batch, save_kv_cache)
        key_pool, value_pool = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        query = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        output = self._prefill_attention(
            query, key_pool, value_pool, self.forward_metadata, layer.scaling
        )
        return output.reshape(q.shape[0], -1)

    def _prefill_attention(self, q, key_pool, value_pool, meta, scale):
        return torch_prefill_attention(
            q,
            key_pool,
            value_pool,
            meta.context_pages,
            meta.context_len,
            self.page_size,
            meta.prefix_len,
            scale,
        )

    def forward_decode(self, q, k, v, layer, batch, save_kv_cache=True, **kwargs):
        self._write_kv(k, v, layer, batch, save_kv_cache)
        key_pool, value_pool = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        query = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        output = self._decode_attention(
            query, key_pool, value_pool, self.forward_metadata, layer.scaling
        )
        return output.reshape(q.shape[0], -1)

    def _decode_attention(self, q, key_pool, value_pool, meta, scale):
        return torch_decode_attention(q, key_pool, value_pool, meta, scale)
