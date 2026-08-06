"""Layer-wise attention dispatch for dots.note.omni.

Full-attention layers use the DSA backend while sliding-window layers use the
dense FA3 MLA backend for both prefill and decode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    normalize_page_table_rows,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class DotsHybridAttnBackend(AttentionBackend):
    def __init__(
        self,
        dsa_backend: AttentionBackend,
        swa_backend: AttentionBackend,
    ):
        self.dsa_backend = dsa_backend
        # Dots does not implement DeepSeek's prefix-expanding MHA one-shot
        # prepare path. Keep DSA on its radix-aware MLA implementation.
        self.dsa_backend.supports_mha_one_shot = False
        self.swa_backend = swa_backend
        self.token_to_kv_pool = swa_backend.token_to_kv_pool
        self.req_to_token_pool = swa_backend.req_to_token_pool
        # SWA latent expansion builds a compact logical-tail index from the
        # host mirrors during prefill.
        self.needs_cpu_seq_lens = True

    @staticmethod
    def _is_swa_layer(layer: RadixAttention) -> bool:
        return (
            layer.sliding_window_size is not None
            and layer.sliding_window_size > -1
        )

    def backend_for_layer(self, layer: RadixAttention) -> AttentionBackend:
        return self.swa_backend if self._is_swa_layer(layer) else self.dsa_backend

    def selected_swa_backend(self, forward_batch: ForwardBatch) -> AttentionBackend:
        select = getattr(self.swa_backend, "_select_backend", None)
        return (
            select(forward_batch.forward_mode)
            if select is not None
            else self.swa_backend
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.dsa_backend.init_forward_metadata(forward_batch)
        self.swa_backend.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.dsa_backend.init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )
        self.swa_backend.init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.dsa_backend.init_forward_metadata_in_graph(forward_batch)
        self.swa_backend.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.dsa_backend.init_cuda_graph_state(max_bs, max_num_tokens)
        self.swa_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.swa_backend.get_cuda_graph_seq_len_fill_value()

    def on_after_cuda_graph_warmup(self):
        self.dsa_backend.on_after_cuda_graph_warmup()
        self.swa_backend.on_after_cuda_graph_warmup()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self.backend_for_layer(layer).forward(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self.backend_for_layer(layer).forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self.backend_for_layer(layer).forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        return self.dsa_backend.get_indexer_metadata(layer_id, forward_batch)

    def get_swa_mla_prefill_latent_cache(
        self, forward_batch: ForwardBatch, layer_id: int
    ):
        backend = self.selected_swa_backend(forward_batch)
        return backend.get_swa_mla_prefill_latent_cache(forward_batch, layer_id)

    def forward_swa_mla_expanded(self, q, k, v, layer, forward_batch):
        backend = self.selected_swa_backend(forward_batch)
        return backend.forward_swa_mla_expanded(q, k, v, layer, forward_batch)

    def forward_swa_mla_absorbed(self, q, layer, forward_batch):
        """Run decode directly against the page64 latent SWA cache."""
        from sglang.srt.layers.attention.flashmla_ops.flashmla_fallback import (
            forward_dense_kvlora_swa_torch_fallback,
        )

        backend = self.selected_swa_backend(forward_batch)
        metadata = backend.forward_metadata
        block_table = metadata.swa_page_table
        if block_table is None:
            raise RuntimeError("Dots SWA latent decode requires an SWA page table.")
        if backend.page_size != 64:
            raise RuntimeError(
                f"Dots SWA latent decode requires page_size=64, got {backend.page_size}."
            )

        bs = forward_batch.batch_size
        block_table = normalize_page_table_rows(block_table, bs)
        cache_seqlens = metadata.cache_seqlens_int32
        if cache_seqlens.shape[0] != bs:
            # A DP-idle row can be appended after draft metadata is planned.
            # Use its normalized dummy length solely to keep row shapes equal.
            cache_seqlens = forward_batch.seq_lens[:bs].to(torch.int32)
        reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        output = forward_dense_kvlora_swa_torch_fallback(
            reshape_q=reshape_q,
            k_cache=k_cache,
            block_table=block_table,
            # Use the backend's phase-adjusted length: target verify includes
            # every proposed token, while draft step i includes its prefix of
            # speculative tokens.  forward_batch.seq_lens is the unadjusted
            # prefix and shifts both the gather and causal mask backwards.
            cache_seqlens=cache_seqlens,
            layer=layer,
            kv_cache_dim=layer.head_dim,
            head_dim_v=layer.v_head_dim,
            window_size=layer.sliding_window_size + 1,
        )
        return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_mha_chunk_metadata(self, forward_batch: ForwardBatch):
        backend = self.selected_swa_backend(forward_batch)
        init = getattr(backend, "init_mha_chunk_metadata", None)
        if init is not None:
            init(forward_batch)
