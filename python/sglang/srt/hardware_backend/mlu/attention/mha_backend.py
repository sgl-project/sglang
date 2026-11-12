"""BF16 paged GQA attention for Cambricon MLUs.

Uses ragged flash attention for extend and cached single-query attention for
decode. Mixed batches preserve the shared scheduler's prefill-then-decode order;
the boundary comes from scheduler request identity, so even a one-token prefill
uses the prefill kernel.
"""

import torch
import torch_mlu_ops

from sglang.srt.hardware_backend.mlu.attention.metadata import build_forward_metadata
from sglang.srt.hardware_backend.mlu.attention.utils import validate_attention
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


class MLUMHABackend(AttentionBackend):
    def __init__(self, model_runner):
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.forward_metadata = None

    def init_forward_metadata(self, forward_batch):
        if forward_batch.forward_mode.is_idle():
            return
        self.forward_metadata = build_forward_metadata(
            forward_batch,
            self.req_to_token,
            self.page_size,
            self.max_context_len,
        )

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        validate_attention(layer, kwargs.get("sinks"), kwargs.get("return_lse", False))
        forward_metadata = self.forward_metadata
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            )
        if forward_metadata.mixed_prefill_reqs == 0:
            return self._forward_decode(q, layer, forward_metadata)
        if 0 < forward_metadata.mixed_prefill_reqs < forward_batch.batch_size:
            out = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
            split = forward_metadata.mixed_prefill_tokens
            self._forward_prefill(
                q[:split],
                k[:split] if k is not None else None,
                v[:split] if v is not None else None,
                layer,
                forward_metadata,
                out=out[:split],
            )
            self._forward_decode(
                q[split:],
                layer,
                forward_metadata,
                forward_metadata.mixed_prefill_reqs,
                out=out[split:],
            )
            return out
        return self._forward_prefill(q, k, v, layer, forward_metadata)

    def _forward_prefill(self, q, k, v, layer, forward_metadata, out=None):
        if forward_metadata.is_uncached_prefill and k is not None and v is not None:
            k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
            v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
            block_tables = None
        else:
            k = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
            count = forward_metadata.mixed_prefill_reqs
            block_tables = (
                forward_metadata.block_tables[:count]
                if count >= 0
                else forward_metadata.block_tables
            )
        window = layer.sliding_window_size
        out = torch_mlu_ops.flash_attention(
            q=q,
            k=k,
            v=v,
            out=out.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            if out is not None
            else None,
            cu_seq_lens_q=forward_metadata.cu_seqlens_q,
            cu_seq_lens_kv=forward_metadata.cu_seqlens_kv,
            alibi_slope=None,
            attn_bias=None,
            max_seq_len_q=forward_metadata.max_seq_len_q,
            max_seq_len_kv=forward_metadata.max_seq_len_kv,
            softmax_scale=layer.scaling,
            is_causal=True,
            compute_dtype=torch.float16,
            return_lse=False,
            block_tables=block_tables,
            out_dtype=q.dtype,
            window_size_left=window if window is not None and window > 0 else -1,
            window_size_right=0 if window is not None and window > 0 else -1,
        )
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        validate_attention(layer, kwargs.get("sinks"), kwargs.get("return_lse", False))
        forward_metadata = self.forward_metadata
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            )
        return self._forward_decode(q, layer, forward_metadata)

    def _forward_decode(self, q, layer, forward_metadata, start_req=0, out=None):
        q = q.view(-1, 1, layer.tp_q_head_num, layer.qk_head_dim)
        window = layer.sliding_window_size
        out = torch_mlu_ops.single_query_cached_kv_attn(
            q=q,
            k_cache=self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_cache=self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            out=(
                out.view(-1, 1, layer.tp_q_head_num, layer.v_head_dim)
                if out is not None
                else q.new_empty((q.shape[0], 1, layer.tp_q_head_num, layer.v_head_dim))
            ),
            block_tables=forward_metadata.block_tables[start_req:],
            context_lens=forward_metadata.seq_lens[start_req:],
            k_cache_quant_scale=None,
            v_cache_quant_scale=None,
            alibi_slopes=None,
            max_context_len=self.max_context_len,
            windows_size_left=window if window is not None and window > 0 else -1,
            windows_size_right=-1,
            softmax_scale=layer.scaling,
            kv_cache_quant_bit_size=-1,
            head_size_v=layer.v_head_dim,
            compute_dtype=torch.float16,
            q_quant_scale=None,
            out_quant_scale=None,
        )
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
