"""BF16 paged MHA and GQA attention for Cambricon MLUs.

Uses ragged flash attention for extend and cached single-query attention for
decode. Mixed batches preserve the shared scheduler's prefill-then-decode order;
the boundary comes from scheduler request identity, so even a one-token prefill
uses the prefill kernel.
"""

from typing import TYPE_CHECKING

import torch
import torch_mlu_ops

from sglang.srt.hardware_backend.mlu.attention.metadata import (
    ForwardMetadata,
    build_forward_metadata,
)
from sglang.srt.hardware_backend.mlu.attention.utils import validate_attention
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class MLUMHABackend(AttentionBackend):
    def __init__(self, model_runner: "ModelRunner") -> None:
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.forward_metadata = None

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        if forward_batch.forward_mode.is_idle():
            return
        self.forward_metadata = build_forward_metadata(
            forward_batch,
            self.req_to_token,
            self.page_size,
            self.max_context_len,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        validate_attention(layer, kwargs.get("sinks"))
        return_lse = kwargs.get("return_lse", False)
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
            return self._forward_decode(
                q, layer, forward_metadata, return_lse=return_lse
            )
        if 0 < forward_metadata.mixed_prefill_reqs < forward_batch.batch_size:
            out = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
            split = forward_metadata.mixed_prefill_tokens
            prefill_result = self._forward_prefill(
                q[:split],
                k[:split] if k is not None else None,
                v[:split] if v is not None else None,
                layer,
                forward_metadata,
                out=out[:split],
                return_lse=return_lse,
            )
            decode_result = self._forward_decode(
                q[split:],
                layer,
                forward_metadata,
                forward_metadata.mixed_prefill_reqs,
                out=out[split:],
                return_lse=return_lse,
            )
            if return_lse:
                return out, torch.cat((prefill_result[1], decode_result[1]), dim=0)
            return out
        return self._forward_prefill(
            q, k, v, layer, forward_metadata, return_lse=return_lse
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: "RadixAttention",
        forward_metadata: ForwardMetadata,
        out: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
            return_lse=return_lse,
            block_tables=block_tables,
            out_dtype=q.dtype,
            window_size_left=window if window is not None and window > 0 else -1,
            window_size_right=0 if window is not None and window > 0 else -1,
        )
        if return_lse:
            out, lse = out
            # Convert [num_heads, num_tokens] to [num_tokens, num_heads].
            return out.flatten(1), lse.transpose(0, 1)
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        validate_attention(layer, kwargs.get("sinks"))
        return_lse = kwargs.get("return_lse", False)
        forward_metadata = self.forward_metadata
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            )
        return self._forward_decode(q, layer, forward_metadata, return_lse=return_lse)

    def _forward_decode(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_metadata: ForwardMetadata,
        start_req: int = 0,
        out: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
            return_lse=return_lse,
            kv_cache_quant_bit_size=-1,
            head_size_v=layer.v_head_dim,
            compute_dtype=torch.float16,
            q_quant_scale=None,
            out_quant_scale=None,
        )
        if return_lse:
            out, lse = out
            # Each request has one query: [batch_size, num_heads, 1] -> [batch_size, num_heads].
            return out.reshape(q.shape[0], -1), lse.squeeze(-1)
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
