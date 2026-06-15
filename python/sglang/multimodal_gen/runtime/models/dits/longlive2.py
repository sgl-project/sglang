# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask

from sglang.multimodal_gen.configs.models.dits.longlive2 import LongLive2VideoConfig
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.dits.causal_wanvideo import (
    CausalWanTransformer3DModel,
    CausalWanTransformerBlock,
)


class LongLive2CausalWanTransformerBlock(CausalWanTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = LayerNormScaleShift(
            self.hidden_dim,
            eps=self.norm1.eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )

    def _cross_attn_with_cache(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        crossattn_cache: CrossAttentionKVCache | None,
    ) -> torch.Tensor:
        attn2 = self.attn2
        q, _ = attn2.to_q(hidden_states)
        if attn2.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, attn2.norm_q)
        else:
            q = attn2.norm_q(q)
        q = q.unflatten(2, (attn2.local_num_heads, attn2.head_dim))

        if crossattn_cache is not None and crossattn_cache.is_init:
            k = crossattn_cache.k
            v = crossattn_cache.v
        else:
            k, _ = attn2.to_k(encoder_hidden_states)
            if attn2.tp_rmsnorm:
                k = tensor_parallel_rms_norm(k, attn2.norm_k)
            else:
                k = attn2.norm_k(k)
            k = k.unflatten(2, (attn2.local_num_heads, attn2.head_dim))

            v, _ = attn2.to_v(encoder_hidden_states)
            v = v.unflatten(2, (attn2.local_num_heads, attn2.head_dim))

            if crossattn_cache is not None:
                crossattn_cache.store(k, v)

        hidden_states = attn2.attn(q, k, v)
        hidden_states = hidden_states.flatten(2)
        hidden_states, _ = attn2.to_out(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
        kv_cache: CausalSelfAttentionKVCache | None = None,
        crossattn_cache: CrossAttentionKVCache | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        num_frames = temb.shape[1]
        bs, _, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        e = self.scale_shift_table + temb.float()
        assert e.shape == (bs, num_frames, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=2
        )
        assert shift_msa.dtype == torch.float32

        norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output = self.attn1(
            query,
            key,
            value,
            freqs_cis,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.zeros(
            (1,), device=hidden_states.device, dtype=hidden_states.dtype
        )
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        attn_output = self._cross_attn_with_cache(
            norm_hidden_states,
            encoder_hidden_states,
            crossattn_cache,
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(ff_output, c_gate_msa, hidden_states)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class LongLive2Transformer3DModel(CausalWanTransformer3DModel):
    _fsdp_shard_conditions = LongLive2VideoConfig()._fsdp_shard_conditions
    _compile_conditions = LongLive2VideoConfig()._compile_conditions
    _supported_attention_backends = LongLive2VideoConfig()._supported_attention_backends
    param_names_mapping = LongLive2VideoConfig().param_names_mapping
    reverse_param_names_mapping = LongLive2VideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = LongLive2VideoConfig().lora_param_names_mapping

    def __init__(
        self,
        config: LongLive2VideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.blocks = nn.ModuleList(
            [
                LongLive2CausalWanTransformerBlock(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.local_attn_size,
                    config.sink_size,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                    self._supported_attention_backends,
                    prefix=f"{config.prefix}.blocks.{i}",
                    quant_config=quant_config,
                )
                for i in range(config.num_layers)
            ]
        )


EntryClass = LongLive2Transformer3DModel
