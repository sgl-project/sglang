"""Inference-only SmolLM3 model compatible with HuggingFace weights."""

from typing import Optional

from transformers import SmolLM3Config

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from sglang.srt.utils import add_prefix, make_layers


class SmolLM3Attention(LlamaAttention):
    """Llama attention, minus RoPE on the layers config.no_rope_layers marks as NoPE."""

    def __init__(
        self, *, config: SmolLM3Config, layer_id: int, start_layer: int = 0, **kwargs
    ) -> None:
        super().__init__(
            config=config, layer_id=layer_id, start_layer=start_layer, **kwargs
        )

        if config.use_sliding_window:
            raise NotImplementedError(
                "SmolLM3 sliding-window attention is not implemented; "
                "no released checkpoint sets use_sliding_window=True."
            )

        # RoPE is only applied on configured layers
        if not config.no_rope_layers[layer_id]:
            self.rotary_emb = None

        # forward_prepare_npu refreshes the shared cos/sin only on layer start_layer;
        # a NoPE layer never refreshes it, so use the PP stage's first RoPE layer.
        self.start_layer = next(
            (
                i
                for i in range(start_layer, config.num_hidden_layers)
                if config.no_rope_layers[i]
            ),
            start_layer,
        )

    # forward_prepare_npu is intentionally not overridden: its fused kernel requires
    # a real rotary_emb, and LlamaAttention.forward's hasattr check already routes
    # NoPE layers (rotary_emb=None) to this native path instead.
    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        return q, k, v


class SmolLM3DecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: SmolLM3Config,
        layer_id: int = 0,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        if config.mlp_bias:
            raise NotImplementedError(
                "SmolLM3 MLP bias is not implemented; "
                "no released checkpoint sets mlp_bias=True."
            )
        super().__init__(
            config=config,
            layer_id=layer_id,
            start_layer=start_layer,
            quant_config=quant_config,
            prefix=prefix,
        )
        # Override attention to use SmolLM3Attention
        self.self_attn = SmolLM3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=config.rope_parameters["rope_theta"],
            rope_scaling=config.rope_parameters,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=config.attention_bias,
        )


class SmolLM3Model(LlamaModel):
    def __init__(
        self,
        config: SmolLM3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        # Override layer creation to use SmolLM3DecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SmolLM3DecoderLayer(
                config=config,
                quant_config=quant_config,
                layer_id=idx,
                start_layer=self.start_layer,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )


class SmolLM3ForCausalLM(LlamaForCausalLM):
    def _init_model(
        self,
        config: SmolLM3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return SmolLM3Model(config, quant_config=quant_config, prefix=prefix)


EntryClass = SmolLM3ForCausalLM
