"""Inference-only SmolLM3 model compatible with HuggingFace weights."""

from typing import Optional

import torch
from transformers import SmolLM3Config

from sglang.srt.distributed import get_pp_group, get_pp_indices
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
)
from sglang.srt.utils import add_prefix, make_layers


class SmolLM3Attention(LlamaAttention):
    """Llama attention, minus RoPE on the layers config.no_rope_layers marks as NoPE."""

    def __init__(self, *, config: SmolLM3Config, layer_id: int, **kwargs) -> None:
        super().__init__(config=config, layer_id=layer_id, **kwargs)

        if config.use_sliding_window:
            raise NotImplementedError(
                "SmolLM3 sliding-window attention is not implemented; "
                "no released checkpoint sets use_sliding_window=True."
            )

        # RoPE is only applied on configured layers
        if not config.no_rope_layers[layer_id]:
            self.rotary_emb = None

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
    # mirrors Llama decoder layer init, except for assignment of self.attn
    def __init__(
        self,
        config: SmolLM3Config,
        layer_id: int = 0,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        torch.nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", 10000)
            rope_scaling = rope_parameters
        else:
            rope_theta = getattr(config, "rope_theta", 10000)
            rope_scaling = getattr(config, "rope_scaling", None)
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = SmolLM3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=config.attention_bias,
        )
        if config.mlp_bias:
            raise NotImplementedError(
                "SmolLM3 MLP bias is not implemented; "
                "no released checkpoint sets mlp_bias=True."
            )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class SmolLM3Model(LlamaModel):
    def __init__(
        self,
        config: SmolLM3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        torch.nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        pp_start_layer, _ = get_pp_indices(
            config.num_hidden_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SmolLM3DecoderLayer(
                config=config,
                quant_config=quant_config,
                layer_id=idx,
                start_layer=pp_start_layer,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self.layers_to_capture = []


class SmolLM3ForCausalLM(LlamaForCausalLM):
    def _init_model(
        self,
        config: SmolLM3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return SmolLM3Model(config, quant_config=quant_config, prefix=prefix)


EntryClass = SmolLM3ForCausalLM
