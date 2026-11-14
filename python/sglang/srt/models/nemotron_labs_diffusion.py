# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Nemotron Labs Diffusion with causal context and bidirectional denoising."""

import torch

from sglang.srt.dllm.nemotron_linear_spec import is_causal_verify
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from sglang.srt.models.utils import WeightsMapper


class NemotronLabsDiffusionAttention(LlamaAttention):
    def __init__(self, *, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.llama4_scaling_beta = config.rope_parameters.get("llama_4_scaling_beta")
        self.original_max_position = config.rope_parameters.get(
            "original_max_position_embeddings", config.max_position_embeddings
        )
        self.diffusion_attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=self.attn.layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
        )

    def forward_prepare_native(self, positions, hidden_states):
        q, k, v = super().forward_prepare_native(positions, hidden_states)
        if self.llama4_scaling_beta is not None:
            scale = 1.0 + self.llama4_scaling_beta * torch.log(
                1.0 + torch.floor(positions.float() / self.original_max_position)
            )
            q = q * scale.unsqueeze(-1).to(q.dtype)
        return q, k, v

    def forward(self, positions, hidden_states, forward_batch):
        q, k, v = self.forward_prepare_native(positions, hidden_states)
        attention = (
            self.diffusion_attn
            if forward_batch.forward_mode.is_dllm_extend()
            and not is_causal_verify(forward_batch)
            else self.attn
        )
        output = attention(q, k, v, forward_batch)
        output, _ = self.o_proj(output)
        return output


class NemotronLabsDiffusionDecoderLayer(LlamaDecoderLayer):
    attention_cls = NemotronLabsDiffusionAttention


class NemotronLabsDiffusionBackbone(LlamaModel):
    decoder_layer_cls = NemotronLabsDiffusionDecoderLayer


class NemotronLabsDiffusionModel(LlamaForCausalLM):
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={"encoder.": "model.", "diffusion_head.": "lm_head."}
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config, quant_config=None, prefix=""):
        if config.tie_word_embeddings:
            raise ValueError(
                "Nemotron Labs Diffusion requires an untied diffusion_head"
            )
        super().__init__(config, quant_config, prefix)
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

    def _init_model(self, config, quant_config=None, prefix=""):
        return NemotronLabsDiffusionBackbone(config, quant_config, prefix)

    def load_weights(self, weights):
        return super().load_weights(self.hf_to_sglang_mapper.apply(weights))


EntryClass = NemotronLabsDiffusionModel
