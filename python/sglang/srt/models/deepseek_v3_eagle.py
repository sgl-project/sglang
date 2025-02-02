"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only DeepSeekV3-Eagle model compatible with HuggingFace weights if stipped from hf checkpoint."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM, RMSNorm, DeepseekV2DecoderLayer


class DeepseekV3Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding( # TODO: Verify if embed_tokens or shared_embed tokens
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    config, i, quant_config=quant_config, 
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # TODO: Verify EPSilon
        self.eh_proj = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        input_embeds_norm = self.enorm(hidden_states)
        hnorm_embeds = self.hnorm(forward_batch.spec_info.hidden_states)
        
        hidden_states = self.eh_proj(
            # TODO: Verify if EH or HE order
            torch.cat((hnorm_embeds, input_embeds_norm), dim=-1)
        )

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        return hidden_states + residual

class DeepseekV3ForCausalLMEagle(DeepseekV3ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config=None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekV3Model(config, quant_config=quant_config)
        # TODO: verify right code path here.
        if self.config.tie_word_embeddings:
            
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=quant_config
            )
        self.logits_processor = LogitsProcessor(config)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        decoder_names = [".self_attn.",".mlp.",".post_attention_layernorm.",".input_layernorm."] # weights to be loaded to the decoderlayer
        layer_new_num = "0" # only support 1 layer
        for name, loaded_weight in weights:
            if ".enorm." in name: 
                # used to be name=model.layers.30.enorm
                name = "model.enorm." + name.split(".enorm.")[1]
                super().load_weights([(name, loaded_weight)])
            elif ".hnorm." in name:
                name = "model.hnorm." + name.split(".hnorm.")[1]
                super().load_weights([(name, loaded_weight)])
            elif "lm_head." in name:
                name = "lm_head." + name.split("lm_head.")[1]
                super().load_weights([(name, loaded_weight)])
            elif "embed_tokens" in name:
                name = "model.embed_tokens." + name.split("model.embed_tokens.")[1]
                super().load_weights([(name, loaded_weight)])
            elif "model.norm." in name:
                name = "model.norm." + name.split("model.norm.")[1]
                super().load_weights([(name, loaded_weight)])
            elif any(n in name for n in decoder_names):
                name_split = name.split(".")
                name_split[2] = layer_new_num
                name = ".".join(name_split)
                super().load_weights([(name, loaded_weight)])


EntryClass = [DeepseekV3ForCausalLMEagle]
