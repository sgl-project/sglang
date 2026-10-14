from typing import Any, Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.iquest_q1 import IQuestQ1MTPConfig
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.iquest_q1 import (
    IQuestQ1Attention,
    IQuestQ1MoEBlock,
    IQuestQ1RMSNorm,
    load_iquest_q1_weights,
)
from sglang.srt.utils import add_prefix


class IQuestQ1MTPInnerLayer(nn.Module):
    def __init__(
        self,
        config: IQuestQ1MTPConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = IQuestQ1Attention(
            config=config,
            layer_id=0,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = IQuestQ1MoEBlock(
            config=config,
            layer_id=0,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.attention_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_out_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.feed_forward_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.ffn_out_norm = IQuestQ1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_out_scale = config.first_layer_attn_out_scale
        self.ffn_out_scale = config.first_layer_ffn_out_scale
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, positions, hidden_states, forward_batch):
        dtype = hidden_states.dtype
        attention = self.self_attn(
            positions, self.attention_norm(hidden_states), forward_batch
        )
        residual = (
            hidden_states.float() if self.fp32_residual_connection else hidden_states
        )
        hidden_states = residual + self.attn_out_norm(attention) * self.attn_out_scale
        output = self.mlp(self.feed_forward_norm(hidden_states).to(dtype))
        return hidden_states + self.ffn_out_norm(output) * self.ffn_out_scale


class IQuestQ1MTPLayer(nn.Module):
    def __init__(
        self,
        config: IQuestQ1MTPConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.enorm = IQuestQ1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = IQuestQ1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mtp_model_layer = IQuestQ1MTPInnerLayer(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mtp_model_layer", prefix),
        )
        self.final_layernorm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, positions, previous_hidden_states, input_embeds, forward_batch):
        dtype = self.eh_proj.weight.dtype
        hidden_states = self.eh_proj(
            torch.cat(
                [
                    self.enorm(input_embeds).to(dtype),
                    self.hnorm(previous_hidden_states).to(dtype),
                ],
                dim=-1,
            )
        )
        hidden_states = self.mtp_model_layer(positions, hidden_states, forward_batch)
        return self.final_layernorm(hidden_states).to(dtype)


class IQuestQ1MTPModel(nn.Module):
    def __init__(
        self,
        config: IQuestQ1MTPConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.mtp_layer = IQuestQ1MTPLayer(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mtp_layer", prefix),
        )

    @property
    def layers(self):
        return (self.mtp_layer.mtp_model_layer,)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        return self.mtp_layer(
            positions,
            forward_batch.spec_info.hidden_states,
            input_embeds,
            forward_batch,
        )


class IQuestQ1MTP(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: IQuestQ1MTPConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = IQuestQ1MTPModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        return

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        def mapped_weights():
            for name, weight in weights:
                if name == "target_final_norm.weight":
                    continue
                if name == "embed_tokens.weight":
                    yield "model.embed_tokens.weight", weight
                elif name.startswith("mtp."):
                    yield "model.mtp_layer." + name.removeprefix("mtp."), weight
                elif name == "lm_head.weight":
                    yield name, weight
                else:
                    raise ValueError(f"Unexpected MTP weight: {name}")

        load_iquest_q1_weights(self, mapped_weights())


EntryClass = [IQuestQ1MTP]
