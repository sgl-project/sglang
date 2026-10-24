# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AliceAI's bundled MTP head for EAGLE/NEXTN speculative decoding."""

import copy
from collections.abc import Iterable

import torch
from torch import nn

from sglang.srt.configs.alice_ai import AliceAIConfig
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.alice_ai import (
    AliceAIForCausalLM,
    AliceAIFullAttention,
    AliceAISparseMoeBlock,
)
from sglang.srt.runtime_context import get_stream
from sglang.srt.utils import add_prefix, is_cuda, make_layers


class AliceAIMTPDecoderBlock(nn.Module):
    def __init__(
        self,
        config: AliceAIConfig,
        prefix: str,
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.self_attn = AliceAIFullAttention(
            config, layer_id=0, quant_config=None, prefix=f"{prefix}.self_attn"
        )
        self.mlp = AliceAISparseMoeBlock(
            config,
            layer_id=0,
            quant_config=None,
            prefix=f"{prefix}.mlp",
            alt_stream=alt_stream,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, hidden_states, positions, forward_batch: ForwardBatch):
        residual = hidden_states
        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = residual + self.self_attn(
                hidden_states, positions, forward_batch
            )
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class AliceAIMultiTokenPredictor(nn.Module):
    def __init__(self, config: AliceAIConfig, prefix: str):
        super().__init__()
        alt_stream = get_stream("alt") if is_cuda() else None
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.pre_fc_norm_embedding = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = make_layers(
            1,
            lambda idx, prefix: AliceAIMTPDecoderBlock(
                config, prefix, alt_stream=alt_stream
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        hidden_states = forward_batch.spec_info.hidden_states
        if not forward_batch.forward_mode.is_idle():
            input_embeds = self.pre_fc_norm_embedding(input_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))
        hidden_states = self.layers[0](hidden_states, positions, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.norm(hidden_states)
        return hidden_states


class AliceAIForCausalLMMTP(AliceAIForCausalLM):
    def __init__(
        self,
        config: AliceAIConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        if config.mtp_num_hidden_layers != 1:
            raise ValueError("AliceAI MTP requires exactly one checkpoint MTP layer")
        if quant_config is not None:
            raise NotImplementedError("AliceAI MTP currently requires BF16 weights")
        config = copy.deepcopy(config)
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        config.num_nextn_predict_layers = 1
        self.config = config
        self.model = AliceAIMultiTokenPredictor(
            config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("lm_head", prefix),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states = self.model(
                input_ids, positions, forward_batch, input_embeds
            )
        # EAGLE chains the final, normalized draft states into the next MTP step.
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def set_embed_and_head(self, embed, head):
        if embed is not None:
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        elif head is not None:
            del self.lm_head.weight
            self.lm_head.weight = head

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # EAGLE shares the target embedding/head before draft capture and inference.
        def draft_weights():
            for name, weight in weights:
                if name.startswith("mtp."):
                    yield "model." + name.removeprefix("mtp."), weight

        return super().load_weights(draft_weights(), is_mtp=True)


EntryClass = AliceAIForCausalLMMTP
