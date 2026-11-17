# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Nemotron Shortcut multi-token prediction (MTP) draft model for EAGLE / NEXTN.

One repeated layer serves every speculative step (``mtp_use_repeated_layer``):
the previous step's final-normed hidden state and the next token's embedding
are RMS-normed, concatenated and projected, run through one shortcut
attention + latent-MoE block on a plain residual, then final-normed. That
output both feeds the target's lm_head and is the next step's hidden state.
The draft's attention is dense (the checkpoint's MTP indexer is not served),
which only lowers acceptance beyond index_topk tokens.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.nemotron_shortcut import (
    ShortcutMoELayer,
    load_nemotron_shortcut_weight,
    nemotron_shortcut_expert_params_mapping,
    warn_unloaded_params,
)

# Checkpoint prefix of the one repeated MTP layer.
_MTP_PREFIX = "model.language_model.mtp.layers.0."


class NemotronShortcutMTPLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        hidden = config.hidden_size
        self.enorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * hidden,
            hidden,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("eh_proj", prefix),
        )
        # The draft worker's KV pool holds this one attention layer as layer 0.
        self.block = ShortcutMoELayer(
            config,
            "shortcut_moe_attn",
            0,
            quant_config,
            add_prefix("block", prefix),
            wide_residual=False,
            use_sdsa=False,
        )
        self.final_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        x, _ = self.eh_proj(
            torch.cat((self.enorm(input_embeds), self.hnorm(hidden_states)), dim=-1)
        )
        x = self.block(x, forward_batch)
        return self.final_layernorm(x)


class NemotronShortcutForCausalLMMTP(nn.Module):
    # Matches the checkpoint's per-projection quantization ignore entries.
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(
        self,
        *,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        text_config = config.get_text_config()
        self.config = config
        self.text_config = text_config
        self.quant_config = quant_config
        self.image_token_id = config.image_token_id
        self.embed_tokens = VocabParallelEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            prefix=add_prefix("model.language_model.embed_tokens", prefix),
        )
        self.mtp = NemotronShortcutMTPLayer(
            text_config, quant_config, prefix=add_prefix(_MTP_PREFIX[:-1], prefix)
        )
        # Adopted from the target in set_lm_head_from_target: the head is FP8
        # with per-tensor scales, so sharing only its weight would drop them.
        self.lm_head: Optional[nn.Module] = None
        self.logits_processor = LogitsProcessor(text_config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.lm_head is None:
            raise RuntimeError(
                "Nemotron Shortcut MTP needs the target's lm_head (EAGLE v2)."
            )
        if input_embeds is None:
            # Training embeds the image token at image positions; SGLang fills
            # them with per-image pad ids past the vocabulary.
            input_ids = torch.where(
                input_ids >= self.text_config.vocab_size,
                self.image_token_id,
                input_ids,
            )
            input_embeds = self.embed_tokens(input_ids)
        hidden_states = forward_batch.spec_info.hidden_states
        if input_embeds.shape[0] > 0:
            hidden_states = self.mtp(input_embeds, hidden_states, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.embed_tokens.weight, None

    def set_embed_and_head(self, embed, head):
        del self.embed_tokens.weight
        self.embed_tokens.weight = embed
        torch.cuda.empty_cache()

    def set_lm_head_from_target(self, target_lm_head: nn.Module) -> None:
        self.lm_head = target_lm_head

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        expert_params_mapping = nemotron_shortcut_expert_params_mapping(
            self.text_config.n_routed_experts
        )
        params_dict = {
            name: param
            for name, param in self.named_parameters()
            # Shared with the target in set_embed_and_head.
            if not name.startswith("embed_tokens.")
        }
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            # The draft attention is dense: the MTP indexer is not served.
            if not name.startswith(_MTP_PREFIX) or ".indexer." in name:
                continue
            load_nemotron_shortcut_weight(
                params_dict,
                "mtp." + name.removeprefix(_MTP_PREFIX),
                loaded_weight,
                loaded,
                expert_params_mapping,
            )
        warn_unloaded_params(params_dict, loaded)


EntryClass = [NemotronShortcutForCausalLMMTP]
