# Copyright 2023-2025 SGLang Team
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
# ==============================================================================

""" Ernie4.5 MTP model compatible with baidu/ERNIE-4.5-*-PT weights. """

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers.models.ernie4_5_moe.configuration_ernie4_5_moe import (
    Ernie4_5_MoeConfig,
)

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.ernie4 import Ernie4_5_ForCausalLM, Ernie4DecoderLayer
from sglang.srt.utils import add_prefix


class Ernie4ModelMTP(nn.Module):
    def __init__(
        self,
        config: Ernie4_5_MoeConfig,
        layer_id: int,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.mtp_emb_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_linear_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=config.use_bias
        )
        self.mtp_block = Ernie4DecoderLayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("mtp_block", prefix),
            is_mtp=True,
        )

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
        # masking inputs at position 0, as not needed by MTP
        hidden_states[positions == 0] = 0

        hidden_states = self.mtp_linear_proj(
            torch.cat(
                (
                    self.mtp_emb_norm(hidden_states),
                    self.mtp_hidden_norm(forward_batch.spec_info.hidden_states),
                ),
                dim=-1,
            )
        )
        residual = None
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=residual,
        )
        hidden_states = residual + hidden_states
        return hidden_states


class Ernie4_5_MoeForCausalLMMTP(nn.Module):
    def __init__(
        self,
        config: Ernie4_5_MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        mtp_layer_id: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.mtp_layer_id = mtp_layer_id

        self.model = Ernie4ModelMTP(
            config=config,
            layer_id=self.mtp_layer_id,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix="lm_head",
            )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        mtp_layer_found = False
        mtp_weight_patterns = [
            f"mtp_block.{self.mtp_layer_id}",
            f"mtp_emb_norm.{self.mtp_layer_id}",
            f"mtp_hidden_norm.{self.mtp_layer_id}",
            f"mtp_linear_proj.{self.mtp_layer_id}",
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # Only name matched patterns should be loaded
            for layer_pattern in mtp_weight_patterns:
                if layer_pattern in name:
                    mtp_layer_found = True
                    break
            else:
                continue
            # But strip mtp_layer_id before loading, because each MTP layer is a MTP model.
            name = name.replace(f".{self.mtp_layer_id}.", ".")
            for (
                param_name,
                weight_name,
                shard_id,
            ) in Ernie4_5_ForCausalLM.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    raise KeyError(f"Parameter '{name}' not found in MTP model.")
        if not mtp_layer_found:
            raise KeyError(
                f"MTP layers 'mtp_*.{self.mtp_layer_id}.*' not found in weights."
            )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            del self.lm_head.weight
            self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = [Ernie4_5_MoeForCausalLMMTP]
