# Copyright 2025 SGLang Team
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

"""Inference-only pure Mamba2 model (no attention layers).

Supports Mamba2ForCausalLM architectures such as mistralai/Mamba-Codestral-7B-v0.1.
All layers are SSM (MambaMixer2) — there are no attention or MLP branches.
Follows the NemotronH model pattern with attention/MoE/MLP removed.
"""

from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn

from sglang.srt.configs.mamba2 import Mamba2Config
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    Mamba2AttnBackend,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers
from sglang.utils import logger


class Mamba2DecoderLayer(nn.Module):
    """Single Mamba2 decoder layer: RMSNorm → MambaMixer2 → residual."""

    def __init__(
        self,
        config: Mamba2Config,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_idx
        self.mixer = MambaMixer2(
            cache_params=config.mamba2_cache_params,
            hidden_size=config.hidden_size,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            n_groups=config.n_groups,
            rms_norm_eps=config.layer_norm_epsilon,
            activation=config.hidden_act,
            use_rms_norm=config.rms_norm,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        output = torch.empty_like(hidden_states)
        attn_backend = forward_batch.attn_backend
        assert isinstance(attn_backend, HybridLinearAttnBackend)
        assert isinstance(attn_backend.linear_attn_backend, Mamba2AttnBackend)
        attn_backend.linear_attn_backend.forward(
            mixer=self.mixer,
            layer_id=self.layer_id,
            hidden_states=hidden_states,
            output=output,
            use_triton_causal_conv=True,
        )
        return output, residual


class Mamba2Model(nn.Module):
    """Mamba2 model body: embeddings → stacked decoder layers → final norm."""

    def __init__(
        self,
        *,
        config: Mamba2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(idx: int, prefix: str):
            return Mamba2DecoderLayer(
                config, idx, quant_config=quant_config, prefix=prefix
            )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        if self.pp_group.is_last_rank:
            self.final_layernorm = RMSNorm(
                config.hidden_size, eps=config.layer_norm_epsilon
            )
        else:
            self.final_layernorm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer.forward(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class Mamba2ForCausalLM(nn.Module):
    """
    Pure Mamba2 model for causal language modeling.

    All layers are SSM (MambaMixer2) with no attention. Supports tensor
    parallelism and pipeline parallelism via the same infrastructure as
    hybrid models (NemotronH, FalconH1, GraniteHybrid).
    """

    # Weight name remapping: HF checkpoint → SGLang model
    # HF uses "backbone.*" prefix; SGLang uses "model.*"
    # HF uses "A_log" for the log-space A parameter; MambaMixer2 uses "A"
    # HF uses "embeddings" for the embedding layer; SGLang uses "embed_tokens"
    # HF uses "norm_f" for final layernorm; SGLang uses "final_layernorm"
    remap_prefix = {"backbone": "model"}
    remap_substr = {
        "A_log": "A",
        "embeddings": "embed_tokens",
        "norm_f": "final_layernorm",
    }

    def __init__(
        self,
        *,
        config: Mamba2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.model = Mamba2Model(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model.forward(
            input_ids, positions, forward_batch, pp_proxy_tensors, input_embeds
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Remap HF checkpoint names to SGLang model names
            for old_prefix, new_prefix in self.remap_prefix.items():
                if name.startswith(old_prefix):
                    name = new_prefix + name[len(old_prefix) :]
                    break
            for old_sub, new_sub in self.remap_substr.items():
                name = name.replace(old_sub, new_sub)

            # Skip layers outside our pipeline-parallel shard
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            # Skip embeddings/head on non-owning PP ranks
            if "embed_tokens" in name and not self.pp_group.is_first_rank:
                continue
            if (
                "final_layernorm" in name or "lm_head" in name
            ) and not self.pp_group.is_last_rank:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.warning(f"Parameter {name} not found in params_dict")


EntryClass = [Mamba2ForCausalLM]
