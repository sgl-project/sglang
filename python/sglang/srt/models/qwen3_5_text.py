# Copyright 2025 Qwen Team
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
"""Inference-only pure-text (non-VL) Qwen3.5 and Qwen3.5-MoE causal LM.

The transformer bodies (embeddings + hybrid GDN/attention decoder layers + final
norm) already live in ``qwen3_5.py`` under the ``*ForCausalLM`` names, where they
are used as the ``language_model`` component of the VL wrappers. Those classes
return plain hidden states and have no ``lm_head``.

This module adds the top-level *servable* classes for the plain-text checkpoints
whose HuggingFace ``architectures`` are ``Qwen3_5ForCausalLM`` /
``Qwen3_5MoeForCausalLM`` (``model_type`` ``qwen3_5_text`` / ``qwen3_5_moe_text``).
Each wraps the matching body with an ``lm_head`` + ``LogitsProcessor``, following
the same structure as ``qwen3_next.py::Qwen3NextForCausalLM``.

The servable class names intentionally match the identically named transformer
bodies in ``qwen3_5``; the two never collide because the model registry keys on
``EntryClass`` names only (the bodies are not entry classes) and the bodies are
always referenced here module-qualified as ``qwen3_5.<name>``.
"""

import logging
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models import qwen3_5
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import LazyValue, add_prefix

logger = logging.getLogger(__name__)

_MODEL_PREFIX = "model."


class Qwen3_5ForCausalLM(nn.Module):
    """Servable pure-text Qwen3.5 (dense) causal LM.

    Also serves as the base for the MoE variant; subclasses override
    ``body_cls`` with the matching transformer body from ``qwen3_5``.
    """

    # Transformer body class (embeddings + decoder layers + final norm).
    body_cls = qwen3_5.Qwen3_5ForCausalLM

    packed_modules_mapping = qwen3_5.Qwen3_5ForCausalLM.packed_modules_mapping
    supported_lora_modules = qwen3_5.Qwen3_5ForCausalLM.supported_lora_modules

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        if quant_config is not None and hasattr(quant_config, "packed_modules_mapping"):
            quant_config.packed_modules_mapping = self.packed_modules_mapping

        self.model = self.body_cls(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    org_num_embeddings=config.vocab_size,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_server_args().enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)

        # Qwen3.5 checkpoints carry an ``mrope_section`` (multimodal RoPE). For
        # text-only inputs the runtime builds ``[3, seq]`` positions whose three
        # rows are identical, which is equivalent to standard 1-D RoPE, so we
        # feed ``forward_batch.mrope_positions`` like the VL wrapper does.
        rope_config = getattr(config, "rope_parameters", None) or getattr(
            config, "rope_scaling", None
        )
        self.is_mrope_enabled = bool(rope_config) and "mrope_section" in rope_config

        # For EAGLE3 / aux hidden state capture.
        self.capture_aux_hidden_states = False

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        # Intermediate PP ranks return proxy tensors for the next stage.
        if not self.pp_group.is_last_rank:
            return hidden_states

        aux_hidden_states = None
        if isinstance(hidden_states, tuple):
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Load a plain-text checkpoint.

        The body's ``load_weights`` (in ``qwen3_5``) handles all stacked / GDN /
        MoE-fused / shared-expert remapping, but operates on body-relative names
        (``embed_tokens.*``, ``layers.*``, ``norm.*``). Checkpoint keys are
        prefixed with ``model.``, so we strip that prefix before delegating and
        re-add it for the returned set. ``lm_head`` is loaded here; ``mtp.*`` and
        other non-``model.`` tensors are ignored (the MTP draft model loads those
        separately).
        """
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        body_weights = []
        for name, loaded_weight in weights:
            if name.startswith(_MODEL_PREFIX):
                body_weights.append((name[len(_MODEL_PREFIX) :], loaded_weight))
            elif name == "lm_head.weight":
                if self.config.tie_word_embeddings:
                    # Tied to embed_tokens; loaded via the body below.
                    continue
                if "lm_head.weight" not in params_dict:
                    continue
                param = params_dict["lm_head.weight"]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add("lm_head.weight")
            # else: mtp.* / rotary inv_freq / etc. -> skip

        body_loaded = self.model.load_weights(body_weights)
        loaded_params.update(f"{_MODEL_PREFIX}{n}" for n in body_loaded)

        if self.config.tie_word_embeddings and self.pp_group.is_last_rank:
            loaded_params.add("lm_head.weight")

        return loaded_params


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """Servable pure-text Qwen3.5-MoE causal LM."""

    body_cls = qwen3_5.Qwen3_5MoeForCausalLM

    packed_modules_mapping = qwen3_5.Qwen3_5MoeForCausalLM.packed_modules_mapping
    supported_lora_modules = qwen3_5.Qwen3_5MoeForCausalLM.supported_lora_modules

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

        # Expose routed-expert weights for EPLB, mirroring qwen3_next.
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = [Qwen3_5MoeForCausalLM, Qwen3_5ForCausalLM]
