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
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import LazyValue, add_prefix

logger = logging.getLogger(__name__)

_MODEL_PREFIX = "model."


class Qwen3_5ForCausalLM(nn.Module):
    body_cls = qwen3_5.Qwen3_5ForCausalLM

    packed_modules_mapping = qwen3_5.Qwen3_5ForCausalLM.packed_modules_mapping
    supported_lora_modules = qwen3_5.Qwen3_5ForCausalLM.supported_lora_modules

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        # The body decides; it is handed this config and quantization verbatim.
        return cls.body_cls.shared_experts_fusion_disable_reason(
            hf_config, quant_config
        )

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
                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)

        # Text-only checkpoints retain mrope_section, but identical position rows
        # make its rotary embedding equivalent to 1-D RoPE.
        rope_config = getattr(config, "rope_parameters", None) or getattr(
            config, "rope_scaling", None
        )
        self.is_mrope_enabled = bool(rope_config) and "mrope_section" in rope_config

        self.capture_aux_hidden_states = False

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def prepare_before_cuda_graph_capture(self, model_runner) -> None:
        """Forward model-owned warmup to the text backbone."""
        self.model.prepare_before_cuda_graph_capture(model_runner)

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        if self.pp_group.world_size > 1:
            raise NotImplementedError("DFLASH/DSPARK aux hidden capture requires PP=1.")
        num_layers = len(self.model.layers)
        if sorted(set(layer_ids)) != list(layer_ids) or not all(
            0 <= layer_id < num_layers - 1 for layer_id in layer_ids
        ):
            raise ValueError(
                "target_layer_ids must be unique, strictly increasing, and in "
                f"[0, {num_layers - 1}); got {layer_ids}"
            )
        self.capture_aux_hidden_states = True
        self.model.set_dflash_layers_to_capture(
            [layer_id + 1 for layer_id in layer_ids]
        )

    def get_embed_and_head(self):
        # PP splits embedding and lm_head across first/last stages; the draft keeps
        # its own copy of whichever half its stage cannot receive.
        embed = (
            None
            if isinstance(self.model.embed_tokens, PPMissingLayer)
            else self.model.embed_tokens.weight
        )
        head = None if isinstance(self.lm_head, PPMissingLayer) else self.lm_head.weight
        return embed, head

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):
        if not self.pp_group.is_last_rank:
            return
        if layers_to_capture is None:
            raise ValueError(
                "DFLASH requires explicit layer ids for aux hidden capture."
            )
        self.capture_aux_hidden_states = True
        self.model.set_dflash_layers_to_capture(layers_to_capture)

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

        if not self.pp_group.is_last_rank:
            return hidden_states

        aux_hidden_states = None
        if isinstance(hidden_states, tuple):
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        # Keep prefix stripping lazy: materializing mmap-backed checkpoint tensors
        # across model processes can exceed host memory before GPU copies finish.
        def body_weights():
            for name, loaded_weight in weights:
                if name.startswith(_MODEL_PREFIX):
                    yield name[len(_MODEL_PREFIX) :], loaded_weight
                elif name == "lm_head.weight":
                    if self.config.tie_word_embeddings:
                        continue
                    if "lm_head.weight" not in params_dict:
                        continue
                    param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add("lm_head.weight")

        body_loaded = self.model.load_weights(body_weights())
        loaded_params.update(f"{_MODEL_PREFIX}{n}" for n in body_loaded)

        if self.config.tie_word_embeddings and self.pp_group.is_last_rank:
            loaded_params.add("lm_head.weight")

        return loaded_params


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
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
