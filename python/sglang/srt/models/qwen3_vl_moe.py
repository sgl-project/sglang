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
"""Inference-only Qwen3-VL model compatible with HuggingFace weights."""
import logging
from functools import lru_cache, partial
from typing import Callable, Iterable, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BatchFeature
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.configs.qwen3_vl import Qwen3VLMoeConfig, Qwen3VLMoeVisionConfig
from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
    global_server_args_dict,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel
from sglang.srt.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
)
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class Qwen3MoeLLMModel(Qwen3MoeModel):
    def __init__(
        self,
        *,
        config: Qwen3VLMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

        self.hidden_size = config.hidden_size

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer]
        ):
            layer_idx = layer_idx + self.start_layer
            if layer_idx in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )

            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

            # process deepstack
            if input_deepstack_embeds is not None and layer_idx in range(3):
                sep = self.hidden_size * layer_idx
                hidden_states = (
                    hidden_states
                    + input_deepstack_embeds[:, sep : sep + self.hidden_size]
                )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(
        self,
        *,
        config: Qwen3VLMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super(Qwen3VLForConditionalGeneration, self).__init__()
        self.config = config

        self.visual = Qwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            # NOTE: Qwen3-VL vision encoder currently supports BitsAndBytes 4-bit quantization.
            # Other quantization methods (e.g., GPTQ, AWQ) are untested and may not be supported.
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )

        self.model = Qwen3MoeLLMModel(
            config=config,
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
                prefix=add_prefix("lm_head", prefix),
            )
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # deepstack
        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes
        self.num_deepstack_embeddings = len(self.deepstack_visual_indexes)

    @property
    def use_deepstack(self) -> bool:
        return hasattr(self, "deepstack_visual_indexes")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        """Run forward pass for Qwen3-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            use_deepstack=self.use_deepstack,
        )

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ):
        param = params_dict[name]
        # weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        weight_loader = param.weight_loader
        ep_rank = get_tensor_model_parallel_rank()
        ep_size = get_moe_expert_parallel_world_size()
        if ep_size == 1:
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
        else:
            experts_per_ep = num_experts // ep_size
            start_expert = ep_rank * experts_per_ep
            end_expert = (
                (ep_rank + 1) * experts_per_ep
                if ep_rank != ep_size - 1
                else num_experts
            )

            for idx, expert_id in enumerate(range(start_expert, end_expert)):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    idx,
                )
        return True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        num_experts = self.config.num_experts

        # Cache params_dict to avoid repeated expensive traversal of model parameters
        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict
        for name, loaded_weight in weights:
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue

                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                # [TODO] Skip layers that are on other devices (check if sglang has a similar function)
                # if is_pp_missing_parameter(name, self):
                #     continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        loaded_weight = loaded_weight.transpose(-1, -2)  # no bias
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        else:
                            self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue
                    if "visual" in name:
                        # adapt to VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                        name = name.replace(r"model.visual.", r"visual.")

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

        # TODO mimic deepseek
        # Lazy initialization of expert weights cache to avoid slowing down load_weights
        # if not hasattr(self, "routed_experts_weights_of_layer"):
        #     self.routed_experts_weights_of_layer = {
        #         layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
        #         for layer_id in range(self.start_layer, self.end_layer)
        #         if isinstance(self.model.layers[layer_id].mlp, Qwen3MoeSparseMoeBlock)
        #     }


EntryClass = Qwen3VLMoeForConditionalGeneration
