# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The code is based on llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py), but modified for KimiVL.
#
# Licensing Information:
# - Code derived from llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
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
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.activations import GELUActivation

from sglang.srt.configs import KimiVLConfig
from sglang.srt.configs.deepseekvl2 import DeepseekV2Config
from sglang.srt.configs.kimi_vl import KimiVLConfig
from sglang.srt.configs.kimi_vl_moonvit import MoonViTConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import QuickGELU
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.models.kimi_vl_moonvit import MoonVitPretrainedModel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


# For dummy input only
@dataclass
class MaxImageTokenMeta:
    width: int = 1024
    height: int = 1024


class KimiVLMultiModalProjector(nn.Module):

    def __init__(self, config: KimiVLConfig):
        super().__init__()

        self.hidden_size = (
            config.vision_config.hidden_size
            * config.vision_config.merge_kernel_size[0]
            * config.vision_config.merge_kernel_size[1]
        )

        self.pre_norm = torch.nn.LayerNorm(config.vision_config.hidden_size, eps=1e-5)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELUActivation()
        self.act = QuickGELU()
        self.linear_2 = nn.Linear(
            self.hidden_size, config.text_config.hidden_size, bias=True
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class KimiVLForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: KimiVLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,  # fix init_tts argument error
    ) -> None:
        super().__init__()
        self.config = config
        assert isinstance(config.vision_config, MoonViTConfig)

        self.vision_tower = MoonVitPretrainedModel(config.vision_config)

        self.multi_modal_projector = KimiVLMultiModalProjector(config=config)
        self.quant_config = quant_config
        text_config = copy.deepcopy(config.text_config)
        text_config.architectures = ["DeepseekV2ForCausalLM"]
        self.language_model = DeepseekV2ForCausalLM(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = (
            torch.cat([item.pixel_values for item in items], dim=0)
            .type(self.vision_tower.dtype)
            .to(self.vision_tower.device)
        )
        image_grid_hws = torch.cat([item.image_grid_hws for item in items], dim=0).to(
            self.vision_tower.device
        )
        image_features = self.vision_tower(pixel_values, image_grid_hws)
        assert isinstance(image_features, list)
        # lengths = [x.shape[0] for x in image_features]
        res = self.multi_modal_projector(torch.cat(image_features))  # .split(lengths)
        return res

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        config = self.config.text_config
        _KEYS_TO_MODIFY_MAPPING = {
            # "language_model.lm_head": "lm_head",
            # "language_model.model": "language_model",
        }
        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if not config.use_mla:
            stacked_params_mapping += [
                (".qkv_proj", ".q_proj", "q"),
                (".qkv_proj", ".k_proj", "k"),
                (".qkv_proj", ".v_proj", "v"),
            ]
        if getattr(config, "n_routed_experts", None):
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=config.n_routed_experts,
            )
        else:
            expert_params_mapping = []

        params_dict = dict(self.named_parameters())
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            use_default_weight_loading = False
            if "vision" in name:
                if self.vision_tower is not None:
                    # We only do sharding for language model and
                    # not vision model for now.
                    use_default_weight_loading = True
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    # We have mlp.experts[0].gate_proj in the checkpoint.
                    # Since we handle the experts below in expert_params_mapping,
                    # we need to skip here BEFORE we update the name, otherwise
                    # name will be updated to mlp.experts[0].gate_up_proj, which
                    # will then be updated below in expert_params_mapping
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                    if ("mlp.experts." in name) and name not in params_dict:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id, **kwargs)
                    break
                else:
                    for idx, (
                        param_name,
                        weight_name,
                        expert_id,
                        shard_id,
                    ) in enumerate(expert_params_mapping):
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)

                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name,
                            expert_id=expert_id,
                            shard_id=shard_id,
                            **kwargs,
                        )
                        break
                    else:
                        use_default_weight_loading = True
            if use_default_weight_loading:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                # if is_pp_missing_parameter(name, self):
                #     continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, **kwargs)
        self.language_model.post_load_weights()


def get_spec_layer_idx_from_weight_name(
    config: DeepseekV2Config, weight_name: str
) -> Optional[int]:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None


EntryClass = [KimiVLForConditionalGeneration]
