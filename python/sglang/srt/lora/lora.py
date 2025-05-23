# Copyright 2023-2024 SGLang Team
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

# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
# and "Punica: Multi-Tenant LoRA Serving"

# LoRA layers class inheritance adapted from:
# https://github.com/vllm-project/vllm/blob/4abf6336ec65c270343eb895e7b18786e9274176/vllm/lora/layers.py

import logging
import re
from typing import Dict, List

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.model_loader.loader import DefaultModelLoader

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    def __init__(self, config: LoRAConfig, base_hf_config: AutoConfig):
        super().__init__()
        self.config: LoRAConfig = config
        self.base_hf_config: AutoConfig = base_hf_config

        # lora weights in cpu. The weights are loaded from checkpoint.
        self.weights: Dict[str, torch.Tensor] = {}


class LoRAAdapter(nn.Module):
    def __init__(
        self,
        uid: str,
        config: LoRAConfig,
        base_hf_config: AutoConfig,
        load_config: LoadConfig,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.uid: str = uid
        self.config: LoRAConfig = config
        assert self.config.hf_config["peft_type"].lower() == "lora"
        self.base_hf_config: AutoConfig = base_hf_config
        self.load_config: LoadConfig = load_config
        self.lora_backend: BaseLoRABackend = lora_backend
        self.scaling: float = self.config.lora_alpha / self.config.r

        self.layers: List[LoRALayer] = nn.ModuleList(
            [
                LoRALayer(config, base_hf_config)
                for i in range(base_hf_config.num_hidden_layers)
            ]
        )

        self.weights: Dict[str, torch.Tensor] = {}

    # initialize the LoRA weights to cpu
    def initialize_weights(self):
        model_path = self.config.path
        loader = DefaultModelLoader(self.load_config)
        revision = getattr(self.config.hf_config, "revision", None)
        for name, loaded_weight in loader._get_weights_iterator(
            DefaultModelLoader.Source(
                model_path, revision=revision, fall_back_to_pt=True
            )
        ):
            match = re.search(r"layers\.(\d+)\.", name)
            if match is not None:
                layer_id = int(match.group(1))
                self.layers[layer_id].weights[name] = loaded_weight.cpu()
            else:
                self.weights[name] = loaded_weight.cpu()

        # stack kv_proj and gate_up_proj
        for i in range(self.base_hf_config.num_hidden_layers):
            layer = self.layers[i]
            weight_names = [name for name, _ in layer.weights.items()]
            self.stack_qkv_proj(weight_names, layer.weights)
            self.stack_gate_up_proj(weight_names, layer.weights)

    def stack_qkv_proj(self, weight_names: List[str], weights: Dict[str, torch.Tensor]):

        # Collect target q/k/v modules. This process is necessary since there might be no lora attached to k_proj
        target_module = set()
        for weight_name in weight_names:
            if "k_proj" in weight_name:
                target_module.add("k_proj")
            if "q_proj" in weight_name:
                target_module.add("q_proj")
            if "v_proj" in weight_name:
                target_module.add("v_proj")
        if len(target_module) == 0:
            return

        for weight_name in weight_names:
            # We assume every lora adaptor should contain lora modules for q_proj
            if "q_proj" in weight_name:
                q_name = weight_name
                k_name = weight_name.replace("q_proj", "k_proj")
                v_name = weight_name.replace("q_proj", "v_proj")
                kv_name = weight_name.replace("q_proj", "kv_proj")
                qkv_name = weight_name.replace("q_proj", "qkv_proj")

                # If k_proj doesn't have lora, initialize it to zero
                k_proj_weight = (
                    weights[k_name]
                    if "k_proj" in target_module
                    else torch.zeros_like(weights[v_name])
                )
                if "lora_A" in weight_name:
                    weights[qkv_name] = torch.cat(
                        (
                            weights[q_name],
                            k_proj_weight,
                            weights[v_name],
                        ),
                        0,
                    )
                    weights.pop(q_name)
                    if "k_proj" in target_module:
                        weights.pop(k_name)
                    weights.pop(v_name)
                else:
                    weights[kv_name] = torch.stack(
                        [
                            k_proj_weight,
                            weights[v_name],
                        ],
                        dim=0,
                    )
                    if "k_proj" in target_module:
                        weights.pop(k_name)
                    weights.pop(v_name)

    def stack_gate_up_proj(
        self, weight_names: List[str], weights: Dict[str, torch.Tensor]
    ):
        for weight_name in weight_names:
            if "gate_proj" in weight_name:
                up_name = weight_name.replace("gate_proj", "up_proj")
                gate_up_name = weight_name.replace("gate_proj", "gate_up_proj")
                if up_name not in weights:
                    logger.warning(
                        f"Gate projection {weight_name} does not have a corresponding up projection {up_name}. "
                        f"Initializing up projection to zero."
                    )
                    weights[up_name] = torch.zeros_like(weights[weight_name])
                    # FIXME: Add gate-only support for flashinfer in future implementations
                    assert self.lora_backend.name == "triton", (
                        f"LoRA weight initialization currently only supported for 'triton' backend. "
                        f"Received backend: {self.lora_backend.name}. Please verify your backend configuration "
                        f"or consider implementing custom initialization logic for other backends."
                    )
                if "lora_A" in weight_name:
                    weights[gate_up_name] = torch.cat(
                        (weights[weight_name], weights[up_name]), 0
                    )
                else:
                    weights[gate_up_name] = torch.stack(
                        [weights[weight_name], weights[up_name]], dim=0
                    )
                weights.pop(weight_name)
                if up_name in weights:
                    weights.pop(up_name)
