"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
# and "Punica: Multi-Tenant LoRA Serving"


import logging
import re

import torch

from sglang.srt.lora.lora import LoRAAdapter, get_lora_layer
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_hip, replace_submodule

logger = logging.getLogger(__name__)


# ROCm: flashinfer available later
if not is_hip():
    from flashinfer import SegmentGEMMWrapper


def get_module_name(name):
    # Fallback solution of mapping from config module name to module name in model class.
    # Please check if it aligns with your base model.
    # Please implement the function in the model class if it is not.
    # You can reference this function in llama.py.
    params_mapping = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
    }
    return params_mapping.get(name, name)


def get_hidden_dim(module_name, config):
    # Fallback solution of get_hidden_dim for different modules
    # Please check if it aligns with your base model.
    # Please implement the function in the model class if it is not.
    # You can reference this function in llama.py.
    if module_name in ["q_proj", "o_proj", "qkv_proj"]:
        return config.hidden_size, config.hidden_size
    elif module_name in ["kv_proj"]:
        return config.hidden_size, config.hidden_size // (
            config.num_attention_heads // config.num_key_value_heads
        )
    elif module_name == "gate_up_proj":
        return config.hidden_size, config.intermediate_size
    elif module_name == "down_proj":
        return config.intermediate_size, config.hidden_size
    else:
        raise NotImplementedError()


def get_stacked_name(name):
    # origin name -> (name for A, name for B)
    params_mapping = {
        "q_proj": ("qkv_proj", "q_proj"),
        "k_proj": ("qkv_proj", "kv_proj"),
        "v_proj": ("qkv_proj", "kv_proj"),
        "gate_proj": ("gate_up_proj", "gate_up_proj"),
        "up_proj": ("gate_up_proj", "gate_up_proj"),
    }
    return params_mapping.get(name, (name, name))


def get_layer_id(name):
    match = re.search(r"layers\.(\d+)\.", name)
    if match is None:
        return None
    return int(match.group(1))


class LoRAManager:
    def __init__(
        self,
        base_model,
        lora_paths,
        base_hf_config,
        max_loras_per_batch,
        load_config,
        dtype,
    ):
        self.base_model = base_model
        self.lora_paths = lora_paths
        self.base_hf_config = base_hf_config
        self.max_loras_per_batch = max_loras_per_batch
        self.load_config = load_config
        self.dtype = dtype

        workspace_buffer = torch.empty(1 * 1024 * 1024, dtype=torch.int8, device="cuda")
        self.segment_gemm = SegmentGEMMWrapper(workspace_buffer)

        self.init_loras()
        self.init_lora_memory_pool()
        self.init_lora_batch()

    def match_target_modules(self, module_name):
        for target_module in self.target_modules:
            if module_name.split(".")[-1] == target_module:
                return True
        return False

    def get_target_modules(self):
        modules = []
        for module_name, module in self.base_model.named_modules():
            if self.match_target_modules(module_name):
                modules.append((module_name, module))
        return modules

    def set_lora_module(self, module_name, module):
        lora_module = get_lora_layer(
            module, self.segment_gemm, self.max_lora_dim, self.scaling
        )
        replace_submodule(self.base_model, module_name, lora_module)
        return lora_module

    def init_loras(self):
        # get configs and target modules
        self.configs = {}
        self.origin_target_modules = set()
        for name, path in self.lora_paths.items():
            self.configs[name] = LoRAConfig(path)
            self.origin_target_modules = set(self.origin_target_modules) | set(
                self.configs[name].target_modules
            )
        if hasattr(self.base_model, "get_module_name"):
            self.target_modules = {
                self.base_model.get_module_name(module)
                for module in self.origin_target_modules
            }
        else:
            logger.warning(
                f"WARNING: get_module_name() is not defined, "
                f"which is used to map config module name to model implementation module name."
                f"Use the default one, but please check if it is correct for your model."
            )
            self.target_modules = {
                get_module_name(module) for module in self.origin_target_modules
            }
        self.target_weights = set(
            [get_stacked_name(module) for module in self.origin_target_modules]
        )

        # load all weights to cpu
        self.loras = []
        self.lora_id = {}
        for name in self.lora_paths.keys():
            self.lora_id[name] = len(self.loras)
            self.loras.append(
                LoRAAdapter(
                    name, self.configs[name], self.base_hf_config, self.load_config
                )
            )
            self.loras[-1].initialize_weights()

        # misc lora configs
        self.max_lora_dim = max([x.hf_config["r"] for x in self.configs.values()])
        self.scaling = self.loras[0].scaling
        # FIXME remove the restrictions
        assert all(x.hf_config["r"] == self.max_lora_dim for x in self.configs.values())
        assert all(x.scaling == self.scaling for x in self.loras)

        # monkey patch to use the LoRA version
        self.lora_modules = []
        for module_name, module in self.get_target_modules():
            self.lora_modules.append(
                (module_name, self.set_lora_module(module_name, module))
            )

    def init_lora_memory_pool(self):
        # preallocate lora memory pool
        self.A_buffer = {}
        self.B_buffer = {}
        num_layer = self.base_hf_config.num_hidden_layers
        for module_A, module_B in self.target_weights:
            # init A tensor, column_major=True
            if hasattr(self.base_model, "get_hidden_dim"):
                hidden_dim_A, _ = self.base_model.get_hidden_dim(module_A)
            else:
                logger.warning(
                    f"WARNING: get_hidden_dim() is not defined, "
                    f"which is used to get the hidden dim for different lora modules"
                    f"Use the default one, but please check if it is correct for your model."
                )
                hidden_dim_A, _ = get_hidden_dim(module_A, self.base_hf_config)
            c = self.loras[-1].get_stacked_multiply(module_A)
            if module_A not in self.A_buffer:
                self.A_buffer[module_A] = [
                    torch.empty(
                        (
                            self.max_loras_per_batch,
                            self.max_lora_dim * c,
                            hidden_dim_A,
                        ),
                        dtype=self.dtype,
                        device="cuda",
                    )
                    for i in range(num_layer)
                ]
            # init B tensor, column_major=True
            if hasattr(self.base_model, "get_hidden_dim"):
                _, hidden_dim_B = self.base_model.get_hidden_dim(module_B)
            else:
                logger.warning(
                    f"WARNING: get_hidden_dim() is not defined, "
                    f"which is used to get the hidden dim for different lora modules"
                    f"Use the default one, but please check if it is correct for your model."
                )
                _, hidden_dim_B = get_hidden_dim(module_B, self.base_hf_config)
            c = self.loras[-1].get_stacked_multiply(module_B)
            if module_B not in self.B_buffer:
                self.B_buffer[module_B] = [
                    torch.empty(
                        (
                            self.max_loras_per_batch,
                            hidden_dim_B * c,
                            self.max_lora_dim,
                        ),
                        dtype=self.dtype,
                        device="cuda",
                    )
                    for i in range(num_layer)
                ]

    def init_lora_batch(self):
        self.active_uids = set()  # set of active loras
        self.buffer_id = {}  # lora uid -> idx in memory pool

    def get_weight_name(self, name, idx):
        for target_weight_name in self.target_weights:
            if target_weight_name[idx] in name:
                return target_weight_name[idx]

    def load_lora(self, uid, buffer_id):
        num_layer = self.base_hf_config.num_hidden_layers
        if uid is None:
            for i in range(num_layer):
                for k in self.A_buffer.keys():
                    self.A_buffer[k][i][buffer_id] *= 0
            return

        for i in range(num_layer):
            layer_weights = self.loras[self.lora_id[uid]].layers[i].weights
            for name, weights in layer_weights.items():
                if "lora_A" in name:
                    lora_weight_name = self.get_weight_name(name, 0)
                    if lora_weight_name:
                        self.A_buffer[lora_weight_name][i][buffer_id].copy_(weights)
                else:
                    lora_weight_name = self.get_weight_name(name, 1)
                    if lora_weight_name:
                        self.B_buffer[lora_weight_name][i][buffer_id].copy_(weights)

    def prepare_lora_batch(self, forward_batch: ForwardBatch):
        # load active loras into lora memory pool
        cur_uids = set(forward_batch.lora_paths)
        assert len(cur_uids) <= self.max_loras_per_batch
        i = 0
        evictable_uids = list(self.active_uids)
        for uid in cur_uids:
            if uid not in self.active_uids:
                while i < len(evictable_uids) and evictable_uids[i] in cur_uids:
                    i += 1
                if i < len(evictable_uids):
                    self.active_uids.remove(evictable_uids[i])
                    self.buffer_id.pop(evictable_uids[i])
                self.load_lora(uid, i)
                self.active_uids.add(uid)
                self.buffer_id[uid] = i
                i += 1

        if cur_uids == set([None]):
            return

        # setup lora in forward modules
        bs = forward_batch.batch_size
        seg_lens = (
            forward_batch.extend_seq_lens
            if forward_batch.forward_mode.is_extend()
            else torch.ones(bs)
        )
        weight_indices = torch.empty((bs,), dtype=torch.int64, device="cuda")
        for i, lora_path in enumerate(forward_batch.lora_paths):
            weight_indices[i] = self.buffer_id[lora_path]

        for module_name, module in self.lora_modules:
            layer_id = get_layer_id(module_name)

            if "qkv_proj" not in module_name:
                weight_name = self.get_weight_name(module_name, 0)
                module.set_lora_info(
                    self.A_buffer[weight_name][layer_id],
                    self.B_buffer[weight_name][layer_id],
                    bs,
                    seg_lens,
                    weight_indices,
                )
            else:
                module.set_lora_info(
                    self.A_buffer["qkv_proj"][layer_id],
                    self.B_buffer["q_proj"][layer_id],
                    self.B_buffer["kv_proj"][layer_id],
                    bs,
                    seg_lens,
                    weight_indices,
                )
