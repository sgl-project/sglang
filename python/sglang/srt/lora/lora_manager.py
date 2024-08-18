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


import re
from dataclasses import dataclass

import torch
from flashinfer import SegmentGEMMWrapper

from sglang.srt.lora.lora import LoRAAdapter, get_lora_layer, params_mapping
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import replace_submodule


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
        self.target_modules = set()
        for path in self.lora_paths:
            self.configs[path] = LoRAConfig(path)
            self.target_modules = set(self.target_modules) | set(
                self.configs[path].target_modules
            )
        self.target_modules = set(
            [params_mapping(module) for module in self.target_modules]
        )

        # load all weights to cpu
        self.loras = []
        self.lora_id = {}
        for path in self.lora_paths:
            self.lora_id[path] = len(self.loras)
            self.loras.append(
                LoRAAdapter(
                    path, self.configs[path], self.base_hf_config, self.load_config
                )
            )
            self.loras[-1].initialize_weights()

        # misc lora configs
        self.max_lora_dim = max([x.hf_config["r"] for x in self.configs.values()])
        self.scaling = self.loras[0].scaling
        # FIXME remove the restrictions
        assert all(x.hf_config["r"] == self.max_lora_dim for x in self.configs.values())
        assert all(x.scaling == self.scaling for x in self.loras)

        # monkey patch to use Lora version
        modules = self.get_target_modules()
        self.lora_modules = []
        for module_name, module in modules:
            self.lora_modules.append(
                (module_name, self.set_lora_module(module_name, module))
            )

    def init_lora_memory_pool(self):
        # preallocate lora memory pool
        self.A_buffer = {}
        self.B_buffer = {}
        num_layer = self.base_hf_config.num_hidden_layers
        for module in self.target_modules:
            c = self.loras[-1].get_stacked_multiply(module)
            hidden_dim_A, hidden_dim_B = self.base_model.get_hidden_dim(module)
            # init A tensor, column_major=True
            self.A_buffer[module] = [
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
            self.B_buffer[module] = [
                torch.empty(
                    (
                        self.max_loras_per_batch,
                        hidden_dim_B,
                        self.max_lora_dim * c,
                    ),
                    dtype=self.dtype,
                    device="cuda",
                )
                for i in range(num_layer)
            ]

    def init_lora_batch(self):
        self.active_uids = [None] * self.max_loras_per_batch  # list of active loras
        self.buffer_id = {}  # lora uid -> idx in memory pool

    def get_target_module_name(self, module_name):
        for module in self.target_modules:
            if module in module_name:
                return module

    def load_lora(self, uid, buffer_id):
        num_layer = self.base_hf_config.num_hidden_layers
        for i in range(num_layer):
            layer_weights = self.loras[self.lora_id[uid]].layers[i].weights
            for module_name, weights in layer_weights.items():
                target_module_name = self.get_target_module_name(module_name)
                if "lora_A" in module_name:
                    self.A_buffer[target_module_name][i][buffer_id].copy_(weights)
                else:
                    assert "lora_B" in module_name
                    self.B_buffer[target_module_name][i][buffer_id].copy_(weights)

    def prepare_lora_batch(
        self, batch, forward_mode: ForwardMode, extend_seq_lens=None
    ):
        # load active loras into lora memory pool
        cur_uids = set([req.lora_path for req in batch.reqs])
        assert len(cur_uids) <= self.max_loras_per_batch
        i = 0
        for uid in cur_uids:
            if uid not in self.active_uids:
                while self.active_uids[i] in cur_uids:
                    i += 1
                self.load_lora(uid, i)
                if self.active_uids[i] is not None:
                    self.buffer_id.pop(self.active_uids[i])
                self.active_uids[i] = uid
                self.buffer_id[uid] = i

        if None in cur_uids:
            return

        # setup lora in forward modules
        bs = len(batch.reqs)
        if forward_mode == ForwardMode.EXTEND:
            seg_lens = extend_seq_lens
        else:
            seg_lens = torch.ones(bs)
        weight_indices = torch.empty((bs,), dtype=torch.int64, device="cuda")
        for i, req in enumerate(batch.reqs):
            weight_indices[i] = self.buffer_id[req.lora_path]

        for module_name, module in self.lora_modules:
            target_model_name = self.get_target_module_name(module_name)
            match = re.search(r"layers\.(\d+)\.", module_name)
            layer_id = int(match.group(1))
            module.set_lora_info(
                self.A_buffer[target_model_name][layer_id],
                self.B_buffer[target_model_name][layer_id],
                bs,
                seg_lens,
                weight_indices,
            )
