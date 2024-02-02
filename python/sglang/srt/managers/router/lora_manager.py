import re
from dataclasses import dataclass

import torch

from sglang.srt.managers.router.infer_adapter import InferAdapter
from sglang.srt.managers.router.infer_batch import ForwardMode
from sglang.srt.memory_pool import TokenToKVPool
from sglang.srt.model_config import LoRAConfig
from sglang.srt.models.lora import get_lora_layer, params_mapping, LoRAAdapter
from sglang.srt.utils import replace_submodule


class LoRAManager:
    def __init__(
        self,
        base_model,
        lora_paths,
        base_config,
        token_to_kv_pool,
    ):
        self.base_model = base_model
        self.lora_paths = lora_paths
        self.base_config = base_config

        self.infer_adapter = InferAdapter.init(token_to_kv_pool)
        self.init_loras()

    def match_target_modules(self, module_name):
        for target_module in self.target_modules:
            target_module = params_mapping(target_module)
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
        lora_module = get_lora_layer(module)
        replace_submodule(self.base_model, module_name, lora_module)
        # TODO disable radixattention
        return lora_module

    def init_loras(self):
        # get configs and target modules
        self.configs = {}
        self.target_modules = set()
        for path in self.lora_paths:
            self.configs[path] = LoRAConfig(path)
            self.target_modules = set(self.target_modules) | set(self.configs[path].target_modules)
        self.target_modules = set([params_mapping(module) for module in self.target_modules])

        # monkey patch to use Lora version
        modules = self.get_target_modules()
        self.lora_modules = []
        for module_name, module in modules:
            self.lora_modules.append((module_name, self.set_lora_module(module_name, module)))

        # load all weights to cpu
        self.loras = []
        self.lora_id = {}
        for path in self.lora_paths:
            self.lora_id[path] = len(self.loras)
            self.loras.append(LoRAAdapter(path, self.configs[path].config, self.base_config))
            self.loras[-1].load_weights(path)

        # load zero lora for base model only
        self.infer_adapter.add_zero_lora()

        # other helper info
        self.max_lora_dim = max([x.config["r"] for x in self.configs.values()])

    def load_loras_from_path(self, lora_paths):
        loras = [self.loras[self.lora_id[path]] for path in lora_paths]
        # load paged modules
        self.infer_adapter.load(loras)
        # load unpaged modules
        for lora in loras:
            lora.load_to_gpu(mode="no-page")

    def set_lora_input_metadata(self, input_metadata, lora_uids):
        for module_name, lora_module in self.lora_modules:
            match = re.search(r"model\.layers\.(\d+)\.*", module_name)
            layer_id = int(match.group(1)) if match else -1
            lora_module.set_lora_info(
                # TODO do not need to use the global max_lora_dim
                input_metadata, self.infer_adapter, lora_uids, self.max_lora_dim, layer_id)
