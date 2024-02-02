import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
from torch import nn
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)

from sglang.srt.managers.router.infer_batch import ForwardMode
from sglang.srt.managers.router.model_runner import InputMetadata


class BaseLayerWithLoRA(nn.Module):
    def create_lora_weights() -> None:
        """Initializes lora matrices."""
        ...

    def reset_lora():
        """Resets the lora weights at index back to 0."""
        ...

    def set_lora(
    ):
        """Overwrites lora tensors at index."""
        ...

    def set_mapping(
    ):
        """Sets the mapping indices."""
        ...


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer

    def forward(self):
        self.base_layer.forward()


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer

    def forward(self):
        self.base_layer.forward()


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)


class QKVParallelLinearWithLora(ColumnParallelLinearWithLoRA):
    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer

    def forward(self):
        self.base_layer.forward()


def get_lora_layer(
        layer: nn.Module,
    ) -> BaseLayerWithLoRA:
    supported_layer_types = {
        VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLora,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if type(layer) is src_layer_type:  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer)
            return ret
    return layer


def params_mapping(module_name):
    params_mapping = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
    }
    if module_name in params_mapping:
        return params_mapping[module_name]
    return module_name


def get_mapped_params(module_names):
    ret = set()
    for module_name in module_names:
        ret.add(params_mapping(module_name))
    return list(ret)


paged_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


def is_paged_module(name):
    pattern = "|".join(paged_modules)
    return re.search(r"{pattern}".format(pattern=pattern), name)


class LoRALayer(nn.Module):
    def __init__(self, config, base_config):
        self.config = config
        self.base_config = base_config
        self.weights = {}
        self.weight_gpu = {}

    def consolidate_weights(self):
        r = self.config["r"]
        num_head = self.base_config.num_attention_heads

        w_list = []
        for paged_module in paged_modules:
            for name in sorted(self.weights):
                if paged_module in name:
                    if self.weights[name].shape[0] == r:
                        w_list.append(self.weights[name].reshape(r, num_head, -1))
                    else:
                        w_list.append(self.weights[name].permute([1, 0]).reshape(r, num_head, -1))

        self.w_combined_home = torch.concat(w_list).reshape(len(w_list) * r // 2, 2, num_head, -1).pin_memory()
        self.w_combined = None

    def load_to_gpu(self, mode="paged"):
        # TODO: add dtype as an option
        for name, weight in self.weights.items():
            if mode == "paged" and is_paged_module(name):
                self.w_combined = self.w_combined_home.to(torch.float16).to("cuda")
            elif mode == "no-page" and not is_paged_module(name):
                self.weight_gpu[name] = weight.to(torch.float16).to("cuda")

    def offload_from_gpu(self, mode="paged"):
        for name, weight in self.weights.items():
            if mode == "paged" and is_paged_module(name):
                self.w_combined = None
            elif mode == "no-page" and not is_paged_module(name):
                self.weight_gpu[name] = None


class LoRAAdapter(nn.Module):
    def __init__(self, uid, config, base_config):
        super().__init__()

        self.uid = uid
        self.config = config
        self.base_config = base_config
        self.r = config["r"]
        self.lora_alpha = config["lora_alpha"]
        self.scaling = self.lora_alpha / self.r
        self.paged_modules = set(paged_modules) & set(config["target_modules"])

        self.layers = nn.ModuleList(
            [
                LoRALayer(config, base_config)
                for i in range(base_config.num_hidden_layers)
            ]
        )

    def load_to_gpu(self, mode):
        for name, weight in self.weights.items():
            weight.to("cuda")
        for layer in self.layers:
            layer.load_to_gpu(mode=mode)

    # initialize the LoRA weights to cpu
    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        self.weights = {}
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            match = re.search(r"layers\.(\d+)\.", name)
            if match is not None:
                layer_id = int(match.group(1))
                self.layers[layer_id].weights[name] = loaded_weight.cpu()
            else:
                self.weights[name] = loaded_weight.cpu()
        for layer in self.layers:
            layer.consolidate_weights()
