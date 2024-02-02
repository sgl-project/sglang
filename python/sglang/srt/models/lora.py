# LoRA layers class inheritance adapted from:
# https://github.com/vllm-project/vllm/blob/4abf6336ec65c270343eb895e7b18786e9274176/vllm/lora/layers.py
# To be refactored.
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
from sglang._kernels import dispatch_bgmv


class BaseLayerWithLoRA(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        if not hasattr(self, "forward"):
             self.forward = self.base_layer.forward

    def set_lora_info(self, *args):
        ...


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__(base_layer)


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__(base_layer)

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)
        return output

    def forward(self, input_):
        # duplicate the logic in ColumnParallelLinear
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)

        # Matrix multiply.
        output_parallel = self.apply_weights(input_, bias)
        if self.base_layer.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)
        return output, output_bias


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)


def _apply_lora_qkv(
        x: torch.Tensor,
        output: torch.Tensor,
        input_metadata,
        infer_adapter,
        lora_uids,
        max_lora_dim,
        key_buffer,
        value_buffer,
        layer_id,
    ) -> torch.Tensor:
    # move the below to lora_manager
    req_bins = torch.zeros(len(lora_uids), dtype=torch.long, device="cuda")
    for i, lora_uid in enumerate(lora_uids):
        # FIX ME @TODO: currently not supporting adapter is None
        if lora_uid is None: continue
        idx = infer_adapter.adapter_uids.index(lora_uid)
        req_bins[i] = idx # TODO single point access!
    seq_lens = input_metadata.seq_lens
    assert req_bins.shape[0] == seq_lens.shape[0]
    batch_req_bins = torch.repeat_interleave(req_bins, seq_lens)

    # do not initiate a delta for each layer
    delta = []
    for _ in range(3):
        delta.append(torch.zeros((len(batch_req_bins), max_lora_dim),
                                      dtype=torch.float16, device="cuda"))
    # x: (b, h)
    # output: (b, 3h)
    if input_metadata.forward_mode != ForwardMode.DECODE:
        delta_qA = delta[0]
        dispatch_bgmv(delta_qA, x, 
                      key_buffer[layer_id][:, 0],
                      infer_adapter.a_start, infer_adapter.a_len, 
                      infer_adapter.a_loc, batch_req_bins, 0, infer_adapter.a_scaling)
        q_idx = (0, output.shape[1] // 3)
        dispatch_bgmv(output[:,q_idx[0]:q_idx[1]], delta_qA,
                      value_buffer[layer_id][:, 1], infer_adapter.a_start, 
                      infer_adapter.a_len, infer_adapter.a_loc, 
                      batch_req_bins, 0, infer_adapter.a_scaling)
    else:
        # TODO
        pass


class QKVParallelLinearWithLora(ColumnParallelLinearWithLoRA):
    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)

    def set_lora_info(self, input_metadata, infer_adapter, lora_uids, max_lora_dim, layer_id):
        self.input_metadata = input_metadata
        self.infer_adapter = infer_adapter
        self.lora_uids = lora_uids
        self.max_lora_dim = max_lora_dim
        self.layer_id = layer_id

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)
        _apply_lora_qkv(
            x,
            output,
            self.input_metadata,
            self.infer_adapter,
            self.lora_uids,
            self.max_lora_dim,
            self.infer_adapter.token_to_kv_pool.kv_data,
            self.infer_adapter.token_to_kv_pool.kv_data,
            self.layer_id,
        )
        return output


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__(base_layer)

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x)
        # TODO change below
        # _apply_lora(
        #     x,
        #     self.lora_a_stacked,
        #     self.lora_b_stacked,
        #     self.indices[:self.indices_len[0]],
        #     output,
        # )
        return output

    def forward(self, input_):
        # duplicate the logic in RowParallelLinear
        # Set up backprop all-reduce.
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        output_parallel = self.apply_weights(input_parallel)
        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (output_ + self.base_layer.bias
                      if self.base_layer.bias is not None else output_)
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias


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
