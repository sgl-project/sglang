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

# LoRA layers class inheritance adapted from:
# https://github.com/vllm-project/vllm/blob/4abf6336ec65c270343eb895e7b18786e9274176/vllm/lora/layers.py


import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
from torch import nn
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.loader import DefaultModelLoader

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class BaseLayerWithLoRA(nn.Module):
    def __init__(self, base_layer, segment_gemm, lora_rank, scaling):
        super().__init__()
        self.base_layer = base_layer
        self.segment_gemm = segment_gemm
        self.lora_rank = lora_rank
        self.scaling = scaling
        self.set_lora = False

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self, base_layer: VocabParallelEmbedding, segment_gemm, lora_rank, scaling
    ) -> None:
        super().__init__(base_layer, segment_gemm, lora_rank, scaling)
        self.weight = base_layer.weight


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self, base_layer: ColumnParallelLinear, segment_gemm, lora_rank, scaling
    ) -> None:
        super().__init__(base_layer, segment_gemm, lora_rank, scaling)

    def apply_lora(self, output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # TODO
        return output

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in ColumnParallelLinear
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_)

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self, base_layer: MergedColumnParallelLinear, segment_gemm, lora_rank, scaling
    ) -> None:
        super().__init__(base_layer, segment_gemm, lora_rank, scaling)

    def set_lora_info(self, A_buffer, B_buffer, bs, seg_indptr, weight_indices):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        self.bs = bs
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.segment_gemm.run(
            x=x,
            weights=self.A_buffer,
            batch_size=self.bs,
            weight_column_major=True,
            seg_indptr=self.seg_indptr,
            weight_indices=self.weight_indices,
        )
        # FIXME
        lora_output = torch.empty_like(base_output)
        output_dim = lora_output.shape[-1] // 2
        for i in range(2):
            left = output_dim * i
            right = left + output_dim
            lora_output[:, left:right] = self.segment_gemm.run(
                x=lora_a_output[
                    :, self.lora_rank * i : self.lora_rank * (i + 1)
                ].contiguous(),
                weights=self.B_buffer[:, left:right, :].contiguous(),
                batch_size=self.bs,
                weight_column_major=True,
                seg_indptr=self.seg_indptr,
                weight_indices=self.weight_indices,
            )
        return base_output + lora_output * self.scaling


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self, base_layer: QKVParallelLinear, segment_gemm, lora_rank, scaling
    ) -> None:
        super().__init__(base_layer, segment_gemm, lora_rank, scaling)

    def set_lora_info(
        self, A_buffer_qkv, B_buffer_q, B_buffer_kv, bs, seg_indptr, weight_indices
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv
        self.B_buffer_q = B_buffer_q
        self.B_buffer_kv = B_buffer_kv
        self.bs = bs
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.segment_gemm.run(
            x=x,
            weights=self.A_buffer_qkv,
            batch_size=self.bs,
            weight_column_major=True,
            seg_indptr=self.seg_indptr,
            weight_indices=self.weight_indices,
        )
        # FIXME parallelize qkv
        lora_output = torch.empty_like(base_output)
        # q
        output_dim_q = self.B_buffer_q.shape[-2]
        lora_output[:, :output_dim_q] = self.segment_gemm.run(
            x=lora_a_output[:, : self.lora_rank].contiguous(),
            weights=self.B_buffer_q,
            batch_size=self.bs,
            weight_column_major=True,
            seg_indptr=self.seg_indptr,
            weight_indices=self.weight_indices,
        )
        # kv
        output_dim_kv = self.B_buffer_kv.shape[-2] // 2
        for i in range(2):
            left = output_dim_kv * i
            right = left + output_dim_kv
            lora_output[:, output_dim_q + left : output_dim_q + right] = (
                self.segment_gemm.run(
                    x=lora_a_output[
                        :, self.lora_rank * (i + 1) : self.lora_rank * (i + 2)
                    ].contiguous(),
                    weights=self.B_buffer_kv[:, left:right, :].contiguous(),
                    batch_size=self.bs,
                    weight_column_major=True,
                    seg_indptr=self.seg_indptr,
                    weight_indices=self.weight_indices,
                )
            )
        return base_output + lora_output * self.scaling


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self, base_layer: RowParallelLinear, segment_gemm, lora_rank, scaling
    ) -> None:
        super().__init__(base_layer, segment_gemm, lora_rank, scaling)

    def set_lora_info(self, A_buffer, B_buffer, bs, seg_indptr, weight_indices):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        self.bs = bs
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.segment_gemm.run(
            x=x,
            weights=self.A_buffer,
            batch_size=self.bs,
            weight_column_major=True,
            seg_indptr=self.seg_indptr,
            weight_indices=self.weight_indices,
        )
        lora_output = self.segment_gemm.run(
            x=lora_output,
            weights=self.B_buffer,
            batch_size=self.bs,
            weight_column_major=True,
            seg_indptr=self.seg_indptr,
            weight_indices=self.weight_indices,
        )
        return base_output + lora_output * self.scaling

    def forward(self, input_):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_parallel)

        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias


def get_lora_layer(
    layer: nn.Module, segment_gemm, lora_rank, scaling
) -> BaseLayerWithLoRA:
    supported_layer_types = {
        # the order matters
        VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer, segment_gemm, lora_rank, scaling)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")


def get_mapped_params(module_names):
    ret = set()
    for module_name in module_names:
        ret.add(params_mapping(module_name))
    return list(ret)


class LoRALayer(nn.Module):
    def __init__(self, config, base_hf_config):
        super().__init__()
        self.config = config
        self.base_hf_config = base_hf_config
        self.weights = {}
        self.weight_gpu = {}

    def load_to_gpu(self):
        for name, weight in self.weights.items():
            self.weight_gpu[name] = weight.to(torch.float16).to("cuda")

    def offload_from_gpu(self):
        for name, weight in self.weights.items():
            self.weight_gpu[name] = None


class LoRAAdapter(nn.Module):
    def __init__(self, uid, config, base_hf_config, load_config):
        super().__init__()
        self.uid = uid
        self.config = config
        assert self.config.hf_config["peft_type"].lower() == "lora"
        self.base_hf_config = base_hf_config
        self.load_config = load_config
        self.scaling = self.config.lora_alpha / self.config.r

        self.layers = nn.ModuleList(
            [
                LoRALayer(config, base_hf_config)
                for i in range(base_hf_config.num_hidden_layers)
            ]
        )

        self.weights = {}
        self.weights_gpu = {}

    def get_stacked_multiply(self, module_name):
        stacked_rank = {
            "qkv_proj": 3,
            "kv_proj": 2,
            "gate_up_proj": 2,
        }
        return stacked_rank[module_name] if module_name in stacked_rank else 1

    def load_to_gpu(self):
        for name, weight in self.weights.items():
            self.weights_gpu[name] = weight.to(torch.float16).to("cuda")
        for layer in self.layers:
            layer.load_to_gpu()

    def offload_from_gpu(self):
        for name, weight in self.weights.items():
            self.weights_gpu[name] = None
        for layer in self.layers:
            layer.offload_from_gpu()

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
            for weight_name in weight_names:
                if "k_proj" in weight_name:
                    q_name = weight_name.replace("k_proj", "q_proj")
                    v_name = weight_name.replace("k_proj", "v_proj")
                    kv_name = weight_name.replace("k_proj", "kv_proj")
                    qkv_name = weight_name.replace("k_proj", "qkv_proj")
                    if "lora_A" in weight_name:
                        layer.weights[qkv_name] = torch.cat(
                            (
                                layer.weights[q_name],
                                layer.weights[weight_name],
                                layer.weights[v_name],
                            ),
                            0,
                        )
                        layer.weights.pop(q_name)
                        layer.weights.pop(weight_name)
                        layer.weights.pop(v_name)
                    else:
                        layer.weights[kv_name] = torch.cat(
                            (
                                layer.weights[weight_name],
                                layer.weights[v_name],
                            ),
                            0,
                        )
                        layer.weights.pop(weight_name)
                        layer.weights.pop(v_name)
                elif "gate_proj" in weight_name:
                    up_name = weight_name.replace("gate_proj", "up_proj")
                    gate_up_name = weight_name.replace("gate_proj", "gate_up_proj")
                    layer.weights[gate_up_name] = torch.cat(
                        (layer.weights[weight_name], layer.weights[up_name]), 0
                    )
                    layer.weights.pop(weight_name)
                    layer.weights.pop(up_name)
