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

import re
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.model_loader.loader import DefaultModelLoader


@dataclass
class LoRABatchInfo:
    # Batch size
    bs: int

    # Lengths of each sequence in shape (bs,)
    seg_lens: torch.Tensor

    # Indice pointers of each sequence in shape (bs + 1, )
    seg_indptr: torch.Tensor

    # Maximum sequence length of current batch
    max_len: int

    # The index of lora adapter used by each sequence, in shape (bs,)
    weight_indices: torch.Tensor


class BaseLayerWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        lora_rank: int,
        scaling: float,
        lora_backend: "BaseLoRABackend",
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.lora_rank: int = lora_rank
        self.scaling: float = scaling
        self.set_lora: bool = False
        self.lora_backend: "BaseLoRABackend" = lora_backend

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
        lora_rank: int,
        scaling: float,
        lora_backend: "BaseLoRABackend",
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)
        self.weight = base_layer.weight


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        lora_rank: int,
        scaling: float,
        lora_backend: "BaseLoRABackend",
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def apply_lora(self, output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # FIXME Implement this!
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
        self,
        base_layer: MergedColumnParallelLinear,
        lora_rank: int,
        scaling: float,
        lora_backend: "BaseLoRABackend",
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x=x, weights=self.A_buffer)

        output_dim = base_output.shape[-1]
        lora_output = torch.empty_like(base_output)
        lora_output[:, :output_dim] = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output[:, 0 : self.lora_rank].contiguous(),
            weights=self.B_buffer[0],
        )

        lora_output[:, output_dim : 2 * output_dim] = (
            self.lora_backend.run_lora_b_sgemm(
                x=lora_a_output[:, self.lora_rank : 2 * self.lora_rank].contiguous(),
                weights=self.B_buffer[1],
            )
        )

        return base_output + lora_output * self.scaling


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def init__(
        self,
        base_layer: QKVParallelLinear,
        lora_rank: int,
        scaling: float,
        lora_backend: "BaseLoRABackend",
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def set_lora_info(
        self,
        A_buffer_qkv: torch.Tensor,
        B_buffer_q: torch.Tensor,
        B_buffer_kv: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv

        if self.lora_backend.fuse_qkv_lora_b:
            assert (
                B_buffer_q.shape[-1] == B_buffer_kv.shape[-1]
            ), "The lora rank of q and kv should be the same when enabling fusion of qkv lora_b"
            output_dim_q, output_dim_kv = B_buffer_q.shape[-2], B_buffer_kv.shape[-2]

            # B_buffer_qkv: (num_lora, output_dim_q + 2 * output_dim_kv, r)
            self.B_buffer_qkv = torch.cat(
                (B_buffer_q[0], B_buffer_kv[0], B_buffer_kv[1]), dim=-2
            ).contiguous()

            # Offsets of q/k/v in output dimension
            self.output_offset = torch.tensor(
                [
                    0,
                    output_dim_q,
                    output_dim_q + output_dim_kv,
                    output_dim_q + 2 * output_dim_kv,
                ],
                dtype=torch.int32,
                device=B_buffer_q.device,
            )
            # For computing number of launched blocks
            self.max_qkv_out_dim = max(output_dim_q, output_dim_kv)
        else:
            self.B_buffer_qkv = (
                B_buffer_q,
                B_buffer_kv,
            )

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output, "scaling": self.scaling}
        if self.lora_backend.fuse_qkv_lora_b:
            backend_kwargs["output_offset"] = self.output_offset
            backend_kwargs["max_qkv_out_dim"] = self.max_qkv_out_dim

        lora_output = self.lora_backend.run_qkv_lora(
            x,
            self.A_buffer_qkv,
            self.B_buffer_qkv,
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: RowParallelLinear,
        lora_rank: int,
        scaling: float,
        lora_backend: "BaseLoRABackend",
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def set_lora_info(self, A_buffer: torch.Tensor, B_buffer: torch.Tensor):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output, "scaling": self.scaling}
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            lora_a_output,
            self.B_buffer[0],
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )

    def forward(self, input_: torch.Tensor):
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
    layer: nn.Module, lora_rank: int, scaling: int, lora_backend: "BaseLoRABackend"
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
            ret = lora_layer_type(layer, lora_rank, scaling, lora_backend)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")


class LoRALayer(nn.Module):
    def __init__(self, config: LoRAConfig, base_hf_config: AutoConfig):
        super().__init__()
        self.config: LoRAConfig = config
        self.base_hf_config: AutoConfig = base_hf_config
        self.weights: Dict[str, torch.Tensor] = {}
        self.weight_gpu: Dict[str, torch.Tensor] = {}

    def load_to_gpu(self):
        for name, weight in self.weights.items():
            self.weight_gpu[name] = weight.to(torch.float16).to("cuda")

    def offload_from_gpu(self):
        for name, weight in self.weights.items():
            self.weight_gpu[name] = None


class LoRAAdapter(nn.Module):
    def __init__(
        self,
        uid: str,
        config: LoRAConfig,
        base_hf_config: AutoConfig,
        load_config: LoadConfig,
        lora_backend: "BaseLoRABackend",
    ):
        super().__init__()
        self.uid: str = uid
        self.config: LoRAConfig = config
        assert self.config.hf_config["peft_type"].lower() == "lora"
        self.base_hf_config: AutoConfig = base_hf_config
        self.load_config: LoadConfig = load_config
        self.lora_backend: "BaseLoRABackend" = lora_backend
        self.scaling: float = self.config.lora_alpha / self.config.r

        self.layers: List[LoRALayer] = nn.ModuleList(
            [
                LoRALayer(config, base_hf_config)
                for i in range(base_hf_config.num_hidden_layers)
            ]
        )

        self.weights: Dict[str, torch.Tensor] = {}
        self.weights_gpu: Dict[str, torch.Tensor] = {}

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
                if "lora_A" in weight_name:
                    weights[gate_up_name] = torch.cat(
                        (weights[weight_name], weights[up_name]), 0
                    )
                else:
                    weights[gate_up_name] = torch.stack(
                        [weights[weight_name], weights[up_name]], dim=0
                    )
                weights.pop(weight_name)
                weights.pop(up_name)
