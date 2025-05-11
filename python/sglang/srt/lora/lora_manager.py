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

import logging
from typing import Dict, List, Set, Tuple

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.lora.backend.base_backend import BaseLoRABackend, get_backend_from_name
from sglang.srt.lora.layers import BaseLayerWithLoRA, get_lora_layer
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.lora.utils import (
    LoRABatchInfo,
    LoRAType,
    get_customized_names_from_hf_names,
    get_layer_id,
    get_stacked_name,
    get_weight_name,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import replace_submodule

logger = logging.getLogger(__name__)


class LoRAManager:
    def __init__(
        self,
        base_model: torch.nn.Module,
        lora_paths: Dict[str, str],
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        load_config: LoadConfig,
        dtype: torch.dtype,
        lora_backend: str = "triton",
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        self.base_model: torch.nn.Module = base_model
        self.lora_paths: Dict[str, str] = lora_paths
        self.base_hf_config: AutoConfig = base_hf_config
        self.max_loras_per_batch: int = max_loras_per_batch
        self.load_config: LoadConfig = load_config
        self.dtype: torch.dtype = dtype
        self.device: torch.device = next(self.base_model.parameters()).device
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank

        # LoRA backend for running sgemm kernels
        logger.info(f"Using {lora_backend} as backend of LoRA kernels.")
        backend_type = get_backend_from_name(lora_backend)
        self.lora_backend: BaseLoRABackend = backend_type(lora_backend)

        self.init_loras()
        self.init_lora_memory_pool()

    def init_cuda_graph_batch_info(self, max_bs_in_cuda_graph: int):
        self.max_bs_in_cuda_graph = max_bs_in_cuda_graph
        with torch.device("cuda"):
            self.cuda_graph_batch_info = LoRABatchInfo(
                bs=self.max_bs_in_cuda_graph,
                seg_lens=torch.zeros(self.max_bs_in_cuda_graph, dtype=torch.int32),
                seg_indptr=torch.zeros(
                    self.max_bs_in_cuda_graph + 1, dtype=torch.int32
                ),
                max_len=0,
                weight_indices=torch.zeros(
                    self.max_bs_in_cuda_graph, dtype=torch.int32
                ),
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),
            )

    def init_loras(self):
        # Config of each LoRA adapter
        self.configs: Dict[str, LoRAConfig] = {}

        # Target module names in huggingface lora configs.
        # e.g., {"k_proj", "q_proj", "v_proj", "o_proj"}
        self.hf_target_names: Set[str] = set()
        for name, path in self.lora_paths.items():
            self.configs[name] = LoRAConfig(path)
            self.hf_target_names.update(self.configs[name].target_modules)

        # Target lora weight names for lora_a and lora_b modules respectively.
        # e.g., {("qkv_proj", "q_proj"), ("qkv_proj", "kv_proj")}
        self.lora_weight_names: Set[Tuple[str]] = set(
            [get_stacked_name(module) for module in self.hf_target_names]
        )

        # load all weights to cpu
        self.loras: Dict[str, LoRAAdapter] = {}
        for name in self.lora_paths.keys():
            lora_adapter = LoRAAdapter(
                name,
                self.configs[name],
                self.base_hf_config,
                self.load_config,
                self.lora_backend,
            )
            lora_adapter.initialize_weights()
            self.loras[name] = lora_adapter

        # misc lora configs
        self.max_lora_dim: int = max([x.hf_config["r"] for x in self.configs.values()])

        if self.lora_backend == "flashinfer":
            # FIXME remove the restrictions after supporting multi-rank for flashinfer backend
            max_lora_dim = max([x.hf_config["r"] for x in self.configs.values()])
            scaling = list(self.loras.values())[0].scaling
            assert all(x.hf_config["r"] == max_lora_dim for x in self.configs.values())
            assert all(x.scaling == scaling for x in self.loras.values())

        # Convert original model layers to layers with LoRA
        self.convert_to_lora_layers()

    def init_lora_memory_pool(self):
        # Initialize memory pool
        self.memory_pool = LoRAMemoryPool(
            self.base_hf_config,
            self.max_loras_per_batch,
            self.max_lora_dim,
            self.dtype,
            self.tp_size,
            self.tp_rank,
            self.lora_modules,
        )

        # Initialize target lora modules in memory pool
        self.memory_pool.init_buffers(self.lora_weight_names, self.base_model)

    def prepare_lora_batch(self, forward_batch: ForwardBatch):
        # load active loras into lora memory pool
        cur_uids = set(forward_batch.lora_paths)
        assert len(cur_uids) <= self.max_loras_per_batch
        self.memory_pool.prepare_lora_batch(cur_uids, self.loras)

        # set up batch info shared by all lora modules
        bs = forward_batch.batch_size

        if (
            hasattr(self, "max_bs_in_cuda_graph")
            and bs <= self.max_bs_in_cuda_graph
            and forward_batch.forward_mode.is_cuda_graph()
        ):
            # Do in-place updates when CUDA graph is enabled and the batch forward mode
            # could use CUDA graph.
            self.cuda_graph_batch_info.bs = bs
            self.cuda_graph_batch_info.seg_lens[:bs].fill_(1)
            torch.cumsum(
                self.cuda_graph_batch_info.seg_lens[:bs],
                dim=0,
                out=self.cuda_graph_batch_info.seg_indptr[1 : bs + 1],
            )
            self.cuda_graph_batch_info.max_len = int(
                torch.max(self.cuda_graph_batch_info.seg_lens[:bs])
            )

            for i, lora_path in enumerate(forward_batch.lora_paths):
                self.cuda_graph_batch_info.weight_indices[i] = (
                    self.memory_pool.get_buffer_id(lora_path)
                )
                if lora_path is not None:
                    lora = self.loras[lora_path]
                    self.cuda_graph_batch_info.lora_ranks[
                        self.cuda_graph_batch_info.weight_indices[i]
                    ] = lora.config.hf_config["r"]
                    self.cuda_graph_batch_info.scalings[
                        self.cuda_graph_batch_info.weight_indices[i]
                    ] = lora.scaling
            batch_info = self.cuda_graph_batch_info
        else:
            seg_lens = (
                forward_batch.extend_seq_lens
                if forward_batch.forward_mode.is_extend()
                else torch.ones(bs, device=self.device)
            )
            seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
            max_len = int(torch.max(seg_lens))
            weight_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)

            lora_ranks = torch.zeros(
                (self.max_loras_per_batch,), dtype=torch.int64, device="cuda"
            )
            scalings = torch.zeros(
                (self.max_loras_per_batch,), dtype=torch.float, device="cuda"
            )
            for i, lora_path in enumerate(forward_batch.lora_paths):
                weight_indices[i] = self.memory_pool.get_buffer_id(lora_path)
                if lora_path is not None:
                    lora = self.loras[lora_path]
                    lora_ranks[weight_indices[i]] = lora.config.hf_config["r"]
                    scalings[weight_indices[i]] = lora.scaling
            batch_info = LoRABatchInfo(
                bs=bs,
                seg_lens=seg_lens,
                seg_indptr=seg_indptr,
                max_len=max_len,
                weight_indices=weight_indices,
                lora_ranks=lora_ranks,
                scalings=scalings,
            )
        self.lora_backend.set_batch_info(batch_info)

        # call set_lora_info for each lora modules
        for layer_id, modules in self.lora_modules.items():
            for module_name, module in modules:
                if "qkv_proj" in module_name:
                    module.set_lora_info(
                        self.memory_pool.get_tensor(
                            "qkv_proj", layer_id, LoRAType.LORA_A
                        ),
                        self.memory_pool.get_tensor(
                            "q_proj", layer_id, LoRAType.LORA_B
                        ),
                        self.memory_pool.get_tensor(
                            "kv_proj", layer_id, LoRAType.LORA_B
                        ),
                    )
                else:
                    weight_name = get_weight_name(
                        module_name, self.lora_weight_names, LoRAType.LORA_A
                    )
                    module.set_lora_info(
                        self.memory_pool.get_tensor(
                            weight_name, layer_id, LoRAType.LORA_A
                        ),
                        self.memory_pool.get_tensor(
                            weight_name, layer_id, LoRAType.LORA_B
                        ),
                    )

    def set_lora_module(self, module_name, module):
        lora_module = get_lora_layer(module, self.lora_backend)
        replace_submodule(self.base_model, module_name, lora_module)
        return lora_module

    def convert_to_lora_layers(self):
        # Target module names of customized layers defined in python/sglang/srt/layers
        # e.g., {"qkv_proj", "o_proj"}
        customized_target_names = get_customized_names_from_hf_names(
            self.hf_target_names, self.base_model
        )

        # Monkey patch to use the LoRA version layers
        self.lora_modules: Dict[int, List[Tuple[str, BaseLayerWithLoRA]]] = {
            i: [] for i in range(self.base_hf_config.num_hidden_layers)
        }
        for module_name, module in self.base_model.named_modules():
            # The module should be converted if it is included in target_names
            if module_name.split(".")[-1] in customized_target_names:
                layer_id = get_layer_id(module_name)
                self.lora_modules[layer_id].append(
                    (module_name, self.set_lora_module(module_name, module))
                )
