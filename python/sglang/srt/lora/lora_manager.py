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
from typing import Dict, Set, Tuple

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
    get_normalized_lora_weight_names,
    get_weight_name,
)
from sglang.srt.managers.io_struct import LoRAUpdateResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import replace_submodule

logger = logging.getLogger(__name__)


class LoRAManager:
    def __init__(
        self,
        base_model: torch.nn.Module,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        load_config: LoadConfig,
        dtype: torch.dtype,
        lora_backend: str = "triton",
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        self.base_model: torch.nn.Module = base_model
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

        # Initialize mutable internal state of the LoRAManager.
        self.init_state()

    def init_cuda_graph_batch_info(self, max_bs_in_cuda_graph: int):
        self.max_bs_in_cuda_graph = max_bs_in_cuda_graph
        with torch.device("cuda"):
            self.cuda_graph_batch_info = LoRABatchInfo(
                bs=self.max_bs_in_cuda_graph,
                seg_lens=torch.zeros(self.max_bs_in_cuda_graph, dtype=torch.int32),
                seg_indptr=torch.zeros(
                    self.max_bs_in_cuda_graph + 1, dtype=torch.int32
                ),
                max_len=1,
                weight_indices=torch.zeros(
                    self.max_bs_in_cuda_graph, dtype=torch.int32
                ),
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),
            )

            # Initialize seg_lens and seg_indptr for CUDA graph as they remain constant
            # across batches.
            self.cuda_graph_batch_info.seg_lens[: self.max_bs_in_cuda_graph].fill_(1)
            torch.cumsum(
                self.cuda_graph_batch_info.seg_lens[: self.max_bs_in_cuda_graph],
                dim=0,
                out=self.cuda_graph_batch_info.seg_indptr[
                    1 : self.max_bs_in_cuda_graph + 1
                ],
            )

    def create_lora_update_result(
        self, success: bool, error_message: str = ""
    ) -> LoRAUpdateResult:
        return LoRAUpdateResult(
            success=success,
            error_message=error_message,
            loaded_adapters={
                name: config.path for name, config in self.configs.items()
            },
        )

    def load_lora_adapters(self, lora_paths: Dict[str, str]) -> LoRAUpdateResult:
        """
        Load LoRA adapters from the specified paths.

        Args:
            lora_paths (Dict[str, str]): A dictionary mapping LoRA adapter names to their file paths.
            If a LoRA adapter is already loaded, it will be skipped with a warning.
        """

        results = []
        for lora_name, lora_path in lora_paths.items():
            result = self.load_lora_adapter(lora_name, lora_path, update_state=False)
            results.append(result)

        self.update_state_from_configs()

        return self.create_lora_update_result(
            success=all(result.success for result in results),
            error_message="\n".join(
                result.error_message for result in results if not result.success
            ),
        )

    def load_lora_adapter(
        self, lora_name: str, lora_path: str, update_state: bool = True
    ) -> LoRAUpdateResult:
        """
        Load a single LoRA adapter from the specified path.

        Args:
            lora_name (str): The name of the LoRA adapter.
            lora_path (str): The file path to the LoRA adapter.
            update_state (bool): Whether to refresh the internal state after loading the adapter. This is useful for batch loading.
        """

        success = True
        error_message = ""

        if lora_name in self.loras:
            success = False
            error_message = f"LoRA adapter {lora_name} is skipped as it is already loaded. If you want to reload it, please unload it first."

        try:
            self.configs[lora_name] = LoRAConfig(lora_path)
        except Exception as e:
            success = False
            error_message = (
                f"Failed to load LoRA adapter {lora_name} from {lora_path}: {str(e)}"
            )

        if update_state:
            self.update_state_from_configs()

        return self.create_lora_update_result(
            success=success,
            error_message=error_message,
        )

    def unload_lora_adapter(self, lora_name: str) -> LoRAUpdateResult:
        """
        Unload LoRA adapters by their names. This will remove the adapters from the memory pool and
        delete the corresponding LoRA modules.
        """

        success = True
        error_message = ""
        if lora_name in self.loras:
            del self.configs[lora_name]
        else:
            error_message = f"LoRA adapter {lora_name} is not loaded."
            success = False

        self.update_state_from_configs()

        return self.create_lora_update_result(
            success=success,
            error_message=error_message,
        )

    def prepare_lora_batch(self, forward_batch: ForwardBatch):
        # load active loras into lora memory pool
        cur_uids = set(forward_batch.lora_paths)
        assert len(cur_uids) <= self.max_loras_per_batch
        self.memory_pool.prepare_lora_batch(cur_uids, self.loras, self.lora_modules)

        # set up batch info shared by all lora modules
        bs = forward_batch.batch_size

        def transfer_adapter_info(
            weight_indices_out: torch.Tensor,
            lora_ranks_out: torch.Tensor,
            scalings_out: torch.Tensor,
        ):
            """
            Transfer adapter metadata (weight indices, LoRA rank, scalings) from host
            to device (CUDA) asynchronously.
            """
            weight_indices = [0] * len(forward_batch.lora_paths)
            lora_ranks = [0] * self.max_loras_per_batch
            scalings = [0] * self.max_loras_per_batch
            for i, lora_path in enumerate(forward_batch.lora_paths):
                weight_indices[i] = self.memory_pool.get_buffer_id(lora_path)
                if lora_path is not None:
                    lora = self.loras[lora_path]
                    lora_ranks[weight_indices[i]] = lora.config.hf_config["r"]
                    scalings[weight_indices[i]] = lora.scaling

            # Use pinned memory to avoid synchronizations during host-to-device transfer
            weight_indices_tensor = torch.tensor(
                weight_indices, dtype=torch.int32, pin_memory=True, device="cpu"
            )
            lora_ranks_tensor = torch.tensor(
                lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"
            )
            scalings_tensor = torch.tensor(
                scalings, dtype=torch.float, pin_memory=True, device="cpu"
            )

            # Copy to device tensors asynchronously
            weight_indices_out[:bs].copy_(weight_indices_tensor, non_blocking=True)
            lora_ranks_out[: self.max_loras_per_batch].copy_(
                lora_ranks_tensor, non_blocking=True
            )
            scalings_out[: self.max_loras_per_batch].copy_(
                scalings_tensor, non_blocking=True
            )

        if (
            hasattr(self, "max_bs_in_cuda_graph")
            and bs <= self.max_bs_in_cuda_graph
            and forward_batch.forward_mode.is_cuda_graph()
        ):
            # Do in-place updates when CUDA graph is enabled and the batch forward mode
            # could use CUDA graph.

            transfer_adapter_info(
                self.cuda_graph_batch_info.weight_indices,
                self.cuda_graph_batch_info.lora_ranks,
                self.cuda_graph_batch_info.scalings,
            )

            self.cuda_graph_batch_info.bs = bs
            self.cuda_graph_batch_info.max_len = 1
            batch_info = self.cuda_graph_batch_info
        else:
            weight_indices = torch.empty((bs,), dtype=torch.int32, device=self.device)
            lora_ranks = torch.zeros(
                (self.max_loras_per_batch,), dtype=torch.int64, device=self.device
            )
            scalings = torch.zeros(
                (self.max_loras_per_batch,), dtype=torch.float, device=self.device
            )
            transfer_adapter_info(
                weight_indices,
                lora_ranks,
                scalings,
            )

            seg_lens = (
                forward_batch.extend_seq_lens
                if forward_batch.forward_mode.is_extend()
                else torch.ones(bs, device=self.device)
            )

            max_len = (
                # Calculate max_len from the CPU copy to avoid D2H transfer.
                max(forward_batch.extend_seq_lens_cpu)
                if forward_batch.forward_mode.is_extend()
                else 1
            )

            seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

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

        # TODO (lifuhuang): one potential perf optimization that is worth considering is to see if we can call
        # this method only when loading/unloading LoRA adapters, instead of calling it for every micro-batch.
        self.update_lora_info()

    def update_lora_info(self):
        """
        Update all LoRA modules to associate them with the latest memory buffer.
        """
        for layer_id, layer_modules in self.lora_modules.items():
            for module_name, module in layer_modules.items():
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

    def init_state(self):
        """
        Initialize the internal (mutable) state of the LoRAManager.

        These states are mutable via the `update_state_from_configs` as LoRA adapters are loaded and unloaded dynamically.
        """

        # Configs of all active LoRA adapters.
        self.configs: Dict[str, LoRAConfig] = {}

        # LoRA adapter weights cached in CPU memory.
        self.loras: Dict[str, LoRAAdapter] = {}

        # Supported weight names (e.g., qkv_proj) for LoRA A and B respectively.
        self.lora_weight_names: Tuple[Set[str]] = (set(), set())

        # Look-up table that essentially maps (layer_index, module_name) to the corresponding LoRA module.
        self.lora_modules: Dict[int, Dict[str, BaseLayerWithLoRA]] = {
            i: {} for i in range(self.base_hf_config.num_hidden_layers)
        }

        # Initialize memory pool
        self.memory_pool = LoRAMemoryPool(
            self.base_hf_config,
            self.max_loras_per_batch,
            self.dtype,
            self.tp_size,
            self.tp_rank,
        )

    def update_state_from_configs(self):
        """
        Update the internal state of the LoRAManager based on the current `self.configs`. This method
        should be called whenever `self.configs` is modified (e.g., when new LoRA adapters are loaded).

        This includes:
        - Initializing LoRA adapters if they are not already loaded.
        - Collect all LoRA weight names based on the current loaded adapters.
        - Lazily monkey-patching the base model to use LoRA layers where applicable.
        - Preparing the GPU buffer pool for active LoRA weights.
        """

        # Target module names in huggingface lora configs.
        # e.g., {"k_proj", "q_proj", "v_proj", "o_proj"}
        hf_target_module_names: Set[str] = set()
        for config in self.configs.values():
            hf_target_module_names.update(config.target_modules)
        max_lora_dim: int = max([x.hf_config["r"] for x in self.configs.values()])

        # Loads / unloads LoRA adapters based on the latest configs.
        self.update_lora_adapters()

        # Lazily update states for new LoRA weight name (e.g., qkv_proj) as needed.
        #
        # Please note that the following update operations are "monotonic" by design, meaning that we update
        # multiple places to support the new weight names when the first adapter targeting such weight names
        # is loaded. However, we never "rollback" the support (e.g., convert LoRA layer back to base layer)
        # even if the associated adapters are unloaded later for both simplicity and practicality reasons: the
        # list of LoRA weight names is expected to be extremely finite and stable.
        self.update_lora_weight_names(hf_target_module_names)
        self.update_lora_modules(hf_target_module_names)
        self.update_memory_buffers(max_lora_dim)

    def update_lora_weight_names(self, hf_target_names: Set[str]):
        """
        Add new LoRA weight names if needed based on the current `self.configs`.
        """

        # Target lora weight names for lora_a and lora_b modules respectively.
        for module in hf_target_names:
            lora_A, lora_B = get_normalized_lora_weight_names(module)
            self.lora_weight_names[0].update(lora_A)
            self.lora_weight_names[1].update(lora_B)

    def update_lora_adapters(self):
        """
        Update the LoRA adapters in CPU memory based on the current `self.configs`.
        It loads any new adapters that are not already loaded, and unloads any adapters
        that are no longer in `self.configs` (e.g., unloaded).
        """

        # Load new adapter weights to cpu
        for name, config in self.configs.items():
            if name not in self.loras:
                logger.info(f"Loading weight of LoRA adapter {name} from {config.path}")
                lora_adapter = LoRAAdapter(
                    name,
                    config,
                    self.base_hf_config,
                    self.load_config,
                    self.lora_backend,
                )
                lora_adapter.initialize_weights()
                self.loras[name] = lora_adapter

        # Clean up unused LoRA adapters, copying the list to avoid modifying the dict during iteration.
        for name in list(self.loras):
            if name not in self.configs:
                logger.info(f"Unloading LoRA adapter {name}")
                del self.loras[name]

        # Additional checks for flashinfer backend
        # FIXME remove the restrictions after supporting multi-rank for flashinfer backend
        if self.lora_backend == "flashinfer":
            lora_dims = set(x.hf_config["r"] for x in self.configs.values())
            scalings = set(x.scaling for x in self.loras.values())
            assert (
                len(lora_dims) == 1 and len(scalings) == 1
            ), "Flashinfer backend currently only supports single LoRA rank and scaling across all adapters. "

    def update_memory_buffers(self, max_lora_dim: int):
        """
        Update the LoRA memory pool buffers based on the current LoRA configurations and update
        LoRA modules to use the new buffers. This method should be called after the LoRA configurations
        are set or updated.
        """

        self.memory_pool.init_buffers(
            self.lora_weight_names, self.base_model, max_lora_dim
        )

    def set_lora_module(self, module_name, module):
        lora_module = get_lora_layer(module, self.lora_backend)
        replace_submodule(self.base_model, module_name, lora_module)
        return lora_module

    def update_lora_modules(self, hf_target_names: Set[str]):
        # Target module names of customized layers defined in python/sglang/srt/layers
        # e.g., {"qkv_proj", "o_proj"}
        customized_target_names = get_customized_names_from_hf_names(
            hf_target_names, self.base_model
        )

        for module_name, module in self.base_model.named_modules():
            # TODO (lifuhuang): in the future, we should consider generalizing the
            # should_apply_lora function to support mapping by full module name instead
            # of just the last part (e.g., "qkv_proj") to support scenarios with multiple
            # attention stacks (e.g., multimodal models).
            # See: https://github.com/sgl-project/sglang/issues/6608
            if getattr(
                self.base_model, "should_apply_lora", None
            ) and not self.base_model.should_apply_lora(module_name):
                continue

            # The module should be converted if it is included in target_names
            if module_name.split(".")[-1] in customized_target_names:
                layer_id = get_layer_id(module_name)
                if module_name not in self.lora_modules[layer_id]:
                    self.lora_modules[layer_id][module_name] = self.set_lora_module(
                        module_name, module
                    )
