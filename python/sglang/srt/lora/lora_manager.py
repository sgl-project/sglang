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
from typing import Dict, Iterable, List, Optional

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.lora.backend.base_backend import BaseLoRABackend, get_backend_from_name
from sglang.srt.lora.layers import BaseLayerWithLoRA, get_lora_layer
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.lora.utils import (
    LoRABatchInfo,
    LoRAType,
    get_layer_id,
    get_normalized_target_modules,
    get_target_module_name,
)
from sglang.srt.managers.io_struct import LoRAUpdateOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import replace_submodule
from sglang.srt.utils.hf_transformers_utils import AutoConfig

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
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[List[LoRARef]] = None,
        server_args: Optional[ServerArgs] = None,
    ):
        self.base_model: torch.nn.Module = base_model
        self.base_hf_config: AutoConfig = base_hf_config
        self.max_loras_per_batch: int = max_loras_per_batch
        self.load_config: LoadConfig = load_config
        self.dtype: torch.dtype = dtype
        self.device: torch.device = next(self.base_model.parameters()).device
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank

        # Store eviction policy from server args
        self.eviction_policy = server_args.lora_eviction_policy

        # LoRA backend for running sgemm kernels
        logger.info(f"Using {lora_backend} as backend of LoRA kernels.")
        backend_type = get_backend_from_name(lora_backend)
        self.lora_backend: BaseLoRABackend = backend_type(
            max_loras_per_batch=max_loras_per_batch,
            device=self.device,
            server_args=server_args,
        )

        # Initialize mutable internal state of the LoRAManager.
        self.init_state(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
            lora_paths=lora_paths,
        )

    def init_cuda_graph_batch_info(self, max_bs_in_cuda_graph: int):
        self.max_bs_in_cuda_graph = max_bs_in_cuda_graph
        with torch.device("cuda"):
            self.cuda_graph_batch_info = LoRABatchInfo(
                bs=max_bs_in_cuda_graph,
                use_cuda_graph=True,
                num_segments=None,
                seg_lens=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32),
                seg_indptr=torch.zeros(max_bs_in_cuda_graph + 1, dtype=torch.int32),
                max_len=1,
                weight_indices=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32),
                permutation=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32),
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),
            )

        self.lora_backend.init_cuda_graph_batch_info(
            cuda_graph_batch_info=self.cuda_graph_batch_info,
            max_bs_in_cuda_graph=max_bs_in_cuda_graph,
        )

    def create_lora_update_result(
        self, success: bool, error_message: str = ""
    ) -> LoRAUpdateOutput:
        return LoRAUpdateOutput(
            success=success,
            error_message=error_message,
            loaded_adapters={
                lora_ref.lora_name: lora_ref.lora_path
                for lora_ref in self.lora_refs.values()
            },
        )

    def load_lora_adapter(self, lora_ref: LoRARef) -> LoRAUpdateOutput:
        """
        Load a single LoRA adapter from the specified path.

        Args:
            lora_ref (LoRARef): The LoRARef object containing the LoRA name, path, and ID.
        """
        assert (
            lora_ref.lora_name is not None and lora_ref.lora_path is not None
        ), "LoRARef must have both lora_name and lora_path set for loading."
        assert (
            lora_ref.lora_id not in self.loras
        ), f"LoRA adapter with ID {lora_ref.lora_id} is already loaded. This should have been verified before request is sent to the backend."

        if lora_ref.pinned and self.num_pinned_loras >= self.max_loras_per_batch - 1:
            return self.create_lora_update_result(
                success=False,
                error_message=(
                    f"Already have {self.num_pinned_loras} pinned adapters, "
                    f"max allowed is {self.max_loras_per_batch - 1} (reserving 1 slot for dynamic use). "
                    f"Please unpin some adapters or increase max_loras_per_batch."
                ),
            )

        try:
            # load configs
            new_adapter = LoRAConfig(lora_ref.lora_path)
            self.validate_new_adapter(new_adapter, lora_ref)
            self.configs[lora_ref.lora_id] = new_adapter

            # load weights
            self.load_lora_weights(lora_ref)

            # keep metadata for displayed messages
            self.lora_refs[lora_ref.lora_id] = lora_ref
            self.num_pinned_loras += int(lora_ref.pinned)
        except Exception as e:
            return self.create_lora_update_result(
                success=False,
                error_message=str(e),
            )

        return self.create_lora_update_result(success=True)

    def validate_new_adapter(self, lora_config: LoRAConfig, lora_ref: LoRARef):
        """
        Validate if an adapter can be loaded into the current LoRA memory pool and generate error if it is incompatible.
        """

        # Check if this LoRA adapter is already loaded
        if any(
            lora_ref.lora_name == existing_lora_ref.lora_name
            for existing_lora_ref in self.lora_refs.values()
        ):
            raise ValueError(
                f"Failed to load LoRA adapter {lora_ref.lora_name} because it is already loaded"
            )

        # Check if the LoRA adapter shape is compatible with the current LoRA memory pool configuration.
        memory_pool = getattr(self, "memory_pool", None)
        incompatible = memory_pool and not memory_pool.can_support(lora_config)
        if incompatible:
            raise ValueError(
                f"LoRA adapter {lora_ref.lora_name} with rank {lora_config.r} is incompatible with the current "
                "LoRA memory pool configuration. Please ensure that the LoRA adapter's rank is within the configured "
                "`--max-lora-rank` and that the target modules are included in `--lora-target-modules`."
            )

        # Ensure pinned LoRA adapters does not exceed maximal limit or cause starvation.
        if lora_ref.pinned and self.num_pinned_loras >= self.max_loras_per_batch - 1:
            raise ValueError(
                f"Failed to load LoRA adapter {lora_ref.lora_name} as a pinned adapter. It is not allowed to pin all slots "
                "in the LoRA memory pool to avoid starvation for unpinned adapters and base models. Please increase your "
                "`--max-loras-per-batch` or load it as unpinned LoRA adapters."
            )

    def unload_lora_adapter(self, lora_ref: LoRARef) -> LoRAUpdateOutput:
        """
        Unload LoRA adapters by their names. This will remove the adapters from the memory pool and
        delete the corresponding LoRA modules.
        """

        adapter = self.configs.get(lora_ref.lora_id)
        lora_ref = self.lora_refs.get(lora_ref.lora_id)
        assert (
            adapter is not None and lora_ref is not None
        ), f"LoRA adapter with ID {lora_ref.lora_id} is not loaded. This should have been verified before request is sent to the backend."

        try:
            del self.configs[lora_ref.lora_id]
            del self.loras[lora_ref.lora_id]
            del self.lora_refs[lora_ref.lora_id]
            self.num_pinned_loras -= int(lora_ref.pinned)
        except Exception as e:
            return self.create_lora_update_result(
                success=False,
                error_message=str(e),
            )

        return self.create_lora_update_result(success=True)

    def validate_lora_batch(self, lora_ids: set[str]) -> bool:
        """
        Validate if the LoRA IDs in the batch can be loaded into the current LoRA memory pool.
        """
        if len(lora_ids) > self.max_loras_per_batch:
            return False

        # skip pinned LoRA check if no pinned LoRA adapters are loaded.
        if self.num_pinned_loras == 0:
            return True

        # counting the number of pinned LoRA adapters in the batch.
        pinned_loras_in_batch = 0
        for lora_id in lora_ids:
            if lora_id is not None:
                lora_ref = self.lora_refs.get(lora_id)
                assert (
                    lora_ref is not None
                ), f"LoRA ID {lora_id} not found in lora_refs."
                pinned_loras_in_batch += int(lora_ref.pinned)

        assert pinned_loras_in_batch <= self.num_pinned_loras, (
            f"Number of pinned LoRA adapters in the batch ({pinned_loras_in_batch}) exceeds the total number of pinned adapters "
            f"({self.num_pinned_loras}). This indicates a bug in the LoRA loading logic."
        )

        required_slots = len(lora_ids) - pinned_loras_in_batch
        mem_pool_vacancy = self.memory_pool.max_loras_per_batch - self.num_pinned_loras

        return required_slots <= mem_pool_vacancy

    def prepare_lora_batch(self, forward_batch: ForwardBatch):
        # Load active loras into lora memory pool
        cur_uids = set(forward_batch.lora_ids)

        assert len(cur_uids) <= self.max_loras_per_batch
        self.memory_pool.prepare_lora_batch(
            cur_uids=cur_uids,
            lora_adapters=self.loras,
            lora_modules=self.lora_modules,
            lora_refs=self.lora_refs.copy(),  # copy snapshot of current lora_refs to avoid mutation during the batch preparation.
        )

        # set up batch info shared by all lora modules
        bs = forward_batch.batch_size

        use_cuda_graph = (
            hasattr(self, "max_bs_in_cuda_graph")
            and bs <= self.max_bs_in_cuda_graph
            and forward_batch.forward_mode.is_cuda_graph()
        )

        weight_indices = [0] * len(forward_batch.lora_ids)
        lora_ranks = [0] * self.max_loras_per_batch
        scalings = [0] * self.max_loras_per_batch
        for i, uid in enumerate(forward_batch.lora_ids):
            weight_indices[i] = self.memory_pool.get_buffer_id(uid)
            if uid is not None:
                lora = self.loras[uid]
                lora_ranks[weight_indices[i]] = lora.config.r
                scalings[weight_indices[i]] = lora.scaling
        # Do in-place updates when CUDA graph is enabled and the batch forward mode
        # could use CUDA graph.
        self.lora_backend.prepare_lora_batch(
            forward_batch=forward_batch,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            batch_info=self.cuda_graph_batch_info if use_cuda_graph else None,
        )

    def update_lora_info(self):
        """
        Update all LoRA modules to associate them with the latest memory buffer.
        """
        for layer_id, layer_modules in enumerate(self.lora_modules):
            for module_name, module in layer_modules.items():
                target_module = get_target_module_name(
                    module_name, self.memory_pool.target_modules
                )
                module.set_lora_info(
                    self.memory_pool.get_tensor(
                        target_module=target_module,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_A,
                    ),
                    self.memory_pool.get_tensor(
                        target_module=target_module,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_B,
                    ),
                )

    def init_state(
        self,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[List[LoRARef]] = None,
    ):
        """
        Initialize the internal (mutable) state of the LoRAManager.

        When `lora_paths` is provided and not empty, it might be used for inferring LoRA shape info such as
        the target modules and max_lora_rank.
        """

        assert lora_paths or (
            max_lora_rank is not None and target_modules is not None
        ), "When no initial --lora-paths is provided, you need to specify both --max-lora-rank and --lora-target-modules for LoRA initialization."

        self.init_lora_adapters(lora_paths)
        self.init_lora_shapes(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
        )
        self.init_lora_modules()
        self.init_memory_pool()
        self.update_lora_info()

    def init_lora_adapters(self, lora_paths: Optional[List[LoRARef]] = None):
        # Configs of all active LoRA adapters, indexed by LoRA ID.
        self.configs: Dict[str, LoRAConfig] = {}

        # LoRA adapter weights cached in CPU memory, indexed by LoRA ID.
        self.loras: Dict[str, LoRAAdapter] = {}

        # Mapping from LoRA ID to LoRARef object.
        self.lora_refs: Dict[str, LoRARef] = {}

        # Count of pinned LoRA adapters.
        self.num_pinned_loras: int = 0

        if lora_paths:
            for lora_ref in lora_paths:
                result = self.load_lora_adapter(lora_ref)
                if not result.success:
                    raise RuntimeError(
                        f"Failed to load LoRA adapter {lora_ref.lora_name}: {result.error_message}"
                    )

    def init_lora_shapes(
        self,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
    ):
        """Infer LoRA target modules and max_lora_rank from loaded adapters if not provided."""

        self.target_modules = (
            get_normalized_target_modules(target_modules) if target_modules else set()
        )

        for lora_id, config in self.configs.items():
            if not isinstance(config.target_modules, list):
                raise ValueError(
                    f"SGLang currently only supports inferring LoRA target modules when a list of "
                    "suffixes is provided in `target_modules` field of PEFT config. Please explicitly "
                    "specify `--lora-target-modules` during server startup. You can specify `all` to "
                    "enable all support modules types. "
                )

            adapter_target_modules = get_normalized_target_modules(
                config.target_modules
            )

            if target_modules is not None:
                # When `--lora-target-modules` is provided, validate adapter target modules is a subset of the specified target modules.
                if not adapter_target_modules.issubset(self.target_modules):
                    unsupported_modules = adapter_target_modules - self.target_modules
                    lora_name = self.lora_refs[lora_id].lora_name
                    raise ValueError(
                        f"LoRA adapter '{lora_name}' contains target modules {sorted(unsupported_modules)} "
                        f"that are not included in the specified --lora-target-modules {sorted(self.target_modules)}. "
                        f"Please update --lora-target-modules to include all required modules: "
                        f"{sorted(self.target_modules | adapter_target_modules)}, or use 'all' to enable all supported modules."
                    )
            else:
                # Otherwise, infer target_modules from adapter configs.
                self.target_modules.update(adapter_target_modules)

        if max_lora_rank is not None:
            self.max_lora_rank = max_lora_rank
        else:
            self.max_lora_rank = max(
                [x.r for x in self.configs.values()],
                default=0,
            )

    def load_lora_weights(self, lora_ref: LoRARef):
        """
        Load the weights of a LoRA adapter to CPU memory and conducts post-loading validation.
        """
        lora_adapter = LoRAAdapter(
            lora_ref.lora_id,
            self.configs[lora_ref.lora_id],
            self.base_hf_config,
            self.load_config,
            self.lora_backend,
        )
        lora_adapter.initialize_weights()
        self.loras[lora_ref.lora_id] = lora_adapter

    def init_memory_pool(self):
        """(Re)initialize the LoRA memory pool based on the current configurations."""
        self.memory_pool = LoRAMemoryPool(
            base_hf_config=self.base_hf_config,
            max_loras_per_batch=self.max_loras_per_batch,
            dtype=self.dtype,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.max_lora_rank,
            target_modules=self.target_modules,
            base_model=self.base_model,
            eviction_policy=self.eviction_policy,
        )

    def set_lora_module(self, module_name, module):
        lora_module = get_lora_layer(module, self.lora_backend)
        replace_submodule(self.base_model, module_name, lora_module)
        return lora_module

    def init_lora_modules(self):
        # Look-up table that essentially maps (layer_index, module_name) to the corresponding LoRA module.
        self.lora_modules: List[Dict[str, BaseLayerWithLoRA]] = [
            {} for _ in range(self.base_hf_config.num_hidden_layers)
        ]

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
            if module_name.split(".")[-1] in self.target_modules:
                layer_id = get_layer_id(module_name)
                self.lora_modules[layer_id][module_name] = self.set_lora_module(
                    module_name, module
                )
