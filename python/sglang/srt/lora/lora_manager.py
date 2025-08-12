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
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.hf_transformers_utils import AutoConfig
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
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[Dict[str, LoRARef]] = None,
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
        self.init_state(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
            lora_paths=lora_paths,
        )

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
                lora_ref.lora_name: lora_ref.lora_path
                for lora_ref in self.lora_refs.values()
            },
        )

    def load_lora_adapter(self, lora_ref: LoRARef) -> LoRAUpdateResult:
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

    def unload_lora_adapter(self, lora_ref: LoRARef) -> LoRAUpdateResult:
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

        def transfer_adapter_info(
            weight_indices_out: torch.Tensor,
            lora_ranks_out: torch.Tensor,
            scalings_out: torch.Tensor,
        ):
            """
            Transfer adapter metadata (weight indices, LoRA rank, scalings) from host
            to device (CUDA) asynchronously.
            """
            weight_indices = [0] * len(forward_batch.lora_ids)
            lora_ranks = [0] * self.max_loras_per_batch
            scalings = [0] * self.max_loras_per_batch
            for i, uid in enumerate(forward_batch.lora_ids):
                weight_indices[i] = self.memory_pool.get_buffer_id(uid)
                if uid is not None:
                    lora = self.loras[uid]
                    lora_ranks[weight_indices[i]] = lora.config.r
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

    def update_lora_info(self):
        """
        Update all LoRA modules to associate them with the latest memory buffer.
        """
        for layer_id, layer_modules in enumerate(self.lora_modules):
            for module_name, module in layer_modules.items():
                weight_name = get_weight_name(
                    module_name, self.memory_pool.lora_weight_names
                )
                module.set_lora_info(
                    self.memory_pool.get_tensor(weight_name, layer_id, LoRAType.LORA_A),
                    self.memory_pool.get_tensor(weight_name, layer_id, LoRAType.LORA_B),
                )

    def init_state(
        self,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[Dict[str, LoRARef]] = None,
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
        self.init_lora_weight_names()
        self.init_lora_modules()
        self.init_memory_pool()
        self.update_lora_info()

    def init_lora_adapters(self, lora_paths: Optional[Dict[str, LoRARef]] = None):
        # Configs of all active LoRA adapters, indexed by LoRA ID.
        self.configs: Dict[str, LoRAConfig] = {}

        # LoRA adapter weights cached in CPU memory, indexed by LoRA ID.
        self.loras: Dict[str, LoRAAdapter] = {}

        # Mapping from LoRA ID to LoRARef object.
        self.lora_refs: Dict[str, LoRARef] = {}

        # Count of pinned LoRA adapters.
        self.num_pinned_loras: int = 0

        if lora_paths:
            for lora_ref in lora_paths.values():
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

        if target_modules is not None:
            self.target_modules = set(target_modules)
        else:
            self.target_modules = set()
            for config in self.configs.values():
                if not isinstance(config.target_modules, list):
                    raise ValueError(
                        f"SGLang currently only supports inferring LoRA target modules when a list of "
                        "suffixes is provided in `target_modules` field of PEFT config. Please explicitly "
                        "specify `--lora-target-modules` during server startup. You can specify `all` to "
                        "enable all support modules types. "
                    )
                self.target_modules.update(config.target_modules)

        if max_lora_rank is not None:
            self.max_lora_rank = max_lora_rank
        else:
            self.max_lora_rank = max(
                [x.r for x in self.configs.values()],
                default=0,
            )

    def init_lora_weight_names(self):
        """
        Add new LoRA weight names if needed based on the current `self.configs`.
        """

        self.lora_weight_names: Set[str] = get_normalized_lora_weight_names(
            self.target_modules
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
            lora_weight_names=self.lora_weight_names,
            base_model=self.base_model,
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
            if module_name.split(".")[-1] in self.lora_weight_names:
                layer_id = get_layer_id(module_name)
                self.lora_modules[layer_id][module_name] = self.set_lora_module(
                    module_name, module
                )
