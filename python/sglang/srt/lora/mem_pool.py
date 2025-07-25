from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch

from sglang.srt.distributed import divide
from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.utils import (
    ROW_PARALLELISM_LINEAR_LORA_NAMES,
    LoRAType,
    get_hidden_dim,
    get_normalized_lora_weight_names,
    get_stacked_multiply,
    get_weight_name,
)


class LoRAMemoryPool:
    """Class for memory pool management of lora modules"""

    def __init__(
        self,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        max_lora_rank: int,
        lora_weight_names: Tuple[Set[str], Set[str]],
        base_model: torch.nn.Module,
    ):
        self.base_hf_config: AutoConfig = base_hf_config
        self.num_layer: int = base_hf_config.num_hidden_layers
        self.max_loras_per_batch: int = max_loras_per_batch
        self.dtype: torch.dtype = dtype
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.max_lora_rank: int = max_lora_rank

        # lora weight names for LoRA A and B respectively.
        self.lora_weight_names: Tuple[Set[str], Set[str]] = lora_weight_names

        # Both A_buffer and B_buffer maps lora weight names to its buffer space.
        # A_buffer contains num_layer number of row-major tensors with shape
        #   (max_loras_per_batch, stacked_num * max_lora_dim, input_dim)
        # B_buffer contains num_layer number of column-major tensors with shape
        #   (stacked_num, max_loras_per_batch, output_dim, max_lora_dim)
        self.A_buffer: Dict[str, List[torch.Tensor]] = {}
        self.B_buffer: Dict[str, List[torch.Tensor]] = {}

        # Lora uid -> buffer idx in memory pool
        self.uid_to_buffer_id: Dict[Optional[str], int] = {}

        # Buffer idx -> lora uid in memory pool
        # All uids are initialized as empty strings for empty buffer slots
        # Here we don't initialize to None since None is a valid uid
        self.buffer_id_to_uid: List[Optional[str]] = [""] * self.max_loras_per_batch

        self.init_buffers(base_model)

    def can_support(self, config: Union[LoRAConfig, Iterable[LoRAConfig]]) -> bool:
        """
        Check if the memory pool can support the given LoRA adapters.
        """

        def _can_support(config: LoRAConfig) -> bool:
            """
            Check if the memory pool can support a single LoRA adapter.
            """
            if config.r > self.max_lora_rank:
                return False
            weights_a, weights_b = get_normalized_lora_weight_names(
                config.target_modules
            )
            return weights_a.issubset(self.lora_weight_names[0]) and weights_b.issubset(
                self.lora_weight_names[1]
            )

        if isinstance(config, LoRAConfig):
            return _can_support(config)
        else:
            return all(_can_support(x) for x in config)

    def get_lora_A_shape(
        self, module_name: str, base_model: torch.nn.Module, max_lora_dim: int
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        input_dim, _ = get_hidden_dim(module_name, self.base_hf_config, base_model)
        c = get_stacked_multiply(module_name)
        if self.tp_size > 1 and module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            input_dim = divide(input_dim, self.tp_size)
        return (
            self.max_loras_per_batch,
            max_lora_dim * c,
            input_dim,
        )

    def get_lora_B_shape(
        self, module_name: str, base_model: torch.nn.Module, max_lora_dim: int
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        _, output_dim = get_hidden_dim(module_name, self.base_hf_config, base_model)
        c = get_stacked_multiply(module_name)
        if self.tp_size > 1 and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            output_dim = divide(output_dim, self.tp_size)
        return (
            c,
            self.max_loras_per_batch,
            output_dim,
            max_lora_dim,
        )

    def init_buffers(self, base_model: torch.nn.Module):
        device = next(base_model.parameters()).device

        def init_buffer(
            buffer: Dict[str, List[torch.Tensor]],
            lora_weight_names: Set[str],
            get_lora_shape_fn: Callable[[str, torch.nn.Module, int], Tuple[int]],
        ):
            for module_name in lora_weight_names:
                lora_shape = get_lora_shape_fn(
                    module_name, base_model, self.max_lora_rank
                )
                buffer[module_name] = [
                    torch.empty(
                        lora_shape,
                        dtype=self.dtype,
                        device=device,
                    )
                    for _ in range(self.num_layer)
                ]

        init_buffer(
            self.A_buffer,
            self.lora_weight_names[0],
            self.get_lora_A_shape,
        )

        init_buffer(
            self.B_buffer,
            self.lora_weight_names[1],
            self.get_lora_B_shape,
        )

    def prepare_lora_batch(
        self,
        cur_uids: Set[Optional[str]],
        lora_adapters: Dict[str, LoRAAdapter],
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
    ):
        def get_available_buffer_slot():
            for buffer_id in range(self.max_loras_per_batch):
                # Prioritize empty slots
                if self.buffer_id_to_uid[buffer_id] == "":
                    return buffer_id

            for buffer_id in range(self.max_loras_per_batch):
                # Evict unneeded lora
                if self.buffer_id_to_uid[buffer_id] not in cur_uids:
                    self.uid_to_buffer_id.pop(self.buffer_id_to_uid[buffer_id])
                    return buffer_id

            raise ValueError(
                "No available buffer slots found. Please ensure the number of active loras is less than max_loras_per_batch."
            )

        for uid in cur_uids:
            if uid not in self.uid_to_buffer_id:
                buffer_id = get_available_buffer_slot()
                lora_adapter = lora_adapters.get(uid, None)
                self.load_lora_weight_to_buffer(
                    uid, buffer_id, lora_adapter, lora_modules
                )
                self.uid_to_buffer_id[uid] = buffer_id
                self.buffer_id_to_uid[buffer_id] = uid

    def load_lora_weight_to_buffer(
        self,
        uid: str,
        buffer_id: int,
        lora_adapter: LoRAAdapter,
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
    ):
        def load_lora_weight_tensor(
            buffer_view: torch.Tensor, weight: Optional[torch.Tensor]
        ):
            if weight is None:
                # If the particular weight is not present in the adapter, we initialize the buffer to zero
                # to avoid contamination from the residual weight of the evicted adapters.
                buffer_view.zero_()
            else:
                assert (
                    buffer_view.shape == weight.shape
                ), f"LoRA buffer shape {buffer_view.shape} does not match weight shape {weight.shape}."
                buffer_view.copy_(weight)

        if uid is None:
            for i in range(self.num_layer):
                for k in self.A_buffer.keys():
                    self.A_buffer[k][i][buffer_id] = 0
            return

        assert lora_adapter is not None
        lora_rank = lora_adapter.config.hf_config["r"]
        for layer_id in range(self.num_layer):
            layer_weights = lora_adapter.layers[layer_id].weights
            temp_A_buffer: Dict[str, Optional[torch.Tensor]] = {
                weight_name: None for weight_name in self.A_buffer
            }
            temp_B_buffer: Dict[str, Optional[torch.Tensor]] = {
                weight_name: None for weight_name in self.B_buffer
            }
            for name, weights in layer_weights.items():
                if "lora_A" in name:
                    lora_weight_name = get_weight_name(
                        name, self.lora_weight_names, LoRAType.LORA_A
                    )
                    temp_A_buffer[lora_weight_name] = weights
                else:
                    lora_weight_name = get_weight_name(
                        name, self.lora_weight_names, LoRAType.LORA_B
                    )
                    temp_B_buffer[lora_weight_name] = weights

            if self.tp_size > 1:
                cur_layer_modules = lora_modules[layer_id]
                for module_name, module in cur_layer_modules.items():
                    weight_name = get_weight_name(
                        module_name, self.lora_weight_names, LoRAType.LORA_A
                    )

                    if temp_A_buffer[weight_name] is None:
                        # Skip weight slicing if the weight is not present in the adapter
                        continue

                    if "qkv_proj" in module_name:
                        temp_A_buffer["qkv_proj"] = module.slice_lora_a_weights(
                            temp_A_buffer["qkv_proj"], self.tp_rank
                        )
                        temp_B_buffer["q_proj"], temp_B_buffer["kv_proj"] = (
                            module.slice_lora_b_weights(
                                [temp_B_buffer["q_proj"], temp_B_buffer["kv_proj"]],
                                self.tp_rank,
                            )
                        )
                    else:
                        # TODO (lifuhuang): Ideally, we should call `get_weight_name` separately for both A and B.
                        # Currently, we're reusing A's weight name as a workaround, relying on the fact that A and
                        # B share the same name except for `qkv_proj`. We should clean this up once we deprecate the
                        # FlashInfer LoRA backend.
                        temp_A_buffer[weight_name] = module.slice_lora_a_weights(
                            temp_A_buffer[weight_name], self.tp_rank
                        )
                        temp_B_buffer[weight_name] = module.slice_lora_b_weights(
                            temp_B_buffer[weight_name], self.tp_rank
                        )

            for name, weights in temp_A_buffer.items():
                c = get_stacked_multiply(name)
                buffer_view = self.A_buffer[name][layer_id][buffer_id][
                    : lora_rank * c, :
                ]
                load_lora_weight_tensor(buffer_view, weights)

            for name, weights in temp_B_buffer.items():
                c = get_stacked_multiply(name)
                if c > 1:
                    for stacked_id in range(c):
                        buffer_view = self.B_buffer[name][layer_id][stacked_id][
                            buffer_id
                        ][:, :lora_rank]
                        weight_slice = (
                            weights[stacked_id] if weights is not None else None
                        )
                        load_lora_weight_tensor(buffer_view, weight_slice)
                else:
                    buffer_view = self.B_buffer[name][layer_id][0][buffer_id][
                        :, :lora_rank
                    ]
                    load_lora_weight_tensor(buffer_view, weights)

    def get_tensor(
        self, weight_name: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:
        if lora_type == LoRAType.LORA_A:
            return self.A_buffer[weight_name][layer_id]

        return self.B_buffer[weight_name][layer_id]

    def get_buffer_id(self, lora_uid: str):
        return self.uid_to_buffer_id[lora_uid]
