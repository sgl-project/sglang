from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.distributed import divide
from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.utils import (
    ROW_PARALLELISM_LINEAR_LORA_NAMES,
    LoRAType,
    get_hidden_dim,
    get_stacked_multiply,
    get_weight_name,
)


class LoRAMemoryPool:
    """Class for memory pool management of lora modules"""

    def __init__(
        self,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        max_lora_dim: int,
        dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        lora_modules: Dict[int, List[Tuple[str, BaseLayerWithLoRA]]],
    ):

        self.base_hf_config: AutoConfig = base_hf_config
        self.num_layer: int = base_hf_config.num_hidden_layers
        self.max_loras_per_batch: int = max_loras_per_batch
        self.max_lora_dim: int = max_lora_dim
        self.dtype: torch.dtype = dtype
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.lora_modules: Dict[int, List[Tuple[str, BaseLayerWithLoRA]]] = lora_modules

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

    def get_lora_A_shape(
        self, module_name: str, base_model: torch.nn.Module
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        input_dim, _ = get_hidden_dim(module_name, self.base_hf_config, base_model)
        c = get_stacked_multiply(module_name)
        if self.tp_size > 1:
            if module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES:
                input_dim = divide(input_dim, self.tp_size)
        return (
            self.max_loras_per_batch,
            self.max_lora_dim * c,
            input_dim,
        )

    def get_lora_B_shape(
        self, module_name: str, base_model: torch.nn.Module
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        _, output_dim = get_hidden_dim(module_name, self.base_hf_config, base_model)
        c = get_stacked_multiply(module_name)
        if self.tp_size > 1:
            if module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES:
                output_dim = divide(output_dim, self.tp_size)
        return (
            c,
            self.max_loras_per_batch,
            output_dim,
            self.max_lora_dim,
        )

    def init_buffers(
        self,
        lora_weight_names: Set[Tuple[str]],
        base_model: torch.nn.Module,
    ):

        # lora_weight_names is a set of name pairs indicating each pair of lora modules to load
        #   e.g., {("qkv_proj", "q_proj"), ("qkv_proj", "kv_proj"), ("o_proj", "o_proj")}
        self.lora_weight_names: Set[Tuple[str]] = lora_weight_names
        device = next(base_model.parameters()).device
        lora_module_A_names = set([name[0] for name in lora_weight_names])
        lora_module_B_names = set([name[1] for name in lora_weight_names])
        # Init A tensor, column_major=False
        for module_A in lora_module_A_names:
            lora_A_shape = self.get_lora_A_shape(module_A, base_model)
            self.A_buffer[module_A] = [
                torch.empty(
                    lora_A_shape,
                    dtype=self.dtype,
                    device=device,
                )
                for i in range(self.num_layer)
            ]
        # Init B tensor, column_major=True
        for module_B in lora_module_B_names:
            lora_B_shape = self.get_lora_B_shape(module_B, base_model)
            self.B_buffer[module_B] = [
                torch.empty(
                    lora_B_shape,
                    dtype=self.dtype,
                    device=device,
                )
                for _ in range(self.num_layer)
            ]

    def prepare_lora_batch(
        self,
        cur_uids: Set[Optional[str]],
        lora_adapters: Dict[str, LoRAAdapter],
    ):

        def get_available_buffer_slot():
            for buffer_id in range(self.max_loras_per_batch):
                # Prioritize empty slots
                if self.buffer_id_to_uid[buffer_id] == "":
                    return buffer_id, ""

            for buffer_id in range(self.max_loras_per_batch):
                # Evict unneeded lora
                if self.buffer_id_to_uid[buffer_id] not in cur_uids:
                    return buffer_id, self.buffer_id_to_uid[buffer_id]

            raise ValueError(
                "No available buffer slots found. Please ensure the number of active loras is less than max_loras_per_batch."
            )

        for uid in cur_uids:
            if uid not in self.uid_to_buffer_id:
                buffer_id, evicted_lora_uid = get_available_buffer_slot()
                if evicted_lora_uid != "":
                    self.uid_to_buffer_id.pop(evicted_lora_uid)
                self.load_lora_weight_to_buffer(
                    uid, buffer_id, lora_adapters.get(uid, None)
                )
                self.uid_to_buffer_id[uid] = buffer_id
                self.buffer_id_to_uid[buffer_id] = uid

    def load_lora_weight_to_buffer(
        self, uid: str, buffer_id: int, lora_adapter: LoRAAdapter = None
    ):

        if uid is None:
            for i in range(self.num_layer):
                for k in self.A_buffer.keys():
                    self.A_buffer[k][i][buffer_id] = 0
            return

        assert lora_adapter is not None
        lora_rank = lora_adapter.config.hf_config["r"]
        for layer_id in range(self.num_layer):
            layer_weights = lora_adapter.layers[layer_id].weights
            temp_A_buffer: Dict[str, torch.Tensor] = {}
            temp_B_buffer: Dict[str, torch.Tensor] = {}
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
                cur_layer_modules = self.lora_modules[layer_id]
                for module_name, module in cur_layer_modules:
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
                        weight_name = get_weight_name(
                            module_name, self.lora_weight_names, LoRAType.LORA_A
                        )
                        temp_A_buffer[weight_name] = module.slice_lora_a_weights(
                            temp_A_buffer[weight_name], self.tp_rank
                        )
                        temp_B_buffer[weight_name] = module.slice_lora_b_weights(
                            temp_B_buffer[weight_name], self.tp_rank
                        )

            for name, weights in temp_A_buffer.items():
                c = get_stacked_multiply(name)
                self.A_buffer[name][layer_id][buffer_id][: lora_rank * c, :].copy_(
                    weights
                )

            for name, weights in temp_B_buffer.items():
                c = get_stacked_multiply(name)
                if c > 1:
                    for stacked_id in range(c):
                        self.B_buffer[name][layer_id][stacked_id][buffer_id][
                            :, :lora_rank
                        ].copy_(weights[stacked_id])
                else:
                    self.B_buffer[name][layer_id][0][buffer_id][:, :lora_rank].copy_(
                        weights
                    )

    def get_tensor(
        self, weight_name: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:

        if lora_type == LoRAType.LORA_A:
            return self.A_buffer[weight_name][layer_id]

        return self.B_buffer[weight_name][layer_id]

    def get_buffer_id(self, lora_uid: str):
        return self.uid_to_buffer_id[lora_uid]
