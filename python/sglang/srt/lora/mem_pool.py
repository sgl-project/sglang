from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.utils import (
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
    ):

        self.base_hf_config: AutoConfig = base_hf_config
        self.num_layer: int = base_hf_config.num_hidden_layers
        self.max_loras_per_batch: int = max_loras_per_batch
        self.max_lora_dim: int = max_lora_dim
        self.dtype: torch.dtype = dtype

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
        # All uids are initalized as empty strings for empty buffer slots
        # Here we don't initalize to None since None is a valid uid
        self.buffer_id_to_uid: List[Optional[str]] = [""] * self.max_loras_per_batch

    def init_buffers(
        self,
        lora_weight_names: Set[Tuple[str]],
        base_model: torch.nn.Module,
    ):

        # lora_weight_names is a set of name pairs indicating each pair of lora modules to load
        #   e.g., {("qkv_proj", "q_proj"), ("qkv_proj", "kv_proj"), ("o_proj", "o_proj")}
        self.lora_weight_names: Set[Tuple[str]] = lora_weight_names

        for module_A, module_B in lora_weight_names:
            # Init A tensor, column_major=False
            input_dim, _ = get_hidden_dim(module_A, self.base_hf_config, base_model)
            c = get_stacked_multiply(module_A)
            if module_A not in self.A_buffer:
                self.A_buffer[module_A] = [
                    torch.empty(
                        (
                            self.max_loras_per_batch,
                            self.max_lora_dim * c,
                            input_dim,
                        ),
                        dtype=self.dtype,
                        device="cuda",
                    )
                    for i in range(self.num_layer)
                ]

            # Init B tensor, column_major=True
            _, output_dim = get_hidden_dim(module_B, self.base_hf_config, base_model)
            c = get_stacked_multiply(module_B)
            if module_B not in self.B_buffer:
                self.B_buffer[module_B] = [
                    torch.empty(
                        (
                            c,  # stacked lora_b modules might need separation
                            self.max_loras_per_batch,
                            output_dim,
                            self.max_lora_dim,
                        ),
                        dtype=self.dtype,
                        device="cuda",
                    )
                    for i in range(self.num_layer)
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
                    self.A_buffer[k][i][buffer_id] *= 0
            return

        assert lora_adapter is not None
        for layer_id in range(self.num_layer):
            layer_weights = lora_adapter.layers[layer_id].weights
            for name, weights in layer_weights.items():
                if "lora_A" in name:
                    lora_weight_name = get_weight_name(
                        name, self.lora_weight_names, LoRAType.LORA_A
                    )
                    if lora_weight_name:
                        self.A_buffer[lora_weight_name][layer_id][buffer_id].copy_(
                            weights
                        )
                else:
                    lora_weight_name = get_weight_name(
                        name, self.lora_weight_names, LoRAType.LORA_B
                    )
                    if lora_weight_name:
                        c = get_stacked_multiply(lora_weight_name)
                        if c > 1:
                            for stacked_id in range(c):
                                self.B_buffer[lora_weight_name][layer_id][stacked_id][
                                    buffer_id
                                ].copy_(weights[stacked_id])
                        else:
                            self.B_buffer[lora_weight_name][layer_id][0][
                                buffer_id
                            ].copy_(weights)

    def get_tensor(
        self, weight_name: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:

        if lora_type == LoRAType.LORA_A:
            return self.A_buffer[weight_name][layer_id]

        return self.B_buffer[weight_name][layer_id]

    def get_buffer_id(self, lora_uid: str):
        return self.uid_to_buffer_id[lora_uid]
