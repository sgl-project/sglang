import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Set, Tuple

import torch

from sglang.srt.hf_transformers_utils import AutoConfig


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

    # ranks of each lora adapter, in shape (lora_num,)
    lora_ranks: torch.Tensor

    # scaling of each lora adapter, in shape (lora_num,)
    scalings: torch.Tensor


class LoRAType(Enum):
    LORA_A = 0
    LORA_B = 1


def get_layer_id(name: str) -> int:
    """
    Extract integer id of layer from its name in string.
    """
    match = re.search(r"layers\.(\d+)\.", name)
    if match is None:
        return None
    return int(match.group(1))


def get_hidden_dim(
    module_name: str, config: AutoConfig, base_model: torch.nn.Module
) -> Tuple[int]:
    """
    Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
    """

    if hasattr(base_model, "get_hidden_dim"):
        return base_model.get_hidden_dim(module_name)
    else:
        """
        WARNING: get_hidden_dim() is not defined,
        which is used to get the hidden dim for different lora modules
        Use the default one, but please check if it is correct for your model.
        Please implement the function in the model class if it is not.
        You can reference this function in llama.py.
        """
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        if module_name == "qkv_proj":
            return config.hidden_size, head_dim * (
                config.num_attention_heads + config.num_key_value_heads * 2
            )
        elif module_name == "o_proj":
            return (
                head_dim * config.num_attention_heads,
                config.hidden_size,
            )
        elif module_name == "gate_up_proj":
            return config.hidden_size, config.intermediate_size * 2
        elif module_name == "down_proj":
            return config.intermediate_size, config.hidden_size
        else:
            raise NotImplementedError()


def get_normalized_target_modules(
    target_modules: Iterable[str],
) -> set[str]:
    """
    Mapping a list of target module name to names of the normalized LoRA weights.
    """
    params_mapping = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
    }

    result = set()
    for name in target_modules:
        normalized_name = params_mapping.get(name, name)
        result.add(normalized_name)
    return result


def get_stacked_multiply(module_name: str) -> int:
    """
    Mapping a lora module name to its magnification at output dimension
    """
    stacked_rank = {
        "qkv_proj": 3,
        "gate_up_proj": 2,
    }
    return stacked_rank[module_name] if module_name in stacked_rank else 1


def get_target_module_name(full_module_name: str, target_modules: Set[str]) -> str:
    """
    Get the target module name in target_modules that can match full_module_name.

    If there is a target module name in target_modules that can match full_module_name, return this name
    Else raise ValueError.
    """
    for target_module in target_modules:
        if target_module in full_module_name:
            return target_module
    raise ValueError(
        f"Cannot find target module name for {full_module_name} in {target_modules}"
    )


ROW_PARALLELISM_LINEAR_LORA_NAMES = ["o_proj", "down_proj"]
