import re
from enum import Enum
from typing import Optional, Set, Tuple

import torch

from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.lora.backend import (
    BaseLoRABackend,
    FlashInferLoRABackend,
    TritonLoRABackend,
)


class LoRAType(Enum):
    LORA_A = 0
    LORA_B = 1


def get_backend_from_name(name: str) -> BaseLoRABackend:
    """
    Get corresponding backend class from backend's name
    """
    backend_mapping = {
        "triton": TritonLoRABackend,
        "flashinfer": FlashInferLoRABackend,
    }

    if name in backend_mapping:
        return backend_mapping[name]

    raise Exception(
        f"No supported lora backend called {name}. It should be one of {list(backend_mapping.keys())}"
    )


def get_layer_id(name: str) -> int:
    """
    Extract integer id of layer from its name in string.
    """
    match = re.search(r"layers\.(\d+)\.", name)
    if match is None:
        return None
    return int(match.group(1))


def get_customized_names_from_hf_names(
    hf_module_names: Set[str], base_model: torch.nn.Module
) -> Set[str]:
    """
    This function takes in a set of huggingface style module names:
         e.g., {"k_proj", "q_proj", "v_proj", "o_proj"}
    and outputs a set of module names of customized sglang layers:
         e.g., {"qkv_proj", "o_proj"}
    """
    if hasattr(base_model, "get_module_name"):
        return {base_model.get_module_name(name) for name in hf_module_names}
    else:
        """
        Fallback solution of mapping from config module name to module name in model class.
        Please check if it aligns with your base model.
        Please implement the function in the model class if it is not.
        You can reference this function in llama.py.
        """
        params_mapping = {
            "q_proj": "qkv_proj",
            "k_proj": "qkv_proj",
            "v_proj": "qkv_proj",
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
        }
        return {params_mapping.get(name, name) for name in hf_module_names}


def get_hidden_dim(
    module_name: str, config: AutoConfig, base_model: torch.nn.Module
) -> Tuple[int]:
    """
    Given a module_name (might be a stacked name), return the hidden dims of modules's input and output.
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
        if module_name in ["q_proj", "o_proj", "qkv_proj"]:
            return config.hidden_size, config.hidden_size
        elif module_name in ["kv_proj"]:
            return config.hidden_size, config.hidden_size // (
                config.num_attention_heads // config.num_key_value_heads
            )
        elif module_name == "gate_up_proj":
            return config.hidden_size, config.intermediate_size
        elif module_name == "down_proj":
            return config.intermediate_size, config.hidden_size
        else:
            raise NotImplementedError()


def get_stacked_name(name: str) -> Tuple[str]:
    """
    Mapping a target module name to (stacked name for Lora A, stacked name for Lora B)
    """
    params_mapping = {
        "q_proj": ("qkv_proj", "q_proj"),
        "k_proj": ("qkv_proj", "kv_proj"),
        "v_proj": ("qkv_proj", "kv_proj"),
        "gate_proj": ("gate_up_proj", "gate_up_proj"),
        "up_proj": ("gate_up_proj", "gate_up_proj"),
    }
    return params_mapping.get(name, (name, name))


def get_stacked_multiply(module_name: str) -> int:
    """
    Mapping a lora module name to its magnification at output dimension
    """
    stacked_rank = {
        "qkv_proj": 3,
        "kv_proj": 2,
        "gate_up_proj": 2,
    }
    return stacked_rank[module_name] if module_name in stacked_rank else 1


def get_weight_name(
    target_name: str, lora_weight_names: Set[Tuple[str]], lora_type: LoRAType
) -> Optional[str]:
    """
    target_name is name of a given module,
    lora_weight_names is a set of lora stacked name pairs (see get_stacked_name method above)
    If there is a weight name in lora_weight_names that can match target_name, return this name
    Else return None
    """
    idx = 0 if lora_type == LoRAType.LORA_A else 1
    for weight_name_pair in lora_weight_names:
        if weight_name_pair[idx] in target_name:
            return weight_name_pair[idx]
