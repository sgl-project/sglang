# Copyright 2023-2024 SGLang Team

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn

from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)


def common_load_weights(
    model: nn.Module,
    weights: Iterable[Tuple[str, torch.Tensor]],
    stacked_params_mapping: List[Tuple[str, str, Union[str, int]]],
    expert_params_mapping: Optional[List[Tuple[str, str, int, Union[str, int]]]] = None,
    skip_prefixes: Optional[List[str]] = None,
    tie_word_embeddings: bool = False,
) -> bool:
    """
    Common implementation for loading weights across different model types.

    Args:
        model: Model to load weights into
        weights: Iterable of (name, tensor) tuples
        stacked_params_mapping: List of (param_name, weight_name, shard_id) tuples
        expert_params_mapping: Optional list of (param_name, weight_name, expert_id, shard_id) for MoE models
        skip_prefixes: List of prefixes to skip when loading weights
        tie_word_embeddings: Whether to skip lm_head.weight if tie_word_embeddings is True

    Returns:
        True if successful
    """
    if skip_prefixes is None:
        skip_prefixes = [
            "rotary_emb.inv_freq",
            "rotary_emb.cos_cached",
            "rotary_emb.sin_cached",
            "projector",
        ]

    params_dict = dict(model.named_parameters())

    for name, loaded_weight in weights:
        # Skip layers that aren't in this rank's range in pipeline parallelism
        layer_id = get_layer_id(name)
        if (
            layer_id is not None
            and hasattr(model.model, "start_layer")
            and (
                layer_id < model.model.start_layer or layer_id >= model.model.end_layer
            )
        ):
            continue

        # Skip parameters with specified prefixes
        if any(prefix in name for prefix in skip_prefixes):
            continue

        # Skip tied weights if applicable
        if tie_word_embeddings and "lm_head.weight" in name:
            continue

        # Skip vision tower parameters if not in the model
        if name.startswith("model.vision_tower") and name not in params_dict:
            continue

        # Handle stacked parameters (qkv, gate_up)
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue

            # Skip experts in stacked param handling (handled by expert_params_mapping)
            if "mlp.experts" in name and expert_params_mapping is not None:
                break

            name_remapped = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models
            if name_remapped.endswith(".bias") and name_remapped not in params_dict:
                break

            if name_remapped in params_dict:
                param = params_dict[name_remapped]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
        else:
            # Handle expert parameters if applicable
            if expert_params_mapping is not None:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue

                    name_remapped = name.replace(weight_name, param_name)
                    param = params_dict[name_remapped]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name_remapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Handle regular parameters
                    _handle_regular_param(name, loaded_weight, params_dict)
            else:
                # Handle regular parameters (non-expert case)
                _handle_regular_param(name, loaded_weight, params_dict)

    # Return success
    return True


def _handle_regular_param(name, loaded_weight, params_dict):
    # Skip loading extra bias for GPTQ models
    if name.endswith(".bias") and name not in params_dict:
        return

    # Skip loading kv_scale if not in model
    if name.endswith(".kv_scale") and name not in params_dict:
        return

    if name in params_dict:
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
    else:
        logger.warning(f"Parameter {name} not found in params_dict")
