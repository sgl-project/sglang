import logging
import re

import torch

logger = logging.getLogger(__name__)


def get_layer_id(weight_name):
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None


def pad_or_narrow_weight(
    loaded_weight: torch.Tensor, input_dim: int, start_idx: int, shard_size: int
) -> torch.Tensor:
    # Padding with zeros for special case such as qwen2_5_VL's mlp which is not 8-aligned
    valid_size = max(loaded_weight.shape[input_dim] - start_idx, 0)

    if valid_size > 0:
        loaded_slice = loaded_weight.narrow(input_dim, start_idx, valid_size)
        pad_shape = list(loaded_weight.shape)
        pad_shape[input_dim] = shard_size - valid_size
        pad = torch.zeros(
            pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
        )
        return torch.cat([loaded_slice, pad], dim=input_dim)

    # All padding
    pad_shape = list(loaded_weight.shape)
    pad_shape[input_dim] = shard_size
    return torch.zeros(
        pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
    )


class PPMissingLayer(torch.nn.Identity):
    # Adapted from
    # https://github.com/vllm-project/vllm/blob/18ed3132d2bfe1df9a74729457b69243955221e8/vllm/model_executor/models/utils.py#L468C1-L486C1
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.return_tuple = kwargs.get("return_tuple", False)

    def forward(self, *args, **kwargs):
        """
        Return the first arg from args or the first value from kwargs.

        Wraps the input in a tuple if `self.return_tuple` is True.
        """
        input = args[0] if args else next(iter(kwargs.values()))
        return (input,) if self.return_tuple else input
