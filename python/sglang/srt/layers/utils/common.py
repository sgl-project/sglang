# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import re

import torch
from torch.nn.parameter import Parameter

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


def is_strict_contiguous(x: torch.Tensor) -> bool:
    expected_stride = 1
    for size, stride in zip(reversed(x.shape), reversed(x.stride())):
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True


def strict_contiguous(x: torch.Tensor) -> torch.Tensor:
    if is_strict_contiguous(x):
        return x
    return x.clone(memory_format=torch.contiguous_format)


def copy_or_rebind_param(
    module: torch.nn.Module, name: str, new_value: torch.Tensor
) -> None:
    """Keep parameter identities stable for CUDA graph reuse and hot reload."""
    new_value = new_value.detach()
    param = getattr(module, name, None)
    if isinstance(param, Parameter):
        if param.data.shape == new_value.shape and param.data.dtype == new_value.dtype:
            param.data.copy_(new_value)
        else:
            param.data = new_value
        param.requires_grad_(False)
    else:
        setattr(module, name, Parameter(new_value, requires_grad=False))


def alias_or_bind_derived_param(
    module: torch.nn.Module,
    source_name: str,
    derived_name: str,
    derived_value: torch.Tensor,
) -> None:
    """Bind a post-processed (derived) tensor to a derived attribute name.

    When `derived_value` is broadcastable to the source Parameter's shape (and
    dtype matches), write it broadcast-filled into the source's storage in
    place and register `derived_name` as an alias of the source Parameter. The
    two attribute names then share one underlying buffer, so:
      - apply() can read via `derived_name`
      - update_weights_from_disk can keep refilling `source_name` (the loader
        re-runs process_weights_after_loading which re-derives in place)
      - peak GPU memory is the source size, not source + derived.

    When the shapes are not broadcast-compatible, fall back to allocating a
    separate Parameter under `derived_name` via copy_or_rebind_param.
    """
    derived_value = derived_value.detach()
    source = getattr(module, source_name, None)
    if isinstance(source, Parameter) and source.data.dtype == derived_value.dtype:
        try:
            broadcast = torch.broadcast_to(derived_value, source.data.shape)
        except RuntimeError:
            broadcast = None
        if broadcast is not None:
            source.data.copy_(broadcast)
            source.requires_grad_(False)
            setattr(module, derived_name, source)
            return
    copy_or_rebind_param(module, derived_name, derived_value)


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
