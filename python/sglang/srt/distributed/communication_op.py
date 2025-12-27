# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_o_proj_dp_group, get_o_proj_tp_group, get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


def o_proj_tensor_model_parallel_reduce_scatter_tensor(
    output: torch.Tensor, input_: torch.Tensor
) -> torch.Tensor:
    """Reduce-scatter the input tensor across o_proj tp group."""
    return get_o_proj_tp_group().reduce_scatter_tensor(output, input_)


def o_proj_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across o_proj tp group."""
    return get_o_proj_tp_group().all_reduce(input_)


def o_proj_data_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across o_proj dp group."""
    return get_o_proj_dp_group().all_gather(input_, dim)
