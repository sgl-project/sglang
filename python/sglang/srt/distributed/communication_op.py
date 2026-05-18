# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed

from sglang.srt.tp_invariant_ops import tree_all_reduce_sum
from sglang.srt.true_on_policy import should_use_tp_invariant_tree_all_reduce

from .parallel_state import (
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_tp_group,
    get_tp_group,
)


def _maybe_custom_tree_all_reduce(input_: torch.Tensor, group) -> Optional[torch.Tensor]:
    if (
        os.environ.get("SGLANG_TRUE_ON_POLICY_CUSTOM_TREE_ALL_REDUCE", "0") != "1"
        and os.environ.get("SGLANG_TRUE_ON_POLICY_TREE_CUSTOM_ALL_REDUCE", "0")
        != "1"
    ):
        return None

    ca_comm = getattr(group, "ca_comm", None)
    custom_tree_all_reduce = getattr(ca_comm, "custom_tree_all_reduce", None)
    if custom_tree_all_reduce is None:
        return None
    return custom_tree_all_reduce(input_)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    if should_use_tp_invariant_tree_all_reduce():
        return tensor_model_parallel_tree_all_reduce(input_)
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_tree_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group in fixed tree order."""
    group = get_tp_group()
    custom_result = _maybe_custom_tree_all_reduce(input_, group)
    if custom_result is not None:
        return custom_result
    return tree_all_reduce_sum(input_, device_group=group.device_group)


def attention_tensor_model_parallel_tree_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across attention TP group in fixed tree order."""
    group = get_attn_tp_group()
    custom_result = _maybe_custom_tree_all_reduce(input_, group)
    if custom_result is not None:
        return custom_result
    return tree_all_reduce_sum(input_, device_group=group.device_group)


def tensor_model_parallel_fused_allreduce_rmsnorm(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused TP all-reduce + RMSNorm.

    Policy and backend selection are owned by GroupCoordinator:
    it may dispatch to communicator-native fused APIs, custom fused kernels,
    or return None so callers can run generic fallback paths.
    """
    return get_tp_group().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)


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


def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    if should_use_tp_invariant_tree_all_reduce():
        return attention_tensor_model_parallel_tree_all_reduce(input_)
    return get_attn_tp_group().all_reduce(input_)


def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe parallel group."""
    return get_moe_tp_group().all_reduce(input_)


def moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe expert parallel group."""
    return get_moe_ep_group().all_reduce(input_)


def moe_expert_parallel_tree_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe expert parallel group in fixed tree order."""
    from sglang.srt.tp_invariant_ops import tree_all_reduce_sum

    group = get_moe_ep_group()
    custom_result = _maybe_custom_tree_all_reduce(input_, group)
    if custom_result is not None:
        return custom_result
    return tree_all_reduce_sum(input_, device_group=group.device_group)
