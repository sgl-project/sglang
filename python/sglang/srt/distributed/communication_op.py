# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed

from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)

from .parallel_state import (
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_tp_group,
    get_tp_group,
)


def _tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(input_)


bcg_tensor_model_parallel_all_reduce = eager_on_graph(True)(
    _tensor_model_parallel_all_reduce
)

_enable_cuda_graph_collective_break = False


def set_cuda_graph_collective_break(enabled: bool) -> bool:
    global _enable_cuda_graph_collective_break
    prev = _enable_cuda_graph_collective_break
    _enable_cuda_graph_collective_break = enabled
    return prev


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    if _enable_cuda_graph_collective_break:
        return bcg_tensor_model_parallel_all_reduce(input_)
    return _tensor_model_parallel_all_reduce(input_)


def tensor_model_parallel_quant_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().quant_all_reduce(input_)


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


def _attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_attn_tp_group().all_reduce(input_)


bcg_attention_tensor_model_parallel_all_reduce = eager_on_graph(True)(
    _attention_tensor_model_parallel_all_reduce
)


def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    if _enable_cuda_graph_collective_break:
        return bcg_attention_tensor_model_parallel_all_reduce(input_)
    return _attention_tensor_model_parallel_all_reduce(input_)


def attention_tensor_model_parallel_quant_all_reduce(
    input_: torch.Tensor,
) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    return get_attn_tp_group().quant_all_reduce(input_)


def _moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_moe_tp_group().all_reduce(input_)


bcg_moe_tensor_model_parallel_all_reduce = eager_on_graph(True)(
    _moe_tensor_model_parallel_all_reduce
)


def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe parallel group."""
    if _enable_cuda_graph_collective_break:
        return bcg_moe_tensor_model_parallel_all_reduce(input_)
    return _moe_tensor_model_parallel_all_reduce(input_)


def _moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_moe_ep_group().all_reduce(input_)


bcg_moe_expert_parallel_all_reduce = eager_on_graph(True)(
    _moe_expert_parallel_all_reduce
)


def moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe expert parallel group."""
    if _enable_cuda_graph_collective_break:
        return bcg_moe_expert_parallel_all_reduce(input_)
    return _moe_expert_parallel_all_reduce(input_)
