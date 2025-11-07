# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/communication_op.py

import torch
import torch.distributed

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_sp_group,
    get_tp_group,
)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


# TODO: remove model, make it sequence_parallel
def sequence_model_parallel_all_to_all_4D(
    input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1
) -> torch.Tensor:
    """All-to-all communication of 4D tensors (e.g. QKV matrices) across sequence parallel group."""
    return get_sp_group().all_to_all_4D(input_, scatter_dim, gather_dim)


def sequence_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_sp_group().all_gather(input_, dim)


def cfg_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1, separate_tensors: bool = False
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_cfg_group().all_gather(input_, dim, separate_tensors)


def cfg_model_parallel_all_reduce(
    input_: torch.Tensor,
    op: torch._C._distributed_c10d.ReduceOp = torch._C._distributed_c10d.ReduceOp.SUM,
) -> torch.Tensor:
    """All-reduce the input tensor across CFG parallel group."""
    return get_cfg_group().all_reduce(input_, op=op)
