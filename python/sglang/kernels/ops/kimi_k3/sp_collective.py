"""K3 SP-MoE reduce-scatter and all-gather.

The kernels are the shared NVLink collectives; this module is only the K3
facing edge of them, and carries the two things the kernel module does not: the
custom-op declarations, and the table saying which strategy to run at a given
size. The communicator they resolve through lives in `comm`.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Optional, TypeAlias

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import nvlink_comm as nvl
from sglang.kernels.ops.kimi_k3 import comm as k3_comm
from sglang.srt.utils.custom_op import register_custom_op

_Strategy: TypeAlias = Literal["push", "pull", "nccl"]


class Boundaries(NamedTuple):
    pull_from: Optional[int]  # None: only when the push workspace is too small
    nccl_from: Optional[int]  # None: never hand back to NCCL


# Measured at H=7168 bf16 on a single-node 8xB200
_BOUNDARIES: dict[tuple[int, int, str], Boundaries] = {
    (100, 8, "reduce_scatter"): Boundaries(pull_from=3072, nccl_from=None),
    (100, 8, "all_gather"): Boundaries(pull_from=3072, nccl_from=16384),
    (100, 4, "reduce_scatter"): Boundaries(pull_from=None, nccl_from=None),
    (100, 4, "all_gather"): Boundaries(pull_from=3072, nccl_from=6144),
}

# GB300 -> B200
_ARCH_ALIASES = {103: 100}


@cache_once
def _arch(device: torch.device) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    arch = major * 10 + minor
    return _ARCH_ALIASES.get(arch, arch)


def supported(world_size: int, device: torch.device) -> bool:
    """Whether this arch and world size have measured boundaries."""
    return (_arch(device), world_size, "reduce_scatter") in _BOUNDARIES


def choose_strategy(
    op: str,
    world_size: int,
    num_tokens: int,
    device: torch.device,
    *,
    push_fits: bool,
) -> _Strategy:
    bounds = _BOUNDARIES[(_arch(device), world_size, op)]
    if bounds.nccl_from is not None and num_tokens >= bounds.nccl_from:
        return "nccl"
    if not push_fits:
        return "pull"
    if bounds.pull_from is not None and num_tokens >= bounds.pull_from:
        return "pull"
    return "push"


@register_custom_op(mutates_args=["output"])
def _reduce_scatter_push_op(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> None:
    nvl.reduce_scatter_push(k3_comm.get(world_size), input, output, residual)


@register_custom_op(mutates_args=["output"])
def _reduce_scatter_pull_op(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_mc_ptr: int,
) -> None:
    nvl.reduce_scatter_pull(
        k3_comm.get(world_size), input, output, residual, in_mc_ptr=input_mc_ptr
    )


@register_custom_op(mutates_args=["output"])
def _all_gather_push_op(
    world_size: int, input: torch.Tensor, output: torch.Tensor
) -> None:
    nvl.all_gather_push(k3_comm.get(world_size), input, output)


@register_custom_op(mutates_args=["output"])
def _all_gather_pull_op(
    world_size: int, input: torch.Tensor, output: torch.Tensor, output_mc_ptr: int
) -> None:
    nvl.all_gather_pull(
        k3_comm.get(world_size), input, output, out_mc_ptr=output_mc_ptr
    )


def local_tokens(world_size: int, num_tokens: int) -> int:
    """This rank's share of a possibly ragged split, as the kernels route it."""
    return nvl.get_token_partion(num_tokens, k3_comm.get(world_size)).num_local_tokens


def reduce_scatter_push(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _reduce_scatter_push_op(world_size, input, output, residual)
    return output


def reduce_scatter_pull(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    input_mc_ptr: int,
) -> torch.Tensor:
    _reduce_scatter_pull_op(world_size, input, output, residual, input_mc_ptr)
    return output


def all_gather_push(
    world_size: int, input: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    _all_gather_push_op(world_size, input, output)
    return output


def all_gather_pull(
    world_size: int, input: torch.Tensor, output: torch.Tensor, *, output_mc_ptr: int
) -> torch.Tensor:
    _all_gather_pull_op(world_size, input, output, output_mc_ptr)
    return output
