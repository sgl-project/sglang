# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/device_communicators/base_device_communicator.py

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup, ReduceOp


class DistributedAutograd:
    """Collection of autograd functions for distributed operations.

    This class provides custom autograd functions for distributed operations like all_reduce,
    all_gather, and all_to_all. Each operation is implemented as a static inner class with
    proper forward and backward implementations.
    """

    class AllReduce(torch.autograd.Function):
        """Differentiable all_reduce operation.

        The gradient of all_reduce is another all_reduce operation since the operation
        combines values from all ranks equally.
        """

        @staticmethod
        def forward(
            ctx: Any,
            group: ProcessGroup,
            input_: Tensor,
            op: dist.ReduceOp | None = None,
        ) -> Tensor:
            ctx.group = group
            ctx.op = op
            output = input_.clone()
            dist.all_reduce(output, group=group, op=op)
            return output

        @staticmethod
        def backward(ctx: Any, grad_output: Tensor) -> tuple[None, Tensor, None]:
            grad_output = grad_output.clone()
            dist.all_reduce(grad_output, group=ctx.group, op=ctx.op)
            return None, grad_output, None

    class AllGather(torch.autograd.Function):
        """Differentiable all_gather operation.

        The operation gathers tensors from all ranks and concatenates them along a specified dimension.
        The backward pass uses reduce_scatter to efficiently distribute gradients back to source ranks.
        """

        @staticmethod
        def forward(
            ctx: Any, group: ProcessGroup, input_: Tensor, world_size: int, dim: int
        ) -> Tensor:
            ctx.group = group
            ctx.world_size = world_size
            ctx.dim = dim
            ctx.input_shape = input_.shape

            input_size = input_.size()
            output_size = (input_size[0] * world_size,) + input_size[1:]
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )

            dist.all_gather_into_tensor(output_tensor, input_, group=group)

            output_tensor = output_tensor.reshape((world_size,) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
            return output_tensor

        @staticmethod
        def backward(ctx: Any, grad_output: Tensor) -> tuple[None, Tensor, None, None]:
            # Split the gradient tensor along the gathered dimension
            dim_size = grad_output.size(ctx.dim) // ctx.world_size
            grad_chunks = grad_output.reshape(
                grad_output.shape[: ctx.dim]
                + (ctx.world_size, dim_size)
                + grad_output.shape[ctx.dim + 1 :]
            )
            grad_chunks = grad_chunks.movedim(ctx.dim, 0)

            # Each rank only needs its corresponding gradient
            grad_input = torch.empty(
                ctx.input_shape, dtype=grad_output.dtype, device=grad_output.device
            )
            dist.reduce_scatter_tensor(
                grad_input, grad_chunks.contiguous(), group=ctx.group
            )

            return None, grad_input, None, None

    class AllToAll4D(torch.autograd.Function):
        """Differentiable all_to_all operation specialized for 4D tensors.

        This operation is particularly useful for attention operations where we need to
        redistribute data across ranks for efficient parallel processing.

        The operation supports two modes:
        1. scatter_dim=2, gather_dim=1: Used for redistributing attention heads
        2. scatter_dim=1, gather_dim=2: Used for redistributing sequence dimensions
        """

        @staticmethod
        def forward(
            ctx: Any,
            group: ProcessGroup,
            input_: Tensor,
            world_size: int,
            scatter_dim: int,
            gather_dim: int,
        ) -> Tensor:
            ctx.group = group
            ctx.world_size = world_size
            ctx.scatter_dim = scatter_dim
            ctx.gather_dim = gather_dim

            if world_size == 1:
                return input_

            assert (
                input_.dim() == 4
            ), f"input must be 4D tensor, got {input_.dim()} and shape {input_.shape}"

            if scatter_dim == 2 and gather_dim == 1:
                bs, shard_seqlen, hn, hd = input_.shape
                seqlen = shard_seqlen * world_size
                shard_hn = hn // world_size

                input_ = input_.transpose(0, 2).contiguous()  # hn, shard_seqlen, bs, hd
                output = torch.empty_like(input_)

                dist.all_to_all_single(
                    output, input_, group=group
                )  # hn, shard_seqlen, bs, hd

                output = torch.cat(
                    output.split(shard_hn), dim=1
                )  # sharded hn, seqlen, bs, hd

                output = output.transpose(
                    0, 2
                ).contiguous()  # bs, seqlen, sharded_hn, hd

                return output
            elif scatter_dim == 1 and gather_dim == 2:
                bs, seqlen, shard_hn, hd = input_.shape
                hn = shard_hn * world_size
                shard_seqlen = seqlen // world_size

                input_ = input_.transpose(0, 2).contiguous()  # shard_hn, seqlen, bs, hd

                input_ = (
                    input_.reshape(shard_hn, world_size, shard_seqlen, bs, hd)
                    .transpose(0, 1)
                    .reshape(shard_hn * world_size, shard_seqlen, bs, hd)
                    .contiguous()
                )

                output = torch.empty_like(input_)

                dist.all_to_all_single(output, input_, group=group)

                output = output.transpose(
                    0, 2
                ).contiguous()  # bs, seqlen, sharded_hn, hd

                return output
            else:
                raise RuntimeError(
                    f"Invalid scatter_dim={scatter_dim}, gather_dim={gather_dim}. "
                    f"Only (scatter_dim=2, gather_dim=1) and (scatter_dim=1, gather_dim=2) are supported."
                )

        @staticmethod
        def backward(
            ctx: Any, grad_output: Tensor
        ) -> tuple[None, Tensor, None, None, None]:
            if ctx.world_size == 1:
                return None, grad_output, None, None, None

            # For backward pass, we swap scatter_dim and gather_dim
            output = DistributedAutograd.AllToAll4D.apply(
                ctx.group, grad_output, ctx.world_size, ctx.gather_dim, ctx.scatter_dim
            )
            return None, output, None, None, None


class DeviceCommunicatorBase:
    """
    Base class for device-specific communicator with autograd support.
    It can use the `cpu_group` to initialize the communicator.
    If the device has PyTorch integration (PyTorch can recognize its
    communication backend), the `device_group` will also be given.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)
        self.ranks = dist.get_process_group_ranks(cpu_group)
        self.global_rank = dist.get_rank()
        self.global_world_size = dist.get_world_size()
        self.rank_in_group = dist.get_group_rank(self.cpu_group, self.global_rank)

    def all_reduce(
        self, input_: torch.Tensor, op: dist.ReduceOp | None = ReduceOp.SUM
    ) -> torch.Tensor:
        """Performs an all_reduce operation with gradient support."""
        return DistributedAutograd.AllReduce.apply(self.device_group, input_, op)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Performs an all_gather operation with gradient support."""
        if dim < 0:
            dim += input_.dim()
        return DistributedAutograd.AllGather.apply(
            self.device_group, input_, self.world_size, dim
        )

    def all_to_all_4D(
        self, input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1
    ) -> torch.Tensor:
        """Performs a 4D all-to-all operation with gradient support."""
        return DistributedAutograd.AllToAll4D.apply(
            self.device_group, input_, self.world_size, scatter_dim, gather_dim
        )

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self) -> None:
        pass
