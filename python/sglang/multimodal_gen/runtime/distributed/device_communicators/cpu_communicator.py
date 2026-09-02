# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/cpu_communicator.py

import os

import torch
from torch.distributed import ProcessGroup

from sglang.multimodal_gen.runtime.distributed.utils import all_gather_single

from .base_device_communicator import DeviceCommunicatorBase


class CpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)

        self._group_shm_backends: dict[tuple[int, ...], _GroupSHMDistributed] = {}
        self._group_shm_available = self._load_group_shm_ops()

    def _load_group_shm_ops(self) -> bool:
        from sglang.multimodal_gen.runtime.platforms import current_platform
        from sglang.multimodal_gen.runtime.platforms.interface import CpuArchEnum

        if current_platform.get_cpu_architecture() != CpuArchEnum.X86:
            return False
        try:
            import sgl_kernel  # noqa: F401
        except ImportError:
            return False

        return (
            hasattr(torch.ops.sgl_kernel, "shm_group_initialize")
            and hasattr(torch.ops.sgl_kernel, "shm_group_allgather")
            and hasattr(torch.ops.sgl_kernel, "shm_group_alltoall")
            and hasattr(torch.ops.sgl_kernel, "shm_group_allreduce")
        )

    @staticmethod
    def _sanitize_shm_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)

    def _get_group_shm_backend(
        self,
        group: ProcessGroup,
    ) -> "_GroupSHMDistributed":
        group_ranks = tuple(torch.distributed.get_process_group_ranks(group))

        backend = self._group_shm_backends.get(group_ranks)
        if backend is None:
            backend = _GroupSHMDistributed(self, group)
            self._group_shm_backends[group_ranks] = backend

        return backend

    def _can_use_group_shm_allgather(
        self,
        input_: torch.Tensor,
    ) -> bool:
        return (
            self._group_shm_available
            and input_.device.type == "cpu"
            and input_.is_contiguous()
        )

    def _can_use_group_shm_allreduce(
        self,
        input_: torch.Tensor,
        op,
    ) -> bool:
        return (
            self._group_shm_available
            and op == torch.distributed.ReduceOp.SUM
            and input_.device.type == "cpu"
            and input_.is_contiguous()
            and input_.dtype
            in (
                torch.float32,
                torch.bfloat16,
                torch.float16,
            )
        )

    def _can_use_group_shm_alltoall(
        self,
        input_: torch.Tensor,
        output: torch.Tensor,
        group_size: int,
    ) -> bool:

        return (
            self._group_shm_available
            and input_.device.type == "cpu"
            and output.device.type == "cpu"
            and input_.is_contiguous()
            and output.is_contiguous()
            and input_.dtype == output.dtype
            and input_.numel() == output.numel()
            and input_.numel() % group_size == 0
        )

    def all_reduce(
        self,
        input_: torch.Tensor,
        op: torch.distributed.ReduceOp | None = torch.distributed.ReduceOp.SUM,
        group=None,
        async_op: bool = False,
    ) -> torch.Tensor:
        if group is None:
            group = self.device_group
        group_size = torch.distributed.get_world_size(group)
        if group_size == 1:
            return input_
        if not async_op and self._can_use_group_shm_allreduce(input_, op):
            backend = self._get_group_shm_backend(group)
            backend.all_reduce(input_)
            return input_

        torch.distributed.all_reduce(
            input_,
            op=op,
            group=group,
            async_op=async_op,
        )
        return input_

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
            input_,
            gather_list,
            dst=self.ranks[dst],
            group=self.device_group,
        )

        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def all_gather(
        self, input_: torch.Tensor, dim: int = -1, group=None
    ) -> torch.Tensor:
        if group is None:
            group = self.device_group
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        world_size = torch.distributed.get_world_size(group)
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * world_size,) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        if self._can_use_group_shm_allgather(input_):
            backend = self._get_group_shm_backend(group)
            backend.all_gather_single(
                output_tensor,
                input_,
            )
        else:
            all_gather_single(
                output_tensor,
                input_,
                group=group,
            )

        # Reshape
        output_tensor = output_tensor.reshape((world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        group: ProcessGroup | None = None,
    ) -> None:
        if group is None:
            group = self.device_group

        assert group is not None

        group_size = torch.distributed.get_world_size(group)

        if group_size == 1:
            output.copy_(input_)
            return

        if self._can_use_group_shm_alltoall(
            input_,
            output,
            group_size,
        ):
            backend = self._get_group_shm_backend(group)
            backend.all_to_all_single(
                output,
                input_,
            )
        else:
            torch.distributed.all_to_all_single(
                output,
                input_,
                group=group,
            )


class _GroupSHMDistributed:
    def __init__(
        self,
        communicator: CpuCommunicator,
        group: ProcessGroup,
    ):
        self.communicator = communicator
        self.group = group
        self.handle = self._init_group_shm()

    def _init_group_shm(self) -> int:
        group_ranks = tuple(torch.distributed.get_process_group_ranks(self.group))
        group_size = torch.distributed.get_world_size(self.group)
        group_rank = torch.distributed.get_rank(self.group)

        addr = self.communicator._sanitize_shm_name(
            os.environ.get("MASTER_ADDR", "localhost")
        )
        port = self.communicator._sanitize_shm_name(os.environ.get("MASTER_PORT", "0"))
        unique_name = self.communicator._sanitize_shm_name(
            self.communicator.unique_name
        )
        ranks_name = "_".join(str(rank) for rank in group_ranks)

        group_name = (
            f"sglang_group_"
            f"{os.getuid()}_"
            f"{addr}_"
            f"{port}_"
            f"{unique_name}_"
            f"{ranks_name}"
        )

        return int(
            torch.ops.sgl_kernel.shm_group_initialize(
                group_name,
                group_size,
                group_rank,
            )
        )

    def all_reduce(self, input_: torch.Tensor) -> None:
        torch.ops.sgl_kernel.shm_group_allreduce(
            self.handle,
            input_,
        )

    def all_gather_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
    ) -> None:
        torch.ops.sgl_kernel.shm_group_allgather(
            self.handle,
            output,
            input_,
        )

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
    ) -> None:
        torch.ops.sgl_kernel.shm_group_alltoall(
            self.handle,
            output,
            input_,
        )
