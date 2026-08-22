# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/cpu_communicator.py

import os

import torch
from torch.distributed import ProcessGroup

from .base_device_communicator import DeviceCommunicatorBase


class CpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        from sglang.multimodal_gen.runtime.platforms import current_platform
        from sglang.multimodal_gen.runtime.platforms.interface import CpuArchEnum

        super().__init__(cpu_group, device, device_group, unique_name)
        self.dist_module = torch.distributed

        if (
            (current_platform.get_cpu_architecture() == CpuArchEnum.X86)
            and hasattr(torch.ops._C, "init_shm_manager")
            and unique_name.startswith("tp")
        ):
            self.dist_module = _CPUSHMDistributed(self)

        self._group_shm_handles: dict[tuple[int, ...], int] = {}
        self._group_shm_available = self._load_group_shm_ops()

    def _load_group_shm_ops(self) -> bool:
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

    def _get_group_shm_handle(self, group: ProcessGroup) -> int:
        group_ranks = tuple(torch.distributed.get_process_group_ranks(group))

        handle = self._group_shm_handles.get(group_ranks)
        if handle is not None:
            return handle

        group_size = torch.distributed.get_world_size(group)
        group_rank = torch.distributed.get_rank(group)

        # Example:
        #
        #   sglang_group_1000_localhost_29500_sp_group_0_0_2
        addr = self._sanitize_shm_name(os.environ.get("MASTER_ADDR", "localhost"))
        port = self._sanitize_shm_name(os.environ.get("MASTER_PORT", "0"))
        unique_name = self._sanitize_shm_name(self.unique_name)
        ranks_name = "_".join(str(rank) for rank in group_ranks)

        group_name = (
            f"sglang_group_"
            f"{os.getuid()}_"
            f"{addr}_"
            f"{port}_"
            f"{unique_name}_"
            f"{ranks_name}"
        )
        handle = int(
            torch.ops.sgl_kernel.shm_group_initialize(
                group_name, group_size, group_rank
            )
        )
        self._group_shm_handles[group_ranks] = handle

        return handle

    def _can_use_group_shm_allgather(
        self,
        input_: torch.Tensor,
    ) -> bool:
        data_size = input_.numel() * input_.element_size()

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
        if not async_op and self._can_use_group_shm_allreduce(
            input_,
            op,
        ):

            handle = self._get_group_shm_handle(group)
            torch.ops.sgl_kernel.shm_group_allreduce(
                handle,
                input_,
            )
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
        self.dist_module.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
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

        if self._can_use_group_shm_allgather(input_):
            handle = self._get_group_shm_handle(group)

            torch.ops.sgl_kernel.shm_group_allgather(
                handle,
                output_tensor,
                input_,
            )
        else:
            self.dist_module.all_gather_into_tensor(
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

        if self._can_use_group_shm_alltoall(input_, output, group_size):
            handle = self._get_group_shm_handle(group)

            torch.ops.sgl_kernel.shm_group_alltoall(
                handle,
                output,
                input_,
            )
        else:
            torch.distributed.all_to_all_single(
                output,
                input_,
                group=group,
            )


class _CPUSHMDistributed:

    def __init__(self, communicator: CpuCommunicator):
        instance_identifier = os.environ["VLLM_DIST_IDENT"]
        unique_name = communicator.unique_name
        instance_identifier = f"{instance_identifier}-{unique_name}"
        self.communicator = communicator

        group_ranks = [str(rank) for rank in self.communicator.ranks]
        shm_group_identifier = f"[{'-'.join(group_ranks)}]"
        self.group_name = f"{instance_identifier}-{shm_group_identifier}-cpushm"

        self.handle = self._init_cpu_shm()

    def _init_cpu_shm(self) -> int:
        handle = torch.ops._C.init_shm_manager(
            self.group_name,
            self.communicator.world_size,
            self.communicator.rank,
        )
        torch.distributed.barrier(self.communicator.device_group)
        torch.ops._C.join_shm_manager(
            handle,
            self.group_name,
        )
        torch.distributed.barrier(self.communicator.device_group)

        return int(handle)

    def all_reduce(
        self, input: torch.Tensor, group: ProcessGroup | None = None
    ) -> None:
        torch.ops._C.shm_allreduce(self.handle, input)

    def gather(
        self,
        input: torch.Tensor,
        gather_list: list[torch.Tensor] | None,
        dst: int = -1,
        group: ProcessGroup | None = None,
    ) -> None:
        # Note: different from the torch gather, here we use local dst rank.
        torch.ops._C.shm_gather(
            self.handle,
            input,
            gather_list,
            torch.distributed.get_group_rank(group, dst),
        )

    def all_gather_into_tensor(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: ProcessGroup | None = None,
    ) -> None:
        torch.ops._C.shm_all_gather(self.handle, input, output)

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: ProcessGroup | None = None,
    ) -> None:
        """
        TODO: Replace this with a native SHM all-to-all primitive to avoid
        gathering data that the destination rank does not need.
        """

        TORCH_WORLD_SIZE = self.communicator.world_size
        rank = self.communicator.rank_in_group

        assert input.device.type == "cpu"
        assert output.device.type == "cpu"
        assert input.is_contiguous()
        assert output.is_contiguous()
        assert input.dtype == output.dtype
        assert input.numel() == output.numel()
        assert input.numel() % TORCH_WORLD_SIZE == 0
        chunk_numel = input.numel() // TORCH_WORLD_SIZE

        input_flat = input.view(-1)
        output_flat = output.view(-1)

        gathered = torch.empty(
            TORCH_WORLD_SIZE * input_flat.numel(),
            dtype=input.dtype,
            device=input.device,
        )
        torch.ops._C.shm_all_gather(self.handle, input_flat, gathered)

        gathered = gathered.view(TORCH_WORLD_SIZE, TORCH_WORLD_SIZE, chunk_numel)

        output_flat.copy_(gathered[:, rank, :].reshape(-1))
