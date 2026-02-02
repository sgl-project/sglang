# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/device_communicators/cuda_communicator.py


import os

import torch
from torch.distributed import ProcessGroup

from sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


class CudaCommunicator(DeviceCommunicatorBase):

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)

        from sglang.multimodal_gen.runtime.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        self.custom_allreduce = None
        self.pynccl_comm: PyNcclCommunicator | None = None
        if self.world_size > 1:
            enable_custom_all_reduce = (
                os.environ.get("SGLANG_DIFFUSION_ENABLE_CUSTOM_ALL_REDUCE", "0") == "1"
            )
            if enable_custom_all_reduce:
                # use lazy import here
                from sglang.srt.distributed.device_communicators.custom_all_reduce import (
                    dispatch_custom_allreduce,
                )

                CustomAllreduce = dispatch_custom_allreduce()
                self.custom_allreduce = CustomAllreduce(
                    group=self.cpu_group,
                    device=self.device,
                )

            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

    def all_reduce(self, input_, op: torch.distributed.ReduceOp | None = None):
        torch_op = op if op is not None else torch.distributed.ReduceOp.SUM

        pynccl_op = torch_op
        if not isinstance(torch_op, torch.distributed.ReduceOp):
            op_name = getattr(torch_op, "name", None)
            if op_name is None:
                op_name = str(torch_op).split(".")[-1]
            pynccl_op = getattr(torch.distributed.ReduceOp, op_name, None)

        custom_allreduce = self.custom_allreduce
        if custom_allreduce is not None and pynccl_op == torch.distributed.ReduceOp.SUM:
            out = custom_allreduce.custom_all_reduce(input_)
            if out is not None:
                return out

        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        out = None
        if pynccl_op is not None:
            out = pynccl_comm.all_reduce(input_, op=pynccl_op)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group, op=torch_op)
        return out

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self) -> None:
        if self.custom_allreduce is not None:
            self.custom_allreduce = None
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
