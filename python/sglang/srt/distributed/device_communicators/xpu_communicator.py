# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/xpu_communicator.py

from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_xpu

from .base import AllReduceMode, BaseCommunicator


class XpuCommunicator(BaseCommunicator):
    name = "xpu"

    def __init__(self, rank_in_group: int, group: ProcessGroup):
        self.rank_in_group = rank_in_group
        self.group = group
        super().__init__(dist.get_world_size(group), disabled=not is_xpu())

    def should_use_custom_op(self) -> bool:
        # Route through the registered custom op so Dynamo treats the
        # all-reduce as an opaque call and does not decompose it into
        # _c10d_functional primitives (which invoke sycl_event.wait() and
        # break XPU graph capture).
        return True

    def get_all_reduce_mode(self, input_: torch.Tensor) -> Optional[AllReduceMode]:
        return AllReduceMode.INPLACE

    @BaseCommunicator.validate
    def all_reduce(
        self,
        input_: torch.Tensor,
        *,
        inplace: Optional[bool] = None,
    ) -> torch.Tensor:
        self.assert_inplace("all_reduce", inplace)
        dist.all_reduce(input_, group=self.group)
        return input_

    @BaseCommunicator.validate
    def gather(
        self,
        input_: torch.Tensor,
        dst: int,
        *,
        dim: int = 0,
    ) -> Optional[torch.Tensor]:
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (self.world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        torch.distributed.all_gather_into_tensor(
            output_tensor, input_, group=self.group
        )
        if self.rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        else:
            output_tensor = None
        return output_tensor
