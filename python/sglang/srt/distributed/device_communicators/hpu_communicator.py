# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/hpu_communicator.py

from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_hpu

from .base import AllReduceMode, BaseCommunicator

if is_hpu():
    import habana_frameworks.torch as htorch  # noqa: F401


class HpuCommunicator(BaseCommunicator):
    name = "hpu"

    def __init__(self, group: ProcessGroup):
        self.group = group
        super().__init__(dist.get_world_size(self.group), disabled=not is_hpu())

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
        # FIXME(kzawora): this is a workaround for a bug in Habana PT bridge
        # occurring when PT_HPU_ENABLE_LAZY_COLLECTIVES=true env var is used
        # (which is required for tensor parallel HPUGraph inference)
        htorch.core.mark_step()
        dist.all_reduce(input_, group=self.group)
        return input_

    @BaseCommunicator.validate
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # HPU performs the concat-style all-gather (and the trailing reshape)
        # itself so the Habana PT bridge mark_step workaround stays attached to
        # this collective; GroupCoordinator.all_gather dispatches here directly
        # rather than through the generic all_gather_into_tensor path.
        world_size = self.world_size
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_tensor = torch.empty(
            (world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        htorch.core.mark_step()
        dist.all_gather_into_tensor(output_tensor, input_, group=self.group)
        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )
