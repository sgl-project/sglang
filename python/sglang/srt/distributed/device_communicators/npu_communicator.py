# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/npu_communicator.py

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_npu

from .base import BaseCommunicator

if is_npu():
    from torch_npu import npu_dynamic_quant


class NpuCommunicator(BaseCommunicator):
    """NPU-specific collectives.

    Plain all-reduce / all-gather on NPU are ordinary ``torch.distributed`` ops
    and are served by the default ``TorchDefaultCommunicator``, so this class
    only carries the NPU-specific ``quant_all_reduce`` path. It is therefore not
    registered in the dispatch lists; ``GroupCoordinator`` calls it directly.
    """

    name = "npu"

    def __init__(self, group: ProcessGroup):
        self.group = group
        super().__init__(dist.get_world_size(self.group), disabled=not is_npu())

    @BaseCommunicator.validate
    def quant_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        All-reduce split into all-gather + reduce: gather in low precision
        (int8), then reduce in full precision.
        """
        world_size = self.world_size
        input_size = input_.size()
        output_size = (input_size[0] * world_size,) + input_size[1:]
        x_q, scale = npu_dynamic_quant(input_, dst_type=torch.int8)
        # Allocate output tensors.
        output_tensor = torch.empty(output_size, dtype=x_q.dtype, device=input_.device)
        output_scale = torch.empty(
            output_size[:1], dtype=scale.dtype, device=scale.device
        )
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, x_q, group=self.group)
        dist.all_gather_into_tensor(output_scale, scale, group=self.group)
        # Dequantize and reduce.
        output_tensor = output_tensor.to(input_.dtype) * output_scale.unsqueeze(-1).to(
            input_.dtype
        )
        output_tensor = output_tensor.reshape((world_size,) + input_size)
        return output_tensor.sum(dim=0)
