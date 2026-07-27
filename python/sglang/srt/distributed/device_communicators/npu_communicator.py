import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_npu

_is_npu = is_npu()

if _is_npu:
    from torch_npu import npu_dynamic_quant


class NpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not _is_npu:
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x

    def quant_all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """
        Note:
        All reduce is split into All gather + reduce.
        All gather is performed in low precision, but reduce in full precision.
        """
        world_size = self.world_size
        input_size = x.size()
        output_size = (input_size[0] * world_size,) + input_size[1:]
        x_q, scale = npu_dynamic_quant(x, dst_type=torch.int8)
        # Allocate output tensor.
        output_tensor = torch.empty(output_size, dtype=x_q.dtype, device=x.device)
        output_scale = torch.empty(
            output_size[:1], dtype=scale.dtype, device=scale.device
        )
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, x_q, group=self.group)
        dist.all_gather_into_tensor(output_scale, scale, group=self.group)

        output_tensor = output_tensor.to(x.dtype) * output_scale.unsqueeze(-1).to(
            x.dtype
        )
        # Reshape
        output_tensor = output_tensor.reshape((world_size,) + input_size)

        return output_tensor.sum(dim=0)

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if dim < 0:
            # Convert negative dim to positive.
            dim += x.dim()
        input_size = x.size()
        output_size = (input_size[0] * world_size,) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size, dtype=x.dtype, device=x.device)
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, x, group=self.group)
        # Reshape
        output_tensor = output_tensor.reshape((world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor
