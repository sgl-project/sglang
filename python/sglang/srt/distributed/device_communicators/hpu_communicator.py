# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/distributed/device_communicators/hpu_communicator.py
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_hpu

if is_hpu():
    import habana_frameworks.torch as htorch  # noqa: F401


class HpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not is_hpu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # FIXME(kzawora): this is a workaround for a bug in Habana PT bridge
        # occurring when PT_HPU_ENABLE_LAZY_COLLECTIVES=true env var is used
        # (which is required for tensor parallel HPUGraph inference)
        htorch.core.mark_step()
        dist.all_reduce(x, group=self.group)
        return x

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if dim < 0:
            # Convert negative dim to positive.
            dim += x.dim()
        input_size = x.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (world_size,) + input_size, dtype=x.dtype, device=x.device
        )
        # All-gather.
        htorch.core.mark_step()
        dist.all_gather_into_tensor(output_tensor, x, group=self.group)
        # Reshape
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor
