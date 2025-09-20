# Adapted from https://github.com/vllm-project/vllm/blob/bf214ca22625e311a2c4c0dfbf7af19128f4919c/vllm/distributed/device_communicators/symm_mem.py
import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.all_reduce_utils import (
    SYMM_MEM_ALL_REDUCE_MAX_SIZES,
)
from sglang.srt.utils import get_device_capability, is_cuda, is_hip

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem_available = False


logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()

symm_mem_is_available = False
if _is_hip:
    symm_mem_is_available = False
if _is_cuda:
    try:
        import sgl_kernel

        symm_mem_is_available = True
    except:
        symm_mem_is_available = False


class SymmMemCommunicator:
    _WORLD_SIZES_MULTIMEM = {
        9: [4, 6, 8],
        10: [6, 8],
    }

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        self.disabled = True

        if not symm_mem_available:
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        torch.cuda.set_device(device)
        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.device_capability = torch.cuda.get_device_capability(device)[0]
        if self.device_capability < 9:
            logger.warning(
                "SymmMemCommunicator: Device capability %s not supported, "
                "communicator is not available.",
                self.device_capability,
            )
            return
        if self.world_size not in SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability]:
            logger.warning(
                "SymmMemCommunicator: World size %d not supported, "
                "communicator is not available.",
                self.world_size,
            )
            return
        self.max_size = SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability][
            self.world_size
        ]
        self.buffer = torch_symm_mem.empty(
            self.max_size // self.dtype.itemsize,
            device=self.device,
            dtype=self.dtype,
        )
        handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
        if handle.multicast_ptr == 0:
            logger.warning(
                "SymmMemCommunicator: symmetric memory "
                "multicast operations are not supported."
            )
            return
        self.disabled = False

    def should_symm_mem_allreduce(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 4 != 0:
            return False
        return inp_size < self.max_size

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if out is None:
            out = torch.empty_like(inp)
        self.buffer[: inp.numel()].copy_(inp.view(-1))
        if self.world_size in self._WORLD_SIZES_MULTIMEM[self.device_capability]:
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        out.copy_(self.buffer[: inp.numel()].view(out.shape))
        return out
