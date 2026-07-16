# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/bf214ca22625e311a2c4c0dfbf7af19128f4919c/vllm/distributed/device_communicators/symm_mem.py
import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_cuda, is_hip

from .base import AllReduceMode, BaseCommunicator

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    _is_cuda = is_cuda()
    _is_hip = is_hip()

    torch_symm_mem_available = False
    if _is_cuda:
        torch_symm_mem_available = True
except ImportError:
    torch_symm_mem_available = False


logger = logging.getLogger(__name__)


class TorchSymmMemCommunicator(BaseCommunicator):
    """
    Thin wrapper around torch-symmetric-memory collectives.

    This communicator:
      - Validates device capability and world size.
      - Allocates a shared symmetric buffer.
      - Chooses between 'multimem' and 'two-shot' all-reduce kernels.
      - Exposes a fast-path all_reduce() compatible with bfloat16 inputs.

    If any prerequisite is not met, the instance remains disabled and will
    decline to perform symmetric-memory all-reduce.
    """

    name = "torch_symm_mem"

    # Mapping: compute capability major -> supported world sizes for multimem
    # If the current (cc_major, world_size) is not listed, we fall back
    # to the two-shot path.
    _WORLD_SIZES_MULTIMEM = {
        9: [4, 6, 8],
        10: [6, 8],
    }

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        """
        Args:
            group: Torch process group used for rendezvous and naming.
            device: Target CUDA device (index, 'cuda:X', or torch.device).
        """

        if not torch_symm_mem_available:
            raise RuntimeError(
                "TorchSymmMemCommunicator requires torch symmetric memory support, "
                "but it is not available. Ensure you have the correct PyTorch version "
                "and that your hardware supports it."
            )

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
        supported_max_sizes = TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES.get(
            self.device_capability
        )
        if supported_max_sizes is None:
            raise RuntimeError(
                "TorchSymmMemCommunicator: "
                f"Device capability {self.device_capability} not supported, "
                "communicator is not available."
            )
        if self.world_size not in supported_max_sizes:
            raise RuntimeError(
                "TorchSymmMemCommunicator: "
                f"World size {self.world_size} not supported, "
                "communicator is not available."
            )
        self.max_size = supported_max_sizes[self.world_size]
        self.buffer = torch_symm_mem.empty(
            self.max_size // self.dtype.itemsize,
            device=self.device,
            dtype=self.dtype,
        )
        handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
        if handle.multicast_ptr == 0:
            self.buffer = None
            raise RuntimeError(
                "TorchSymmMemCommunicator: torch symmetric memory "
                "multicast operations are not supported."
            )
        super().__init__(world_size=self.world_size)

    def should_use_custom_op(self) -> bool:
        return True

    def get_all_reduce_mode(self, input_: torch.Tensor) -> Optional[AllReduceMode]:
        """
        Fast-path eligibility check for a given tensor.

        Conditions:
          - Communicator must be enabled.
          - dtype must be bfloat16 (matches kernel + buffer dtype).
          - Total byte size must be 4-byte aligned (hardware requirement).
          - Payload must be smaller than the symmetric-memory max size.

        Returns:
            `AllReduceMode.OUTPLACE` if the symmetric-memory path can handle
            this tensor.
        """
        if self.disabled:
            return None
        if input_.device != self.device:
            return None
        if input_.dtype != self.dtype:
            return None
        inp_size = input_.numel() * input_.element_size()
        # enforce 4-byte alignment
        if inp_size % 4 != 0:
            return None
        return AllReduceMode.OUTPLACE if inp_size < self.max_size else None

    @BaseCommunicator.validate
    def all_reduce(
        self,
        input_: torch.Tensor,
        *,
        inplace: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Perform an out-of-place sum all-reduce via torch symmetric memory.

        Args:
            input_: Input tensor on the target CUDA device (bfloat16).
            inplace: Must be `False` or `None`.

        Returns:
            The reduced tensor.

        Implementation details:
            - Stages 'input_' into the symmetric buffer.
            - Selects 'multimem' or 'two_shot' kernel based on topology.
            - Copies the result into a newly allocated output tensor.
        """
        self.assert_outplace("all_reduce", inplace)
        assert self.buffer is not None, "Symmetric buffer not initialized"
        out = torch.empty_like(input_)
        self.buffer[: input_.numel()].copy_(input_.view(-1))
        if self.world_size in self._WORLD_SIZES_MULTIMEM.get(
            self.device_capability, ()
        ):
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[: input_.numel()], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[: input_.numel()], "sum", self.group.group_name
            )
        out.copy_(self.buffer[: input_.numel()].view(out.shape))
        return out


MiB = 1024 * 1024

TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    9: {
        2: 64 * MiB,  # 64 MB
        4: 64 * MiB,  # 64 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
    10: {
        2: 64 * MiB,  # 64 MB
        4: 64 * MiB,  # 64 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
}
