"""Symmetric-memory all-reduce for Intel XPU."""

import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.symm_mem_workspace import (
    SymmMemWorkspace,
    all_ranks_agree,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_xpu

logger = logging.getLogger(__name__)


class XpuSymmMemCommunicator:
    """One-shot all-reduce over torch symmetric memory on Intel XPU.

    A separate class from ``TorchSymmMemCommunicator``, which is built on
    ``symm_mem::{multimem,two_shot}_all_reduce_``: torch-xpu-ops registers
    neither, and Xe reports ``multicast_ptr == 0``. Only the allocator half of
    the backend exists for XPU, hence the Triton reduce kernel.
    """

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        self.disabled = True
        self.workspace: Optional[SymmMemWorkspace] = None
        self.max_bytes = 0
        self._staging = {}
        if isinstance(device, int):
            device = torch.device(f"xpu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(group)

        if not is_xpu():
            return

        # Lazy: the reduce kernel pulls in Triton.
        from sglang.srt.hardware_backend.xpu.kernels.comm.one_shot_all_reduce import (
            SUPPORTED_DTYPES,
        )

        max_bytes = envs.SGLANG_XPU_SYMM_MEM_MAX_BYTES.get()
        # The buffer is reinterpreted as every supported dtype, so keep it
        # aligned to the widest element (fp32).
        self.max_bytes = max_bytes - max_bytes % 4
        if self.max_bytes <= 0:
            logger.warning(
                "XpuSymmMemCommunicator: SGLANG_XPU_SYMM_MEM_MAX_BYTES=%d leaves "
                "no usable buffer, communicator is not available.",
                max_bytes,
            )
            return

        workspace = SymmMemWorkspace.create(
            group=group,
            device=device,
            nbytes=self.max_bytes,
            owner="XpuSymmMemCommunicator",
        )
        # Agreed on every rank, including one whose rendezvous failed alone:
        # ranks that disagree would diverge on every later all-reduce.
        ready = workspace is not None and self._compile_kernel(workspace)
        if not all_ranks_agree(ready, group):
            return

        self.workspace = workspace
        self._staging = {
            dtype: workspace.buffer.view(dtype) for dtype in SUPPORTED_DTYPES
        }
        self.disabled = False
        logger.info(
            "XpuSymmMemCommunicator: enabled on %s (world_size=%d, max_bytes=%d).",
            device,
            self.world_size,
            self.max_bytes,
        )

    def _compile_kernel(self, workspace: SymmMemWorkspace) -> bool:
        """Build the Triton specialization up front, so a build failure becomes
        a fallback here instead of an error on a forward pass. Reads only
        buffers rendezvous already mapped, so no barrier is involved."""
        from sglang.srt.hardware_backend.xpu.kernels.comm.one_shot_all_reduce import (
            one_shot_all_reduce,
        )

        try:
            probe = torch.empty(64, dtype=torch.bfloat16, device=self.device)
            one_shot_all_reduce(workspace.peer_ptrs, probe, self.world_size)
            torch.xpu.synchronize()
        except Exception as e:
            logger.warning(
                "XpuSymmMemCommunicator: one-shot kernel failed to build (%s); "
                "falling back to the default all-reduce.",
                e,
            )
            return False
        return True

    def should_torch_symm_mem_allreduce(self, inp: torch.Tensor) -> bool:
        """Whether the one-shot path can take this tensor."""
        if self.disabled:
            return False
        if inp.device != self.device:
            return False
        if inp.dtype not in self._staging:
            return False
        if not inp.is_contiguous():
            return False
        inp_bytes = inp.numel() * inp.element_size()
        if inp_bytes % 4 != 0:
            return False
        return inp_bytes <= self.max_bytes

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Sum-all-reduce ``inp`` by reading every rank's buffer once.

        Returns ``None`` for an ineligible input, which sends the caller to the
        default collective. ``out=inp`` reduces in place: peers read the staging
        copy, not ``out``.
        """
        from sglang.srt.hardware_backend.xpu.kernels.comm.one_shot_all_reduce import (
            one_shot_all_reduce,
        )

        if not self.should_torch_symm_mem_allreduce(inp):
            return None
        if out is None:
            out = torch.empty_like(inp)

        numel = inp.numel()
        self._staging[inp.dtype][:numel].copy_(inp.view(-1))
        # The first barrier publishes every rank's input before anyone reads its
        # peers; the second stops the next call's staging write from racing a
        # peer that is still reading this one.
        self.workspace.barrier()
        one_shot_all_reduce(self.workspace.peer_ptrs, out.view(-1), self.world_size)
        self.workspace.barrier()
        return out
