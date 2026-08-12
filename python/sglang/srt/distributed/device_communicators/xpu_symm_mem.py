"""Symmetric-memory all-reduce for Intel XPU."""

import logging
import socket
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.environ import envs
from sglang.srt.utils import is_xpu

logger = logging.getLogger(__name__)

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    _import_error: Optional[BaseException] = None
except ImportError as e:  # pragma: no cover
    torch_symm_mem = None
    _import_error = e


class XpuSymmMemCommunicator:
    """One-shot all-reduce over torch symmetric memory on Intel XPU.

    Mirrors the ``TorchSymmMemCommunicator`` call contract so GroupCoordinator's
    dispatch needs no device special case. It is a separate class because that
    one is built on ``symm_mem::{multimem,two_shot}_all_reduce_``, which
    torch-xpu-ops does not register, and disables itself when
    ``multicast_ptr == 0`` -- always the case on Xe. Only the allocator half of
    the backend exists for XPU, hence the Triton reduce kernel.
    """

    # Peer mapping is intra-node only.
    _MAX_WORLD_SIZE = 8

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        self.disabled = True
        self.buffer = None
        self.handle = None
        self.max_bytes = 0
        self._staging = {}
        # Assigned before the guards below: a disabled communicator still gets
        # introspected (e.g. group.group_name by model-side custom-AR probes).
        if isinstance(device, int):
            device = torch.device(f"xpu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(group)

        if not is_xpu():
            return
        if torch_symm_mem is None:
            logger.warning(
                "XpuSymmMemCommunicator: torch symmetric memory is unavailable "
                "(%s); falling back to the default all-reduce.",
                _import_error,
            )
            return

        # Lazy: the reduce kernel pulls in Triton.
        from sglang.srt.hardware_backend.xpu.kernels.comm.one_shot_all_reduce import (
            SUPPORTED_DTYPES,
            pack_peer_ptrs,
        )

        backend = torch_symm_mem.get_backend(device)
        if backend != "XPU":
            logger.warning(
                "XpuSymmMemCommunicator: symmetric-memory backend for %s is %s, "
                "expected 'XPU'; falling back to the default all-reduce.",
                device,
                backend,
            )
            return
        if self.world_size > self._MAX_WORLD_SIZE:
            logger.warning(
                "XpuSymmMemCommunicator: world size %d exceeds the supported "
                "maximum %d, communicator is not available.",
                self.world_size,
                self._MAX_WORLD_SIZE,
            )
            return
        if not self._is_single_node(group):
            logger.warning(
                "XpuSymmMemCommunicator: group spans multiple hosts, "
                "communicator is not available (peer mapping is intra-node)."
            )
            return

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

        try:
            self.buffer = torch_symm_mem.empty(
                self.max_bytes, dtype=torch.uint8, device=device
            )
            self.handle = torch_symm_mem.rendezvous(self.buffer, group.group_name)
        except Exception as e:
            # Importing a peer's allocation goes through pidfd_getfd, which needs
            # ptrace permission; restrictive seccomp / yama fails here.
            logger.warning(
                "XpuSymmMemCommunicator: symmetric-memory rendezvous failed (%s); "
                "falling back to the default all-reduce.",
                e,
            )
            self.buffer = None
            self.handle = None
            return

        self.peer_ptrs = pack_peer_ptrs(self.handle.buffer_ptrs, device)
        self._staging = {dtype: self.buffer.view(dtype) for dtype in SUPPORTED_DTYPES}
        if not self._compile_kernel(group):
            self.buffer = None
            self.handle = None
            self._staging = {}
            return

        self.disabled = False
        logger.info(
            "XpuSymmMemCommunicator: enabled on %s (world_size=%d, max_bytes=%d).",
            device,
            self.world_size,
            self.max_bytes,
        )

    def _is_single_node(self, group: ProcessGroup) -> bool:
        hostnames = [None] * self.world_size
        dist.all_gather_object(hostnames, socket.gethostname(), group=group)
        return len(set(hostnames)) == 1

    def _compile_kernel(self, group: ProcessGroup) -> bool:
        """Build the Triton specialization up front.

        A first launch from inside a graph capture would otherwise JIT
        mid-capture, and a build failure becomes a fallback here instead of an
        error on a forward pass. Reads only buffers rendezvous already mapped,
        so no barrier is involved. The verdict is reduced across the group
        because ranks that disagree would diverge on every later all-reduce.
        """
        from sglang.srt.hardware_backend.xpu.kernels.comm.one_shot_all_reduce import (
            one_shot_all_reduce,
        )

        compiled = torch.ones(1, dtype=torch.int32)
        try:
            probe = torch.empty(64, dtype=torch.bfloat16, device=self.device)
            one_shot_all_reduce(self.peer_ptrs, probe, self.world_size)
            torch.xpu.synchronize()
        except Exception as e:
            logger.warning(
                "XpuSymmMemCommunicator: one-shot kernel failed to build (%s); "
                "falling back to the default all-reduce.",
                e,
            )
            compiled.zero_()
        dist.all_reduce(compiled, op=dist.ReduceOp.MIN, group=group)
        return bool(compiled.item())

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
        self.handle.barrier()
        one_shot_all_reduce(self.peer_ptrs, out.view(-1), self.world_size)
        self.handle.barrier()
        return out
